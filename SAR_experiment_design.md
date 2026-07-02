# 单比特 SAR 图像恢复 — 基于 SwinIR 的时序扩展实验设计

## 一、问题定义

### 1.1 数据特性

| 属性 | 值 |
|---|---|
| 输入序列 | 9 帧 **1-bit SAR** 数据，成像质量 H/L/H 交替 |
| 帧间偏移 | 固定 128 像素（已知，无需光流） |
| 单帧尺寸 | 512×512，单通道 |
| 成像机制 | 512×512 成像窗口，每次步进 128px 滑动采集 |
| 输入 | 1-bit 量化后的 SAR 图像（H/L 表示 1-bit 采样质量） |
| 真值 | 全精度 SAR 数据 |
| 任务 | 从 1-bit SAR 序列恢复全精度 SAR 图像（1-bit 信号恢复 + 质量不均匀补偿） |
| 输出 | 全部 9 帧对应的全精度恢复结果 |

### 1.2 序列结构

512px 成像窗口以 128px 步进滑动，9 帧的地理覆盖和 1-bit 采样质量如下：

```
地面: [0,512] 全精度H ── [512,1024] 全精度L ── [1024,1536] 全精度H

F₀ [0,    512  ]  → 全部 H 区 → 1-bit 采样质量最高  ← 信息最丰富
F₁ [128,  640  ]  → H[128,512]+L[512,640]     → 混合 (384H+128L)
F₂ [256,  768  ]  → H[256,512]+L[512,768]     → 混合 (256H+256L)
F₃ [384,  896  ]  → H[384,512]+L[512,896]     → 混合 (128H+384L)
F₄ [512,  1024 ]  → 全部 L 区 → 1-bit 采样质量最低  ← 信息最匮乏
F₅ [640,  1152 ]  → L[640,1024]+H[1024,1152]  → 混合 (384L+128H)
F₆ [768,  1280 ]  → L[768,1024]+H[1024,1280]  → 混合 (256L+256H)
F₇ [896,  1408 ]  → L[896,1024]+H[1024,1408]  → 混合 (128L+384H)
F₈ [1024, 1536 ]  → 全部 H 区 → 1-bit 采样质量最高  ← 信息最丰富
```

**恢复目标：** 全部 9 帧，每帧都是 1-bit → 全精度的恢复。
**核心挑战：** 1-bit 量化带来的信息丢失 + 滑窗造成的非均匀质量分布。

### 1.3 关键洞察

- F₀ 和 F₄ 地理范围完全不重叠（[0,512] vs [512,1024]）。F₄ 的任意像素没有其他帧提供同位置的高质量 1-bit 参考。
- 相邻帧共享 384px 重叠区，形成渐进式质量退化/恢复链。
- 时序 attention 的作用：F₄ 的 1-bit 数据最差，但相邻帧 F₃/F₅ 在重叠区域有较好的 1-bit 样本，attention 可跨帧借用信息。
- H/L mask 编码的是 **1-bit 采样质量**，作为 attention 的置信度提示。

---

## 二、SwinIR 原生架构回顾

### 2.1 三部分设计

```
输入 (B, C, H, W)
   │
   ├─ conv_first: Conv2d(C, embed_dim, 3, 1, 1)
   │
   ├─ RSTB × N:
   │   ├─ Patch Embed: (B, C, H, W) → (B, L, C)
   │   ├─ WindowAttention(8,8):
   │   │   → 窗口内 64 token 自注意力 + 2D 相对位置偏置
   │   │   → cyclic shift: 交替 (0,0) / (4,4)
   │   └─ Patch UnEmbed → 残差连接
   │
   ├─ conv_after_body + 全局残差
   │
   └─ conv_last: Conv2d(embed_dim, out_chans, 3, 1, 1)
```

### 2.2 当前 token 定义

SwinIR 的 token = **窗口内的像素**。8×8 窗口内 64 个像素互看，只关注帧内空间关系，不处理帧间时序。

---

## 三、核心设计思路

### 3.1 三种方案的血缘关系

```
SwinIR (2D window)
    │
    ├── 扩展窗口到 T 轴 ───────────────── 方案 A: 3D 统一窗口
    │
    └── 拆分为两个阶段 ──┬── SwinIR 不动 (空间特征提取) ── 方案 B: 时序 Transformer
                         │
                         └── + 复用 3D attention 做时序融合 ── 方案 C: 融合方案 (推荐)
```

### 3.2 关键共识

1. **conv_first 逐帧独立**：`Conv2d` 逐帧处理再 reshape，不跨帧混合
2. **T=5 滑动窗口**：奇数帧数提供对称的时序上下文，5×128=640px 覆盖宽度
3. **H/L mask 辅助输入**：为每帧每个像素编码 1-bit 采样质量，作为额外通道输入
4. **全部 9 帧统一回归全精度**：无需区分帧权重，全帧参与 loss 计算

### 3.3 T=5 滑动窗口的物理意义

9 帧、T=5、步进 1，共 5 个窗口：

```
窗口 0: [F₀, F₁, F₂, F₃, F₄]  → 地面 [0, 1024]   含 H→L 过渡
窗口 1: [F₁, F₂, F₃, F₄, F₅]  → 地面 [128, 1152]  L 区居中
窗口 2: [F₂, F₃, F₄, F₅, F₆]  → 地面 [256, 1280]  纯 L 居中
窗口 3: [F₃, F₄, F₅, F₆, F₇]  → 地面 [384, 1408]  L→H 过渡
窗口 4: [F₄, F₅, F₆, F₇, F₈]  → 地面 [512, 1536]  含 L→H 过渡
```

每个窗口内 5 帧的 3D window 内，Δt ∈ {-2, -1, 0, 1, 2}，相对位置偏置对称分布。

**注意：** 3D 窗口内 (T=5, 8, 8) 的 token 不是"同一位置的 5 次观测"，而是 **"相隔 128px 的 5 个不同位置"**。Attention 学的是不同空间位置的纹理迁移模式——如何将 H 区丰富的纹理结构迁移到 L 区去恢复缺失的信息。

---

## 四、方案设计：四种递进方法

### 4.1 Baseline：通道堆叠

```
9 帧沿通道堆叠: (9, 512, 512)
   → SwinIR(in_chans=9, upscale=1)

Attention 范围: 帧内 8×8 窗口 (64 token)
帧间关系: 仅在 conv_first 的 3×3 conv 中混合
```

| 改动 | 状态 |
|---|---|
| `network_swinir.py` | **0 行** |
| 仅改配置 `in_chans=9` | |

**作用：** 跑通管线，建立指标下界。

---

### 4.2 方案 A：3D 统一窗口

将 SwinIR 的 2D 窗口扩展为 3D (T=5, H=8, W=8)，帧内和帧间关系在同一个注意力矩阵中联合建模。

#### 结构

```
输入: 9 帧 1-bit 图像 (B, 9, 512, 512)
        + 9 帧 H/L mask (B, 9, 512, 512)
        → concat → (B, 18, 512, 512)
   │
   ├─ conv_first: Conv2d(18, embed_dim, 3, 1, 1)
   │     逐帧独立: reshape → (B×9, 18, 512, 512) → Conv2d → (B×9, embed_dim, ...)
   │     → reshape → (B, embed_dim, 9, 512, 512)
   │
   ├─ RSTB × 6:
   │   ├─ Patch Embed → (B, 9×H×W, embed_dim)
   │   ├─ T=5 滑动窗口处理:
   │   │   对序列做 T=5 滑窗, 步进 1, 共 5 个窗口
   │   │   每个窗口: window_partition(5,8,8) → (B×nW, 320, embed_dim)
   │   ├─ 3D WindowAttention(5,8,8):
   │   │   ├─ 相对位置偏置表: (9, 15, 15) × num_heads
   │   │   │   Δt∈{-2,-1,0,1,2}, Δh∈{-7,...,7}, Δw∈{-7,...,7}
   │   │   ├─ 320 token 自注意力
   │   │   ├─ cyclic shift: (0,0,0)/(2,4,4)
   │   │   └─ 同时建模帧内空间 + 帧间纹理迁移
   │   └─ Patch UnEmbed + 残差
   │
   ├─ conv_after_body + 全局残差
   │
   └─ conv_last: Conv2d(embed_dim, 1, 3, 1, 1)
          逐帧独立
      → 输出 (B, 9, 512, 512)  全精度恢复结果
```

#### 改动

| 文件 | 改动 | 行数 |
|---|---|---|
| `models/network_swinir.py` | `window_partition/reverse` → 3D | +16 |
| `models/network_swinir.py` | `WindowAttention` 位置偏置 → 3D | +15 |
| `models/network_swinir.py` | `SwinIR.__init__/forward`, 滑窗逻辑 | +40 |
| `models/network_swinir.py` | `check_image_size` → 含 T 轴 | +5 |
| `data/dataset_sar_1bit.py` | 新建数据集 + mask 生成 | ~100 |
| `options/swinir/train_swinir_sar_3d.json` | 新建配置 | ~70 |

**提前注意：** 512×512 下 8×8 窗口数 = 4096 个/帧，对 5 窗口 = 20480 个 3D 窗口，计算量大。

---

### 4.3 方案 B：共享 SwinIR + 时序 Transformer

SwinIR 原封不动作为空间 backbone，帧内和帧间由不同模块分工。

#### 结构

```
输入: 9 帧 1-bit 图像 + 9 帧 H/L mask (B, 18, 512, 512)
   │
   ├─ SwinIR backbone (去掉 conv_last, 0 行修改):
   │     conv_first: Conv2d(18→embed_dim, 3)
   │     RSTB × 6 → 输出特征图
   │     每帧独立通过 (参数共享):
   │       feat₀, feat₁, ..., feat₈  ← 每帧 (B, embed_dim, 128, 128)
   │     堆叠: (B, embed_dim, 9, 128, 128)
   │
   ├─ 时序模块 (新文件, network_swinir_temporal.py):
   │   ├─ T=5 滑动窗口处理
   │   ├─ TemporalBlock × Nt (建议 4~6):
   │   │   ├─ Temporal WindowAttention(5, 1, 1)
   │   │   │   token: [featₜ₋₂(h,w), ..., featₜ₊₂(h,w)]
   │   │   │   同一空间位置, 5 帧全连接自注意力
   │   │   │   1D 相对位置偏置 (9, num_heads)
   │   │   │   每个空间位置独立做 5×5 矩阵
   │   │   ├─ LayerNorm + GEGLU MLP + 残差
   │   │   └─ cyclic shift 沿 T 轴
   │   └─ 输出: (B, embed_dim, 9, 128, 128)
   │
   └─ conv_last: Conv2d(embed_dim, 1, 3, 1, 1)
          逐帧独立
      → 输出 (B, 9, 512, 512)
```

#### 和 VRT 的关系

| 组件           | VRT                      | 方案 B                        |
| ------------ | ------------------------ | --------------------------- |
| 空间编码         | Conv + 多尺度 3D Swin       | SwinIR backbone (可预训练)      |
| 光流           | SpyNet                   | ❌ 不需要                       |
| 可变形卷积        | Modulated DeformConv     | ❌ 不需要                       |
| 帧间 attention | Mutual attention (半窗口交叉) | **Self-attention (全序列全连接)** |
| 多尺度          | U-Net encoder-decoder    | ❌ 单尺度                       |
| 时序窗口         | (6,8,8) 混合时空             | **(5,1,1) 纯时序**             |

#### 改动

| 文件 | 改动 | 行数 |
|---|---|---|
| `models/network_swinir_temporal.py` | 新建时序模块 | ~150 |
| `data/dataset_sar_1bit.py` | 新建数据集 + mask 生成 | ~100 |
| `options/swinir/train_swinir_sar_temporal.json` | 新建配置 | ~70 |
| `network_swinir.py` | **0 行** (仅改 in_chans) | 0 |

---

### 4.4 方案 C：融合方案 (推荐)

SwinIR 原封不动 + 时序模块改用方案 A 的 3D 窗口注意力（运行在特征图上）。

#### 结构

```
输入: 9 帧 1-bit 图像 + 9 帧 H/L mask (B, 18, 512, 512)
   │
   ├─ SwinIR backbone (同方案 B, 0 行修改):
   │     输出: (B, embed_dim, 9, 128, 128)
   │
   ├─ 3D Temporal Block (新, 复用 3D attention):
   │   ├─ T=5 滑动窗口
   │   ├─ 3D WindowAttention(5, 8, 8)   ← 和方案 A 相同
   │   │   窗口内: 5×8×8 = 320 token
   │   │   3D 相对位置偏置 (9, 15, 15)
   │   │   cyclic shift (0,0,0)/(2,4,4)
   │   │   特征图上 (128×128) 共 16²=256 个窗口/帧
   │   │   5 个窗口总计 = 1280 个 3D 窗口
   │   ├─ LayerNorm + GEGLU MLP + 残差
   │   └─ × Nt (建议 4)
   │
   └─ conv_last: Conv2d(embed_dim, 1, 3, 1, 1)
          逐帧独立
      → 输出 (B, 9, 512, 512)
```

#### 为什么推荐方案 C

| | A: 3D 统一窗口 | C: 融合方案 |
|---|---|---|
| 3D attention 位置 | 像素空间 (512×512) | 特征空间 (128×128) |
| SwinIR 代码 | 需要改 ~76 行 | **不改** |
| 窗口数/帧 | (512/8)² = 4096 | (128/8)² = 256 |
| 总 3D 窗口数 | 4096×5 = 20480 | 256×5 = 1280 |
| 预训练权重 | 不可用 | 可加载 |

方案 C = 方案 B 的结构化设计 + 方案 A 的 attention 粒度，特征图下采样 4×使计算量可控。

---

## 五、H/L Mask 辅助输入设计

### 5.1 含义

Mask 编码的不是真值质量，而是 **1-bit 采样质量**——告诉网络哪些区域有更丰富的 1-bit 信息，哪些区域信息丢失严重。本质是 attention 的置信度提示。

### 5.2 生成规则

每个像素的 mask 值由该帧成像窗口覆盖地面坐标 x 决定：

```python
def generate_mask(frame_idx, img_size=512, stride=128):
    x_start = frame_idx * stride
    mask = torch.zeros(img_size, img_size)
    for x in range(img_size):
        ground_x = x_start + x
        if ground_x < img_size:
            mask[:, x] = 1.0
        elif ground_x < 2 * img_size:
            mask[:, x] = 0.0
        else:
            mask[:, x] = 1.0
    return mask
```

### 5.3 输入拼接

```
所有方案: conv_first.in_chans = 18  = 9 帧图像 + 9 帧 mask
```

---

## 六、Loss 设计

全部 9 帧统一回归全精度真值，无需区分帧权重：

```python
loss = L1_loss(output, ground_truth)   # output/gt shape: (B, 9, 512, 512)
```

F₀/F₈ 虽然 1-bit 质量高，但与全精度真值仍有差距，同样需要学习 1-bit→全精度的映射。

---

## 七、四种方案对比

| 维度 | Baseline | A: 3D 统一窗口 | B: 时序 Transformer | C: 融合方案 |
|---|---|---|---|---|
| 帧内 attention | ✅ 8×8 | ✅ (5,8,8) 内 | ✅ SwinIR 独立做完 | ✅ SwinIR 独立做完 |
| 帧间 attention | ❌ 无 | ✅ 同窗口 (5,8,8) | ✅ (5,1,1) 纯时序 | ✅ (5,8,8) 特征空间 |
| 帧间方式 | — | 全连接 320² | 每个位置 5² | 全连接 320² (特征图) |
| T 轴滑动窗口 | 无 | ✅ T=5 步进1 | ✅ T=5 步进1 | ✅ T=5 步进1 |
| H/L mask 输入 | 可选 | ✅ 推荐 | ✅ 推荐 | ✅ 推荐 |
| SwinIR 修改 | 0 行 | ~76 行 | **0 行** | **0 行** |
| 可加载预训练 | ✅ | ❌ | ✅ | ✅ |
| 计算量 (相对) | 低 (~1×) | 高 (~6×) | 中 (~2×) | 中 (~2.5×) |
| 调试难度 | 低 | 高 | 中 | 中 |

---

## 八、实验优先级

```
P0: Baseline 通道堆叠     → 跑通管线，指标下界
P1: 方案 C 融合方案       → 主方案
P2: 方案 B 时序窗口       → 消融：时序 attention 粒度的贡献
P3: 方案 A 3D 统一窗口    → 消融：端到端 3D vs 分离式
```

---

## 九、文件清单

```
KAIR/
├── data/
│   ├── dataset_sar_1bit.py                [新] 数据集 + H/L mask 生成
│   └── select_dataset.py                  [改] 注册新数据集
├── models/
│   ├── network_swinir.py                  [改: 仅方案A, ~76行]
│   └── network_swinir_temporal.py         [新: 方案B/C 时序模块 + 3D attention 复用]
├── options/swinir/
│   ├── train_swinir_sar_1bit.json         [新: baseline]
│   ├── train_swinir_sar_3d.json           [新: 方案A]
│   └── train_swinir_sar_temporal.json     [新: 方案B/C]
└── main_train_psnr.py                     [不变]
```
