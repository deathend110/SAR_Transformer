# 单比特 SAR 图像恢复 — 基于 SwinIR 的时序扩展实验设计

## 一、问题定义

### 1.1 数据特性

| 属性 | 值 |
|---|---|
| 输入序列 | 12 帧连续 SAR 图像，H/L/H 交替 |
| 帧间偏移 | 固定 32 像素（已知，无需光流） |
| 单帧尺寸 | 256×256，单通道 |
| 任务 | 从 H/L 交替帧序列恢复高质量 SAR 图像 |
| 输出 | 单帧（中间帧）或完整序列 |

### 1.2 数据预处理

所有方案的前置条件：data 层按已知 32px 偏移将全部 12 帧对齐到同一地理坐标系。

```
原始: F₀(x,y), F₁(x+32,y), F₂(x+64,y), ..., F₁₁(x+352,y)
对齐: F₀(x,y), F₁(x,y),   F₂(x,y),   ..., F₁₁(x,y)
      ↑ 12 帧的 (x,y) 对应地面上同一位置
```

### 1.3 数据流

```python
# DataLoader 输出:
#   'L': (12, 256, 256)  — 对齐后的输入帧堆叠
#   'H': (1, 256, 256)   — 中间帧真值
```

数据增强：随机翻转/旋转 + 乘性斑点噪声 `I * exp(N(0, σ²))`

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

SwinIR 的 token = **窗口内的像素**，不是帧。8×8 窗口内 64 个像素互看，只关注帧内空间关系。

---

## 三、方案设计：四种递进方法

### 3.1 Baseline：通道堆叠

```
12 帧对齐后沿通道堆叠: (12, 256, 256)
   → SwinIR(in_chans=12, upscale=1, upsampler=null)

Attention 范围: 同帧内 8×8 窗口 (64 token)
帧间关系: 仅在 conv_first 的 3×3 卷积中混合 (1 个像素的 12 个通道混合)
          无帧间 attention
```

| 改动                  | 状态      |
| ------------------- | ------- |
| `network_swinir.py` | **0 行** |
| 仅改配置 `in_chans=12`  |         |

**作用：** 跑通实验管线，建立指标下界。

---

### 3.2 方案 A：3D 统一窗口

将 SwinIR 的 2D 窗口扩展为 3D (T, H, W)，帧内和帧间关系在同一个注意力矩阵中联合建模。

#### 结构

```
输入 (B, 12, 256, 256) → reshape → (B, 1, 12, 256, 256)
   │
   ├─ conv_first: Conv2d(1, embed_dim, 3, 1, 1)
   │     逐帧独立，不跨帧混合。reshape 后逐帧处理:
   │     x: (B×12, 1, 256, 256) → Conv2d → (B×12, embed_dim, 256, 256)
   │     → reshape → (B, embed_dim, 12, 256, 256)
   │
   ├─ RSTB × 6:
   │   ├─ Patch Embed: (B, embed_dim, T, H, W) → (B, T×H×W, embed_dim)
   │   ├─ 3D WindowPartition: window_size=(12,8,8)
   │   │   每个窗口 shape: (nW×B, 12×8×8, embed_dim) = (nW×B, 768, embed_dim)
   │   │   token 排列:
   │   │     [F₀ₚ₍₀,₀₎ ... F₀ₚ₍₇,₇₎,     ← 第0帧 64 token
   │   │      F₁ₚ₍₀,₀₎ ... F₁ₚ₍₇,₇₎,     ← 第1帧 64 token
   │   │      ...
   │   │      F₁₁ₚ₍₀,₀₎ ... F₁₁ₚ₍₇,₇₎]   ← 第11帧 64 token
   │   │
   │   ├─ 3D WindowAttention(12,8,8):
   │   │   ├─ QKV: Linear(embed_dim → embed_dim×3)
   │   │   ├─ 相对位置偏置表: (23, 15, 15) × num_heads
   │   │   │   Δt=0, Δh=1, Δw=0 → 帧内空间邻域
   │   │   │   Δt=1, Δh=0, Δw=0 → 相邻帧同位置
   │   │   │   Δt=2, Δh=0, Δw=0 → 隔帧同位置
   │   │   │   Δt=1, Δh=1, Δw=1 → 相邻帧空间邻域
   │   │   ├─ 768×768 自注意力矩阵
   │   │   ├─ cyclic shift: (0,0,0)/(6,4,4)
   │   │   └─ 同时建模帧内空间 + 帧间时序
   │   └─ Patch UnEmbed + 残差
   │
   ├─ conv_after_body + 全局残差
   │
   └─ conv_last: Conv2d(embed_dim, 1, 3, 1, 1)
          逐帧独立
      → 输出 (B, 1, 256, 256)
```

#### 改动

| 文件 | 改动 | 行数 |
|---|---|---|
| `models/network_swinir.py` | `window_partition/reverse` → 3D | +16 |
| `models/network_swinir.py` | `WindowAttention` 位置偏置 → 3D | +15 |
| `models/network_swinir.py` | `SwinIR.__init__/forward` | +25 |
| `models/network_swinir.py` | `check_image_size` → 含 T 轴 | +5 |
| `data/dataset_sar_1bit.py` | 新建数据集 | ~80 |
| `options/swinir/train_swinir_sar_3d.json` | 新建配置 | ~70 |

#### 优缺点

| 优点 | 缺点 |
|---|---|
| 改动集中在一个文件 | 无法加载 SwinIR 预训练权重 |
| 帧内/帧间联合优化 | attention 从 64→768，计算量↑ |
| 像素级时序建模 | 帧间 H/L 交替模式需自主学习 |
| 结构简洁 | 调试难度高 |

---

### 3.3 方案 B：共享 SwinIR + 时序 Transformer

SwinIR 原封不动作为空间 backbone，帧内和帧间由不同模块分工。

#### 结构

```
12 帧 → 共享 SwinIR backbone (去掉 conv_last, 0 行修改)
  每帧独立通过, 参数共享:
    F₀ → SwinIR → feat₀ (B, embed_dim, 64, 64)
    F₁ → SwinIR → feat₁ (B, embed_dim, 64, 64)
    ...
    F₁₁→ SwinIR → feat₁₁(B, embed_dim, 64, 64)
  堆叠: (B, embed_dim, 12, 64, 64)
   │
   ├─ 时序模块 (新):
   │   ├─ Patch Embed: (B, embed_dim, T, H', W') → (B, H'×W', T, embed_dim)
   │   │
   │   ├─ TemporalBlock × Nt (建议 2~4):
   │   │   ├─ Temporal WindowAttention(12, 1, 1)
   │   │   │   token: [feat₀(h,w), feat₁(h,w), ..., feat₁₁(h,w)]
   │   │   │          ↑ 同一空间位置, 12 帧全连接自注意力
   │   │   │          1D 相对位置偏置 (23, num_heads)
   │   │   │          cyclic shift 沿 T 轴
   │   │   │          → 纯帧间 attention
   │   │   ├─ LayerNorm + MLP + 残差
   │   │   └─ 每个空间位置独立做 (12×12 矩阵)
   │   │
   │   └─ Patch UnEmbed: (B, H'×W', T, embed_dim) → (B, embed_dim, T, H', W')
   │
   └─ conv_last: Conv3d(embed_dim, 1, (1,3,3), padding=(0,1,1))
        → 输出 (B, 1, 256, 256)
```

#### 时序模块的两种设计

| 选项 | token 数/序列 | attention 矩阵 | 粒度 |
|---|---|---|---|
| **B1 (帧级)** | 12 (GAP 压缩每帧) | 12×12 | 帧级语义 |
| **B2 (特征位置级, 推荐)** | H'×W' 个独立序列, 每序列 12 | 每个位置 12×12 | 像素级 |

**推荐 B2**：在 backbone 下采样的特征图 (64×64) 上，每个空间位置独立做 12 帧 attention，保留像素级精度。

#### 和 VRT 的关系

| 组件 | VRT | 方案 B |
|---|---|---|
| 空间编码 | Conv + 多尺度 3D Swin | SwinIR backbone (预训练) |
| 光流 | SpyNet | ❌ 不需要 (已知偏移) |
| 可变形卷积 | Modulated DeformConv | ❌ 不需要 |
| 帧间 attention | Mutual attention (半窗口交叉) | **Self-attention (全序列全连接)** |
| 多尺度 | U-Net encoder-decoder | ❌ 单尺度 |
| 时序窗口 | (6,8,8) 混合时空 | **(12,1,1) 纯时序** |

方案 B 只借用了 VRT 将 `WindowAttention` 扩展到 3D 空间的**编码方法**，但去掉了 flow、warping、mutual attention 和 U-Net，因为 SAR 场景不需要这些。

#### 改动

| 文件 | 改动 | 行数 |
|---|---|---|
| `models/network_swinir_temporal.py` | 新建时序模块 | ~120 |
| `data/dataset_sar_1bit.py` | 新建数据集 | ~80 |
| `options/swinir/train_swinir_sar_temporal.json` | 新建配置 | ~70 |
| `network_swinir.py` | **0 行** | 0 |

#### 优缺点

| 优点 | 缺点 |
|---|---|
| SwinIR 完全不变，可加载预训练 | backbone 每帧独立，无法在 attention 中利用帧间信息 |
| 帧内/帧间模块分工明确，收敛快 | 时序融合粒度受限于特征图分辨率 |
| 时序模块轻量 (每个 12×12) | 多了一次特征图重排 |
| 调试友好，可分别测试 | |

---

### 3.4 方案 C：融合方案 (推荐)

SwinIR 原封不动 + 时序模块改用方案 A 的 3D 窗口注意力 (运行在特征图上)。

#### 结构

```
12 帧 → 共享 SwinIR backbone (同方案 B)
  每帧输出 (B, embed_dim, 64, 64), 堆叠 → (B, embed_dim, 12, 64, 64)
   │
   ├─ 3D Temporal Block (新, 复用方案 A 的 3D attention):
   │   ├─ 3D WindowAttention(12, 8, 8)   ← 和方案 A 相同
   │   │   窗口内: 12×8×8 = 768 token
   │   │   3D 相对位置偏置 (23, 15, 15)
   │   │   cyclic shift (0,0,0)/(6,4,4)
   │   │   运行在特征图上 (已下采样 4×)
   │   ├─ LayerNorm + MLP + 残差
   │   └─ × Nt (建议 2)
   │
   └─ conv_last: Conv3d → 输出
```

#### 为什么和方案 A 不重复

| | A: 3D 统一窗口 | C: 融合方案 |
|---|---|---|
| 3D attention 位置 | 像素空间 (256×256) | 特征空间 (64×64) |
| SwinIR 代码 | 需要改 | **不改** |
| 计算量 | (B, 1024个窗口, 768) | (B, 64个窗口, 768) |
| 预训练权重 | 不可用 | 可加载 |
| backbone 可替换 | 固定 3D SwinIR | 可换任意 backbone |

方案 C 本质 = 方案 B 的结构化设计 + 方案 A 的 attention 粒度，但运行在 backbone 下采样后的特征图上，计算量可控。

---

## 四、四种方案对比

| 维度 | Baseline (通道堆叠) | A: 3D 统一窗口 | B: 时序 Transformer | C: 融合方案 |
|---|---|---|---|---|
| 帧内 attention | ✅ 8×8 窗口 | ✅ (12,8,8) 窗口内 | ✅ SwinIR 独立做完 | ✅ SwinIR 独立做完 |
| 帧间 attention | ❌ 无 | ✅ 同窗口内 | ✅ (12,1,1) 纯时序 | ✅ (12,8,8) 特征空间 |
| 帧间方式 | — | 全连接 768² | 每个位置 12² | 全连接 768² |
| SwinIR 修改 | 0 行 | ~60 行 | **0 行** | **0 行** |
| 可加载预训练 | ✅ | ❌ | ✅ | ✅ |
| 注意力粒度 | 像素 64² | 像素 768² | 特征位置 12² | 特征 768² |
| 计算量 | 低 | 高 | 低 | 中 |
| 调试难度 | 低 | 高 | 中 | 中 |

---

## 五、实验优先级

```
P0: Baseline 通道堆叠     → 跑通管线，指标下界
P1: 方案 C 融合方案       → 主方案
P2: 方案 B 时序窗口       → 消融：时序 attention 粒度的贡献
P3: 方案 A 3D 统一窗口   → 消融：端到端 3D vs 分离式
```

---

## 六、文件清单

```
KAIR/
├── data/
│   ├── dataset_sar_1bit.py                [新]
│   └── select_dataset.py                  [改]
├── models/
│   ├── network_swinir.py                  [改: 仅方案A]
│   └── network_swinir_temporal.py         [新: 方案B/C]
├── options/swinir/
│   ├── train_swinir_sar_1bit.json         [新: baseline]
│   ├── train_swinir_sar_3d.json           [新: 方案A]
│   └── train_swinir_sar_temporal.json     [新: 方案B/C]
└── main_train_psnr.py                     [不变]
```
