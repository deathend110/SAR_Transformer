# 单比特 SAR 雷达图像恢复 — SwinIR 实验指南

## 1. 项目结构概览

```
SAR_Transformer/
├── KAIR/                          # 深度学习训练工具箱 (cszn/KAIR)
│   ├── main_train_psnr.py         # PSNR 训练入口脚本 ← 你的训练起点
│   ├── main_test_swinir.py        # SwinIR 测试脚本
│   ├── models/
│   │   ├── network_swinir.py      # SwinIR 网络架构定义（核心）
│   │   ├── model_plain.py         # PSNR 训练模型（训练循环、loss、优化器）
│   │   ├── model_base.py          # 模型基类（设备、DDP、保存/加载）
│   │   ├── select_model.py        # 模型调度（"plain" → ModelPlain）
│   │   ├── select_network.py      # 网络调度（"swinir" → SwinIR）
│   │   ├── loss.py / loss_ssim.py # 损失函数
│   ├── data/
│   │   ├── select_dataset.py      # 数据集调度
│   │   ├── dataset_sr.py          # 超分数据集
│   │   ├── dataset_dncnn.py       # 去噪数据集（在线加噪）
│   │   ├── dataset_blindsr.py     # 盲超分数据集（BSRGAN退化）
│   │   ├── dataset_jpeg.py        # JPEG压缩伪影数据集
│   ├── options/swinir/            # SwinIR 配置 JSON 文件
│   │   ├── train_swinir_sr_classical.json      # 经典超分
│   │   ├── train_swinir_sr_lightweight.json    # 轻量超分
│   │   ├── train_swinir_denoising_gray.json    # 灰度去噪
│   │   ├── train_swinir_denoising_color.json   # 彩色去噪
│   │   └── train_swinir_car_jpeg.json          # JPEG压缩伪影
│   ├── utils/
│   │   ├── utils_option.py        # JSON配置解析、断点续训
│   │   ├── utils_image.py         # 图像I/O、增强、PSNR/SSIM计算
│   │   ├── utils_dist.py          # 分布式训练
│   │   └── utils_model.py         # 模型工具
│   └── docs/README_SwinIR.md      # SwinIR 文档
│
└── SwinIR/                        # 原始 SwinIR 仓库（仅测试推理）
    ├── models/network_swinir.py   # 与 KAIR 中完全相同的网络定义
    ├── main_test_swinir.py        # 测试推理脚本
    ├── utils/util_calculate_psnr_ssim.py
    ├── model_zoo/                 # 预训练权重说明
    ├── testsets/                  # 标准测试集（Set5, Set14 等）
    └── README.md
```

**关键结论：** 两个文件夹共享**完全相同的** `network_swinir.py` 模型定义。**KAIR 是训练框架**（包含完整的训练管道），**SwinIR 文件夹仅用于测试/推理**。你的训练工作将在 KAIR 中进行。

---

## 2. SwinIR 网络架构详解

### 2.1 三部分设计

```
输入图像 (B, C, H, W)
    │
    ├─① 浅层特征提取: conv_first (3×3 conv, in_chans→embed_dim)
    │
    ├─② 深层特征提取: N 个 RSTB (Residual Swin Transformer Block)
    │   └─ 全局残差连接: conv_after_body(deep) + shallow
    │
    └─③ 高质量图像重建（根据任务选择不同上采样器）
```

### 2.2 RSTB（核心构建块）

每个 RSTB 包含：
- **Patch Embed**: `(B, C, H, W) → (B, L, C)`，将特征图转换为 token 序列
- **BasicLayer**: 多个 SwinTransformerBlock 堆叠
  - Window-based Multi-head Self-Attention（交替使用 W-MSA / SW-MSA）
  - 相对位置偏置（可学习参数）
  - 循环移位实现跨窗口交互
- **Conv 残差连接** (`resi_connection`): `1conv`（单层 3×3）或 `3conv`（瓶颈结构）
- 残差连接：`output = patch_embed(conv(layer(x))) + x`

### 2.3 上采样器模式

| upsampler 值 | 用途 | 实现方式 |
|---|---|---|
| `"pixelshuffle"` | 经典超分 | conv_before → 多层 PixelShuffle → conv_last |
| `"pixelshuffledirect"` | 轻量超分 | 单步 conv + PixelShuffle（省参数） |
| `"nearest+conv"` | 真实世界超分 | 最近邻插值 + conv（避免棋盘伪影） |
| `null` / `""` | 去噪/JPEG压缩 | `x + conv_last(deep)`，空间尺寸不变 |

### 2.4 关键可配置参数

| 参数                | 含义                    | 典型值                       |
| ----------------- | --------------------- | ------------------------- |
| `in_chans`        | 输入通道数                 | 1（灰度）、3（彩色）               |
| `out_chans`       | 输出通道数                 | 默认等于 in_chans             |
| `embed_dim`       | 嵌入维度 C                | 60（轻量）、180（标准）、240（大）     |
| `depths`          | 每层 RSTB 的 SwinBlock 数 | [6,6,6,6] 或 [6,6,6,6,6,6] |
| `num_heads`       | 每层注意力头数               | 与 depths 长度一致             |
| `window_size`     | 局部窗口大小                | 8（SR/去噪）、7（JPEG CAR）      |
| `mlp_ratio`       | MLP 隐藏层比率             | 2（SwinIR 默认）              |
| `upscale`         | 上采样倍数                 | 1（去噪/CAR）、2/3/4/8（SR）     |
| `img_range`       | 图像范围                  | 1.0（默认）、255.0（JPEG CAR）   |
| `resi_connection` | 残差连接类型                | `"1conv"` 或 `"3conv"`     |
| `drop_path_rate`  | 随机深度衰减率               | 0.1                       |

---

## 3. 训练管道（KAIR 框架）

### 3.1 配置驱动 (JSON)

所有训练参数通过 JSON 文件指定。框架解析 JSON 后自动完成一切配置。

```bash
# 单卡/DataParallel
python main_train_psnr.py --opt options/swinir/train_swinir_sr_classical.json

# 多卡分布式（推荐）
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 \
    main_train_psnr.py --opt options/swinir/train_swinir_sr_classical.json --dist True
```

### 3.2 数据处理流

```
JSON 配置 → option.parse() → opt 字典
                ↓
define_Dataset(opt['datasets']['train']) → Dataset 类
                ↓
DataLoader → batches of {'L': low_quality, 'H': high_quality, 'L_path': ..., 'H_path': ...}
                ↓
model.feed_data(train_data) → self.L, self.H (移至GPU)
                ↓
model.optimize_parameters() → forward → loss → backward → optimizer.step
                ↓
model.update_learning_rate() → scheduler.step
```

### 3.3 数据集接口

每个数据集返回一个包含以下键的字典：
- `'L'`: 低质量图像 (C×H×W, [0,1] 归一化)
- `'H'`: 高质量真值 (C×H×W, [0,1] 归一化)
- `'L_path'` / `'H_path'`: 图像路径

数据增强：随机裁剪 + 随机翻转/旋转（8种模式）

### 3.4 训练循环 (`main_train_psnr.py`)

```python
for epoch in range(1000000):           # 无限epoch
    for train_data in train_loader:
        current_step += 1
        model.update_learning_rate(current_step)     # 调度器
        model.feed_data(train_data)                   # L → GPU
        model.optimize_parameters(current_step)       # 前向+反向
        # logging / saving / testing per checkpoint intervals
```

### 3.5 基础训练超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| Loss | L1 Loss | 首选，也可选 L2/SSIM/Charbonnier |
| 优化器 | Adam, lr=2e-4 | weight_decay=0 |
| 调度器 | MultiStepLR, gamma=0.5 | 里程碑: [250k, 400k, 450k, 475k, 500k] |
| EMA 衰减 | 0.999 | 指数移动平均 |
| Batch size | 32（4 per GPU × 8 GPUs） | 总 batch |
| 裁剪尺寸 | 48~128 | 取决于任务 |
| 保存/测试间隔 | 每 5000 步 | |
| 日志间隔 | 每 200 步 | |

### 3.6 断点续训

框架自动检测 `models/` 目录下最新的 `*_G.pth`、`*_E.pth`、`*_optimizerG.pth`，从最大迭代数恢复训练。

---

## 4. 单比特 SAR 任务适配方案

### 4.1 最接近的现有任务

| 你的任务 | 最相似的内置任务 | 原因 |
|---|---|---|
| 单比特 SAR 恢复 | **去噪 (Denoising)** | 单比特量化可视为强噪声/信息丢失；upscale=1，空间尺寸不变 |
| 或 | **JPEG 压缩伪影减少** | 单比特量化与 JPEG 类似，都是非线性有损压缩过程 |

### 4.2 推荐配置调整

```json
{
  "task": "swinir_sar_1bit",
  "model": "plain",
  "gpu_ids": [0],
  "scale": 1,
  "n_channels": 1,
  "path": {
    "root": "sar_1bit"
  },
  "netG": {
    "net_type": "swinir",
    "upscale": 1,
    "in_chans": 1,
    "out_chans": 1,
    "img_size": 64,
    "window_size": 8,
    "img_range": 1.0,
    "depths": [6, 6, 6, 6, 6, 6],
    "embed_dim": 180,
    "num_heads": [6, 6, 6, 6, 6, 6],
    "mlp_ratio": 2,
    "upsampler": null,
    "resi_connection": "1conv",
    "init_type": "default"
  },
  "train": {
    "G_lossfn_type": "l1",
    "G_lossfn_weight": 1.0,
    "E_decay": 0.999,
    "G_optimizer_type": "adam",
    "G_optimizer_lr": 2e-4,
    "G_optimizer_wd": 0,
    "G_optimizer_clipgrad": null,
    "G_optimizer_reuse": true,
    "G_scheduler_type": "MultiStepLR",
    "G_scheduler_milestones": [250000, 400000, 450000, 475000, 500000],
    "G_scheduler_gamma": 0.5,
    "G_param_strict": true,
    "E_param_strict": true,
    "checkpoint_test": 5000,
    "checkpoint_save": 5000,
    "checkpoint_print": 200
  }
}
```

### 4.3 需要创建的新文件

1. **`options/swinir/train_swinir_sar_1bit.json`** — 训练配置
2. **`data/dataset_sar_1bit.py`** — 自定义数据集类
   - 输入 L: 单比特 SAR 图像（你的数据）
   - 真值 H: 全精度 SAR 图像
   - 实现原理：数据增强（SAR 特有的斑点噪声模拟 + 几何变换）

### 4.4 自定义数据集类模板

参考 `dataset_dncnn.py`（在线降质）或 `dataset_jpeg.py`（在线压缩），新数据集只需：
- `__init__`: 读取你的 H 图像路径列表
- `__getitem__`:
  1. 读取 H 图像
  2. 随机裁剪
  3. 数据增强
  4. 生成 L（单比特量化模拟）
  5. 返回 `{'L': L, 'H': H, 'L_path': path, 'H_path': path}`

### 4.5 评估指标

- PSNR、SSIM（框架内置，`util.calculate_psnr` / `util.calculate_ssim`）
- 可补充 SAR 专用指标（如等效视数 ENL、边缘保持指数 EPI）

---

## 5. 实验启动步骤

### Step 1: 准备数据
```
trainsets/
└── sar_1bit/
    ├── train/
    │   └── H/       # 全精度 SAR 图像
    └── test/
        └── H/
```

### Step 2: 创建自定义数据集 `data/dataset_sar_1bit.py`

### Step 3: 在 `data/select_dataset.py` 注册新数据集

### Step 4: 创建训练配置文件 `options/swinir/train_swinir_sar_1bit.json`

### Step 5: 启动训练
```bash
# 单卡调试
python main_train_psnr.py --opt options/swinir/train_swinir_sar_1bit.json

# 多卡
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 \
    main_train_psnr.py --opt options/swinir/train_swinir_sar_1bit.json --dist True
```

---

## 6. 关键文件参考速查

| 文件 | 你需要做什么 |
|---|---|
| `KAIR/models/network_swinir.py` | **无需修改**，模型定义 |
| `KAIR/models/model_plain.py` | **无需修改**，训练循环 |
| `KAIR/data/select_dataset.py` | **修改**：注册新数据集 |
| `KAIR/main_train_psnr.py` | **无需修改**，训练入口 |
| `KAIR/options/swinir/train_swinir_*.json` | **创建新配置**：`train_swinir_sar_1bit.json` |
| `KAIR/data/dataset_*.py` | **创建新文件**：`dataset_sar_1bit.py` |
