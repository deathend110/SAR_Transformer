# 单比特 SAR 恢复实验使用说明

当前实验是基于 KAIR 的单帧 SwinIR 基线：将每条 sequence 的九帧分别输入网络，学习 1-bit 图像到对应全精度 GT 的恢复。评估时按 F₀～F₈ 汇总结果，为后续序列模型提供对照。

## 1. 环境与工作目录

以下命令均在仓库根目录执行，实验输出路径也以此为基准：

```bash
cd /data/cjz/SAR_Transformer
uv sync --locked
```

项目使用 Python 3.12、PyTorch 2.6.0 和 CUDA 12.4 版 PyTorch，依赖由 `pyproject.toml`、`uv.lock` 管理。已有环境可直接使用下文的 `uv run --no-sync` 命令。

## 2. 数据和默认配置

训练入口：`KAIR/main_train_sar.py`。

默认配置：`KAIR/options/swinir/train_swinir_sar_single.json`。该文件支持 `//` 注释，修改时保持现有格式。

当前数据版本为 `dataset_range_2dsft/range_2d_sft_qh2p5_ql1p5_s1_fr0p9_fa0p5_phi0_lp0p99_hp99p9_roi600_eb64`，其中 `LQ/` 和 `GT/` 下按同名 sequence 配对，每条包含 `000.png`～`008.png`。

| 项目             | 当前设置                                                 |
| -------------- | ---------------------------------------------------- |
| 数据划分           | seed 42，按完整 sequence 划分 80% 训练、20% 测试                |
| 训练集            | 3,278 条 sequence，共 29,502 帧                          |
| 测试集            | 820 条 sequence，共 7,380 帧                             |
| 划分列表           | `docs/splits/sar_single_seed42/train.txt`、`test.txt` |
| 训练输入           | 单通道，LQ/GT 同位置随机裁剪 128×128，不翻转或旋转                     |
| 测试输入           | 单通道，完整 512×512 图像                                    |
| 总 batch size   | 6（GPU 0、1 各 3 张）                                     |
| 随机种子           | 42                                                   |
| 损失与优化器         | Charbonnier、Adam，初始学习率 `1.732e-4`                    |
| 学习率衰减          | 1066667、1600000、1866667、2000000、2133333 步，各乘 0.5     |
| 日志 / 保存 / 评估间隔 | 267 / 6667 / 6667 步                                  |

数据和 split 路径已配置为当前机器的绝对路径。迁移仓库时需更新 `datasets.train`、`datasets.test` 中对应路径。问题定义见 [Define.md](Define.md)，划分方法见 [数据划分说明](splits/sar_single_seed42/README.md)。

## 3. 选择 GPU 并启动

当前可用物理 GPU 为 **0、1、3**。入口会用配置中的 `gpu_ids` 覆盖 `CUDA_VISIBLE_DEVICES`，因此通过配置选择 GPU。

### 默认双卡运行（GPU 0、1）

默认配置为 `"gpu_ids": [0,1]`、`"dist": false`，总 batch size 为 6，每卡处理 3 张 128×128 patch：

```bash
uv run --no-sync python KAIR/main_train_sar.py --opt KAIR/options/swinir/train_swinir_sar_single.json
```

该方式使用现有 `DataParallel`，不需要 `torchrun` 或 `--dist`。GPU 0 是主卡，承担结果聚合；验证 batch size 为 1，主要使用主卡。

当前总 batch 从 2 调为 6，初始学习率按 `1e-4 × sqrt(6/2)` 设为 `1.732e-4`，后续根据收敛情况评估。milestones 和日志、保存、评估间隔按原 batch 2 配置除以 3 并四舍五入到整数步，近似保持相应节点前累计处理的训练样本量一致；这不等价于原 batch 2 的优化过程。

此前 batch 2 的实测训练占用约为每卡 7～7.5 GB（用户监控读数）。该读数对应旧配置，batch 6 的实际占用需在重启后观察。

要换用 GPU 0、3，将 `gpu_ids` 改为 `[0,3]`。默认使用双卡，每卡 3 张 patch。

### 后台运行与查看进度

使用默认双卡配置，执行一次即可：

```bash
mkdir -p logs
nohup uv run --no-sync python KAIR/main_train_sar.py \
  --opt KAIR/options/swinir/train_swinir_sar_single.json \
  > logs/swinir_sar_single.console.log 2>&1 &
echo $!
```

记录输出的后台 PID。查看控制台输出和 GPU 使用情况：

```bash
tail -f logs/swinir_sar_single.console.log
nvidia-smi
```

`tail -f` 中按 Ctrl+C 只退出日志查看。完整测试会逐帧处理 7,380 张图像并保存恢复图，评估期间训练暂停。

## 4. 输出位置与指标

输出目录由 `path.root` 和 `task` 拼接。默认实验为 `denoising/swinir_sar_single/`：

```text
denoising/<task>/
├── train.log                     # 配置、训练 loss、学习率及评估指标
├── options/                      # 本次运行保存的配置
├── models/                       # 网络和优化器检查点
└── images/
    └── 000006667/                # 九位迭代号，例如第 6667 步
        ├── <sequence_name>/
        │   ├── 000.png
        │   └── ... 008.png       # 九帧恢复图
        ├── metrics_per_frame.csv
        └── frame_position_summary.csv
```

- `metrics_per_frame.csv`：每个 sequence、每帧的输入与恢复 PSNR/SSIM，以及 `delta = output - input`。
- `frame_position_summary.csv`：F₀～F₈ 各位置的样本数、指标均值和标准差，标准差使用 `ddof=0`。
- `train.log`：同时记录全部测试帧等权平均的六项指标。

指标沿用 KAIR 实现，在转为 uint8 的完整图像上计算，`border=0`。分析时可同时查看整体提升、中央 F₄ 的改善和九帧质量分布。

在某次评估完成后生成帧位置曲线：

```bash
uv run --no-sync python KAIR/scripts/plot_sar_metrics.py \
  denoising/swinir_sar_single/images/000006667
```

该目录会新增 `frame_position_psnr.png` 和 `frame_position_ssim.png`。其他实验替换 task，其他评估步数替换末尾目录名。

## 5. 停止、续训与新实验

训练入口采用持续训练循环，最后一个学习率 milestone **不是自动停止步数**。前台训练可按 Ctrl+C 停止；后台训练先定位对应 Python 进程，再向该 PID 发送终止信号：

```bash
pgrep -af 'python KAIR/main_train_sar.py'
kill <对应训练进程的PID>
```

停止时不会额外保存检查点，需以最近一次完整保存的检查点为准。建议在日志显示保存及当次评估完成后停止。

修改配置后需重启训练进程才能生效。若从旧 batch 的检查点续训，入口沿用保存的迭代号，不会按已处理样本量换算历史步数；上述样本量对齐以从头训练为基准。

**续训**：从同一工作目录重新执行原启动命令。入口会自动查找该 task 的 `models/` 下最新网络和优化器检查点，恢复迭代步数并继续训练。保留配套网络、优化器文件；该恢复不保证随机裁剪和数据顺序与未中断运行逐步完全一致。

**从头开始新实验**：复制配置并修改 `task` 为新名称，保持预训练路径为 `null`。不同实验使用独立 task，以区分输出和自动续训来源。不要同时启动两个使用相同 task 输出目录的训练进程。

## 6. 快速检查

需要检查训练与评估流程时，可运行现有回归测试：

```bash
uv run --no-sync python -m unittest discover \
  -s KAIR/tests -p 'test_train_sar.py' -v
```

此前的实现审查和验证结果见 [单帧基线审查记录](sar_single_review_20260910.md)。其中数据路径待填、种子为 0 的历史描述已由当前 seed 42 配置和数据划分更新。
