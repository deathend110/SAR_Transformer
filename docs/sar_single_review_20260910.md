审查范围：`docs/mission 1.md`、`docs/mission 2.md`、`docs/mission 3.md` 对应的三次提交：

| 提交 | 内容 | 审查结论 |
| --- | --- | --- |
| `7dd58cb` | 单帧 SAR Dataset | 数据展开、严格配对、paired crop、metadata 和测试整图行为符合 mission 1。 |
| `c2e0474` | 单帧 SwinIR 配置 | 网络、损失、优化器和调度参数符合 mission 2；补充固定随机种子。 |
| `8d3d542` | SAR 训练、评估、绘图和测试 | 单帧训练与九帧评估结构符合 mission 3；修正种子保存、学习率显示及批次数日志。 |

先完成上述提交及数据结构审查，再集中修改。审查期间新增的 uv 环境提交不属于这三次提交的审查范围。

发现及修复：

1. **随机种子未固定，也未写入实验配置。** 配置缺少 `manual_seed`，入口在保存 options 后才生成随机种子，相同配置无法重现网络初始化和随机采样。现在配置固定 `manual_seed=0`，入口先确定并保存实际种子；显式设置为 `null` 时，生成的种子同样写入 options 和配置日志。
2. **milestone 处显示的学习率多衰减一次。** SAR 入口复用的 `ModelBase.current_learning_rate()` 调用 `get_lr()`，在 `MultiStepLR` 已执行调度后再次计算衰减。现在改为 `get_last_lr()`，只读取调度结果。该修复不改变优化器实际使用的学习率或 scheduler 配置。
3. **训练批次数日志与 `drop_last=True` 不一致。** 原入口使用向上取整；18 帧、batch 8 会报告 3 批，实际只有 2 批。现在直接记录 `len(train_loader)`，同时适用于现有 sampler 分支。

运行验证中发现并修复的两项问题：

4. **优化器恢复强制把全部状态移到 CUDA。** 原 `ModelBase.load_optimizer()` 将 Adam 原本在 CPU 上的 `step` 计数也移到 GPU，恢复状态的设备与保存前不一致。改为先在 CPU 加载，再由 `optimizer.load_state_dict()` 按参数设备和优化器规则恢复各项状态。GPU 上的动量仍在 GPU，普通 Adam 的步数计数保持在 CPU。
5. **测试导入阶段过早查询 CUDA。** 原 `skipUnless(torch.cuda.is_available(), ...)` 在配置解析前执行，导致此机器上 GPU 可见性提前确定，单卡测试意外进入多卡通信并报 NCCL 错误。优化器恢复已支持 CPU，因此移除这项装饰器及测试中强制指定 GPU 的语句，统一使用 setUp 在读取配置后选择的设备。

数据检查：

- mission 1 指定版本为 `dataset_range_2dsft/range_2d_sft_qh2p5_ql1p5_s1_fr0p9_fa0p5_phi0_lp0p99_hp99p9_roi600_eb64`。
- 该版本 LQ/GT 各有 4,098 条 sequence，两侧名称完全一致；每侧 36,882 帧，即 Dataset 长度为 `4098 × 9 = 36882`。
- 逐目录检查目标版本，每条 sequence 两侧均有且只有 `000.png`～`008.png`。
- 另外两个参数版本也各有 4,098 条配对 sequence。三个版本总计检查 221,292 个预期 PNG 的存在性和头信息，均为 512×512、8 位、灰度图，没有缺帧。此项是文件头检查，实际解码由后续真实数据 smoke test 覆盖。
- 三个版本都没有现成的 split 文件。本次未划分或修改原始数据。

实现核对：

- Dataset 每个 item 的 L/H 为 `[1,128,128]`（训练）或 `[1,512,512]`（测试），沿用 KAIR 的 `[0,1]` 浮点转换。训练两侧共用裁剪坐标，不做旋转或翻转。
- `sequence_name` 保留原目录名，`frame_idx` 为 0～8。无 split 时按 sequence 名字典序排列；提供 split 时按文件顺序展开每条 sequence 的九帧。
- `ModelPlain.feed_data()` 只读取 L/H，网络输入为 `[B,1,H,W]`，metadata 不进入网络。
- 配置保持 `in_chans=1`、`upscale=1`、`upsampler=null`、`img_size=128`、`window_size=8`、六组深度 6、`embed_dim=180`、六组 head 数 6、`mlp_ratio=2`、`resi_connection="1conv"`。
- Charbonnier 使用 `sqrt(diff² + eps)`，`eps=1e-9` 与 KAIR 灰度基线一致。Adam 学习率 `2e-4`、betas `[0.9,0.999]`；MultiStepLR 的 milestones 为 `[800000,1200000,1400000,1500000,1600000]`，gamma 为 `0.5`。验证使用 netG，`E_decay=0` 正确。
- 验证 loader 固定 batch 1、不打乱、不丢尾批；输入和恢复的 PSNR/SSIM 均调用 KAIR 原函数，传入 `border=0`。
- 图片保存为 `images/<九位迭代号>/<sequence_name>/000.png`～`008.png`，不同 sequence 的同名图片互不覆盖。
- `metrics_per_frame.csv` 按 `(sequence_name, frame_idx)` 排序，列为 `sequence_name, frame_idx, input_psnr, output_psnr, delta_psnr, input_ssim, output_ssim, delta_ssim`。
- `frame_position_summary.csv` 固定九行，包含 `frame_idx`、`count`，以及六项指标各自的 `_mean` 和 `_std`；标准差使用 `ddof=0`。总体日志对全部单帧等权求六项均值。
- 绘图脚本独立读取 CSV，输出 `frame_position_psnr.png` 和 `frame_position_ssim.png`。

验证结果（使用仓库现有 uv 环境）：

- 环境：Python 3.12.14、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、timm 1.0.29、OpenCV 4.11.0；CUDA 可用。
- 执行 `uv run --no-sync python -m unittest discover -s KAIR/tests -p 'test_train_sar.py' -v`：5 项测试全部通过，无跳过，耗时约 11 秒。
- 回归覆盖：实际随机种子写入 options；18 帧、batch 8、drop_last 的日志为 2 批；单帧训练 metadata；权重恢复和 milestone 学习率；GPU Adam 完整状态恢复及恢复后继续训练的参数一致性；CSV 列、排序、均值及标准差；border=0；不同 sequence 图片不覆盖；独立绘图输出。
- 真实数据评估使用目标版本前两条完整 sequence，共 18 帧；恢复图片全部保存，九个 frame-position 分组的 count 均为 2。回归测试仅缩小模型容量，不改变真实输入的 512×512 尺寸。
- 另外使用 JSON 原始完整网络配置构造模型，参数量为 11,497,681。在真实数据上完成一轮 `[1,1,128,128]` 的前向、反向和优化器更新，loss 有限；随后完成 `[1,1,512,512]` 整图推理，输出同尺寸且全部有限。此检查使用 batch 1，不代表已验证正式 batch 8 的训练显存预算。
- 实际构造 train/test Dataset，长度均为 36,882；将随机裁剪坐标固定为 `(17,29)`，LQ/GT tensor 与原图同位置裁剪逐元素完全一致。
- 相关 Python 文件语法检查、配置解析和 `git diff --check` 均通过。

正式训练前的配置事项：

- 按 mission 2 保留 JSON 中的数据路径和 split 路径占位值；使用前应填入目标版本的 LQ/GT 路径及确定的 train/test sequence 列表。
- batch 8、128×128 patch、完整 512×512 验证及原始调度预算保持不变。训练入口沿用 KAIR 持续训练循环，最后一个 milestone 不是自动停止步数。
- 本次不下载预训练权重，不启动正式长时间训练，不改动 SwinIR 网络源码。
