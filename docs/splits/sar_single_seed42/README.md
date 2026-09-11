# 单帧 SAR 数据划分

数据版本：`dataset_range_2dsft/range_2d_sft_qh2p5_ql1p5_s1_fr0p9_fa0p5_phi0_lp0p99_hp99p9_roi600_eb64`。

将 LQ sequence 目录名按字典序排序，用 Python `random.Random(42).shuffle` 打乱，前 `int(4098 * 0.9) = 3688` 条用于训练，其余 410 条用于测试。列表保留打乱后的顺序，每条 sequence 的全部九帧归属同一集合。

- 训练：3,688 条 sequence，33,192 帧。
- 测试：410 条 sequence，3,690 帧。
- 训练随机种子同样设为 42。

配置使用当前机器的绝对路径；迁移时需更新数据与列表路径。
