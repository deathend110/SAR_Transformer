"""从 validation CSV 绘制九个帧位置的输入与恢复指标曲线。"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # 服务器无需显示器；绘图独立于训练进程。
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint_dir', type=Path, help='包含 frame_position_summary.csv 的目录')
    args = parser.parse_args()
    with (args.checkpoint_dir / 'frame_position_summary.csv').open(
            encoding='utf-8', newline='') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row['frame_idx']))

    frame_indices = [int(row['frame_idx']) for row in rows]
    if frame_indices != list(range(9)):
        raise ValueError('Expected exactly one summary row for each frame_idx from 0 to 8.')

    for metric, ylabel in [('psnr', 'PSNR (dB)'), ('ssim', 'SSIM')]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for source, label in [('input', 'LQ'), ('output', 'SwinIR restored')]:
            values = [float(row[f'{source}_{metric}_mean']) for row in rows]
            ax.plot(frame_indices, values, marker='o', label=label)
        ax.set_xticks(frame_indices, [f'F{idx}' for idx in frame_indices])
        ax.set_xlabel('Frame position')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.checkpoint_dir / f'frame_position_{metric}.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
