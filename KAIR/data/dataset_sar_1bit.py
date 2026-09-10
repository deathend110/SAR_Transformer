import random
from pathlib import Path

import torch.utils.data as data
import utils.utils_image as util

# -----------------------------------------
# one-bit SAR image restoration dataset

class DatasetSAR1bit(data.Dataset):
    """
    Read one-bit SAR image pairs.
    """

    def __init__(self, opt):
        super(DatasetSAR1bit, self).__init__()
        self.opt = opt
        if opt['phase'] == 'train':
            self.patch_size = opt['H_size']
            if type(self.patch_size) is not int or not 1 <= self.patch_size <= 512:
                raise ValueError('H_size must be an integer between 1 and 512.')

        root_L = Path(opt['dataroot_L'])
        root_H = Path(opt['dataroot_H'])
        for root in (root_L, root_H):
            if not root.is_dir():
                raise FileNotFoundError(f'Dataset directory not found: {root}')

        split_file = opt.get('split_file')
        if split_file:
            with open(split_file, encoding='utf-8') as stream:
                sequence_names = [line.strip() for line in stream if line.strip()]
            if len(sequence_names) != len(set(sequence_names)):
                raise ValueError(f'Duplicate sequence names in split_file: {split_file}')
            for name in sequence_names:
                if name in ('.', '..') or '/' in name or '\\' in name:
                    raise ValueError(f'Expected a sequence folder name in {split_file}: {name}')
        else:
            names_L = {path.name for path in root_L.iterdir() if path.is_dir()}
            names_H = {path.name for path in root_H.iterdir() if path.is_dir()}
            if names_L != names_H:
                raise ValueError(
                    f'Sequence mismatch: missing in {root_H}: {sorted(names_L - names_H)}; '
                    f'missing in {root_L}: {sorted(names_H - names_L)}'
                )
            sequence_names = sorted(names_L)

        # 每条 sequence 按 F0～F8 展开为独立单帧样本，保留原始顺序。
        self.samples = []
        for sequence_name in sequence_names:
            for root in (root_L, root_H):
                sequence_dir = root / sequence_name
                if not sequence_dir.is_dir():
                    raise FileNotFoundError(f'Sequence directory not found: {sequence_dir}')
            for frame_idx in range(9):
                filename = f'{frame_idx:03d}.png'
                L_path = root_L / sequence_name / filename
                H_path = root_H / sequence_name / filename
                for path in (L_path, H_path):
                    if not path.is_file():
                        raise FileNotFoundError(f'Paired frame not found: {path}')
                self.samples.append((str(L_path), str(H_path), sequence_name, frame_idx))

    def __getitem__(self, index):
        L_path, H_path, sequence_name, frame_idx = self.samples[index]
        img_L = util.imread_uint(L_path, 1)
        img_H = util.imread_uint(H_path, 1)
        for img, path in ((img_L, L_path), (img_H, H_path)):
            if img.shape != (512, 512, 1):
                raise ValueError(f'Expected image shape (512, 512, 1), got {img.shape}: {path}')

        if self.opt['phase'] == 'train':
            # LQ/GT 共用裁剪坐标，不做翻转或旋转，以保留 SAR 轴向。
            top = random.randint(0, 512 - self.patch_size)
            left = random.randint(0, 512 - self.patch_size)
            img_L = img_L[top:top + self.patch_size, left:left + self.patch_size, :]
            img_H = img_H[top:top + self.patch_size, left:left + self.patch_size, :]

        # metadata 用于测试时按 sequence 和帧序重组结果，不参与网络输入。
        return {
            'L': util.uint2tensor3(img_L),
            'H': util.uint2tensor3(img_H),
            'L_path': L_path,
            'H_path': H_path,
            'sequence_name': sequence_name,
            'frame_idx': frame_idx,
        }

    def __len__(self):
        return len(self.samples)
