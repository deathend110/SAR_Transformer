"""SAR 入口回归测试：在具备 KAIR 依赖的环境中以 unittest discover 运行。"""

import csv
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

KAIR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KAIR_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.dataset_sar_1bit import DatasetSAR1bit
from main_train_sar import METRIC_NAMES, evaluate, main
from models.model_plain import ModelPlain
from utils import utils_image as util
from utils import utils_option as option


class SARTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.opt = option.dict_to_nonedict(option.parse(
            str(KAIR_ROOT / 'options/swinir/train_swinir_sar_single.json')))
        self.opt['gpu_ids'] = [0] if torch.cuda.is_available() else None
        self.opt['path']['models'] = str(self.root / 'models')
        (self.root / 'models').mkdir()
        # 只缩小测试实例的容量；仍使用真实 SwinIR、ModelPlain 和优化器。
        self.opt['netG'].update(depths=[1], embed_dim=12, num_heads=[3])
        self.opt['train']['G_scheduler_milestones'] = [2]
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(0)

    def make_dataset(self, phase='test'):
        for seq_idx, name in enumerate(['sequence_B', 'sequence_A']):
            for quality in ['LQ', 'GT']:
                (self.root / quality / name).mkdir(parents=True, exist_ok=True)
            for idx in range(9):
                y, x = np.indices((512, 512))
                gt = ((x + y + idx + seq_idx * 30) % 120 + 40).astype(np.uint8)
                lq = gt + 2
                # 边缘额外误差使 border=0 与 border=1 的 PSNR 明显不同。
                lq[[0, -1], :] = 240
                lq[:, [0, -1]] = 240
                util.imsave(gt, str(self.root / 'GT' / name / f'{idx:03d}.png'))
                util.imsave(lq, str(self.root / 'LQ' / name / f'{idx:03d}.png'))
        split = self.root / 'sequences.txt'
        split.write_text('sequence_B\nsequence_A\n', encoding='utf-8')
        dataset_opt = dict(self.opt['datasets'][phase], dataroot_L=str(self.root / 'LQ'),
                           dataroot_H=str(self.root / 'GT'), split_file=str(split))
        return DatasetSAR1bit(dataset_opt)

    def make_model(self):
        model = ModelPlain(self.opt)
        model.init_train()
        return model

    def read_csv(self, path):
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)

    def test_main_records_seed_and_loader_length(self):
        train_set = self.make_dataset('train')
        test_set = self.make_dataset('test')
        self.opt['path'].update(root=str(self.root), task=str(self.root),
                                log=str(self.root), options=str(self.root / 'options'),
                                images=str(self.root / 'images'))
        self.opt['train']['manual_seed'] = None
        # 只执行真实入口的配置和 DataLoader 初始化，不进入持续训练循环。
        with patch.object(sys, 'argv', ['main_train_sar.py']), \
                patch('main_train_sar.option.parse', return_value=self.opt), \
                patch('main_train_sar.random.randint', return_value=1234), \
                patch('main_train_sar.utils_logger.logger_info'), \
                patch('main_train_sar.define_Dataset', side_effect=[train_set, test_set]), \
                patch('main_train_sar.define_Model', side_effect=RuntimeError('stop after setup')), \
                self.assertLogs('train', level='INFO') as logs:
            with self.assertRaisesRegex(RuntimeError, 'stop after setup'):
                main()

        saved_path, = (self.root / 'options').glob('*.json')
        saved = json.loads(saved_path.read_text(encoding='utf-8'))
        self.assertEqual(saved['train']['manual_seed'], 1234)
        self.assertEqual(torch.initial_seed(), 1234)
        # 18 帧 / batch 8，drop_last=True 实际只能形成 2 个训练 batch。
        self.assertTrue(any('Number of train images: 18, iters: 2' in line for line in logs.output))

    def test_validation_csv_images_and_plot(self):
        dataset = self.make_dataset()
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        model = self.make_model()
        logger = logging.getLogger('sar_test')
        with self.assertLogs(logger, level='INFO') as logs:
            means = evaluate(model, loader, str(self.root / 'images'), 50000, logger)
        checkpoint = self.root / 'images' / '000050000'
        fields, rows = self.read_csv(checkpoint / 'metrics_per_frame.csv')
        self.assertEqual(fields, ['sequence_name', 'frame_idx', *METRIC_NAMES])
        self.assertEqual([(r['sequence_name'], int(r['frame_idx'])) for r in rows],
                         [(s, i) for s in ['sequence_A', 'sequence_B'] for i in range(9)])
        self.assertEqual(len(list(checkpoint.glob('*/*.png'))), 18)
        self.assertTrue(model.netG.training)

        for row in rows:
            name, idx = row['sequence_name'], int(row['frame_idx'])
            lq = util.imread_uint(str(self.root / 'LQ' / name / f'{idx:03d}.png'), 1)
            gt = util.imread_uint(str(self.root / 'GT' / name / f'{idx:03d}.png'), 1)
            out = util.imread_uint(str(checkpoint / name / f'{idx:03d}.png'), 1)
            self.assertEqual(out.shape, (512, 512, 1))
            for metric, calculate in [('psnr', util.calculate_psnr), ('ssim', util.calculate_ssim)]:
                input_value = calculate(lq, gt, border=0)
                output_value = calculate(out, gt, border=0)
                self.assertAlmostEqual(float(row[f'input_{metric}']), input_value)
                self.assertAlmostEqual(float(row[f'output_{metric}']), output_value)
                self.assertAlmostEqual(float(row[f'delta_{metric}']), output_value - input_value)
            self.assertNotAlmostEqual(float(row['input_psnr']), util.calculate_psnr(lq, gt, border=1))

        fields, summary = self.read_csv(checkpoint / 'frame_position_summary.csv')
        self.assertEqual(fields, ['frame_idx', 'count'] + [
            f'{key}_{stat}' for key in METRIC_NAMES for stat in ['mean', 'std']])
        self.assertEqual([int(r['frame_idx']) for r in summary], list(range(9)))
        for frame in summary:
            self.assertEqual(int(frame['count']), 2)
            for key in METRIC_NAMES:
                values = [float(r[key]) for r in rows if r['frame_idx'] == frame['frame_idx']]
                self.assertAlmostEqual(float(frame[f'{key}_mean']), np.mean(values))
                self.assertAlmostEqual(float(frame[f'{key}_std']), np.std(values, ddof=0))
        for key in METRIC_NAMES:
            self.assertAlmostEqual(means[key], np.mean([float(r[key]) for r in rows]))
            self.assertIn(f'mean {key}:', logs.output[-1])

        subprocess.run([sys.executable, str(KAIR_ROOT / 'scripts/plot_sar_metrics.py'),
                        str(checkpoint)], check=True)
        for metric in ['psnr', 'ssim']:
            self.assertGreater((checkpoint / f'frame_position_{metric}.png').stat().st_size, 0)

    def test_training_metadata_and_weight_resume(self):
        dataset = self.make_dataset('train')
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        self.assertEqual(tuple(batch['L'].shape), (2, 1, 128, 128))
        self.assertEqual(tuple(batch['H'].shape), (2, 1, 128, 128))
        self.assertIn('sequence_name', batch)
        self.assertEqual(batch['frame_idx'].tolist(), [0, 1])
        model = self.make_model()
        model.feed_data(batch)
        received = []
        hook = model.netG.register_forward_pre_hook(lambda module, args: received.append(args))
        model.update_learning_rate(1)
        model.optimize_parameters(1)
        hook.remove()
        self.assertEqual(len(received[0]), 1)
        self.assertIs(received[0][0], model.L)
        self.assertTrue(np.isfinite(model.current_log()['G_loss']))
        model.save(1)
        step, path = option.find_last_checkpoint(self.opt['path']['models'], 'G')
        self.assertEqual(step, 1)
        self.opt['path']['pretrained_netG'] = path
        restored = self.make_model()
        for key, value in model.netG.state_dict().items():
            torch.testing.assert_close(restored.netG.state_dict()[key], value)
        restored.update_learning_rate(step + 1)
        self.assertAlmostEqual(restored.current_learning_rate(),
                               self.opt['train']['G_optimizer_lr'] * self.opt['train']['G_scheduler_gamma'])

    def test_optimizer_resume(self):
        model = self.make_model()
        batch = {'L': torch.rand(1, 1, 128, 128), 'H': torch.rand(1, 1, 128, 128)}
        model.feed_data(batch)
        model.update_learning_rate(1)
        model.optimize_parameters(1)
        model.save(1)
        for label, key in [('G', 'pretrained_netG'), ('optimizerG', 'pretrained_optimizerG')]:
            step, path = option.find_last_checkpoint(self.opt['path']['models'], label)
            self.assertEqual(step, 1)
            self.opt['path'][key] = path
        restored = self.make_model()
        torch.testing.assert_close(restored.G_optimizer.state_dict(), model.G_optimizer.state_dict())
        for instance in [model, restored]:
            instance.update_learning_rate(step + 1)
            instance.feed_data(batch)
            instance.optimize_parameters(step + 1)
        self.assertEqual(restored.current_learning_rate(), model.current_learning_rate())
        torch.testing.assert_close(restored.netG.state_dict(), model.netG.state_dict())

    def test_local_dataset_evaluation(self):
        roots = sorted((KAIR_ROOT.parent / 'dataset_range_2dsft').glob('*/LQ'))
        if not roots:
            self.skipTest('Local SAR dataset is unavailable')
        dataset_opt = dict(self.opt['datasets']['test'], dataroot_L=str(roots[0]),
                           dataroot_H=str(roots[0].parent / 'GT'), split_file=None)
        dataset = DatasetSAR1bit(dataset_opt)
        # 原始数据只读，取前两条完整 sequence，不改变正式 split。
        loader = DataLoader(Subset(dataset, range(18)), batch_size=1, shuffle=False)
        evaluate(self.make_model(), loader, str(self.root / 'images'), 1,
                 logging.getLogger('sar_local_test'))
        checkpoint = self.root / 'images' / '000000001'
        self.assertEqual(len(list(checkpoint.glob('*/*.png'))), 18)
        _, summary = self.read_csv(checkpoint / 'frame_position_summary.csv')
        self.assertEqual([int(row['count']) for row in summary], [2] * 9)


if __name__ == '__main__':
    unittest.main()
