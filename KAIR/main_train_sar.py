import os.path
import csv
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# Single-frame SAR training, based on main_train_psnr.py
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


METRIC_NAMES = (
    'input_psnr', 'output_psnr', 'delta_psnr',
    'input_ssim', 'output_ssim', 'delta_ssim',
)


def evaluate(model, test_loader, images_dir, current_step, logger, save_images=True):
    """逐帧评估完整测试集，按需保存恢复图像，始终输出指标 CSV。"""
    if len(test_loader.dataset) == 0:
        raise ValueError('SAR evaluation dataset is empty.')

    checkpoint_dir = os.path.join(images_dir, f'{current_step:09d}')
    util.mkdir(checkpoint_dir)
    rows = []
    for test_data in test_loader:
        # batch_size=1；元数据只用于结果归档，不传入网络。
        sequence_name = test_data['sequence_name'][0]
        frame_idx = test_data['frame_idx'][0].item()
        model.feed_data(test_data)
        model.test()
        visuals = model.current_visuals()
        L_img = util.tensor2uint(visuals['L'])
        E_img = util.tensor2uint(visuals['E'])
        H_img = util.tensor2uint(visuals['H'])

        if save_images:
            sequence_dir = os.path.join(checkpoint_dir, sequence_name)
            util.mkdir(sequence_dir)
            util.imsave(E_img, os.path.join(sequence_dir, f'{frame_idx:03d}.png'))

        # 使用完整图像及 KAIR 现有指标定义，不额外 shave border。
        input_psnr = util.calculate_psnr(L_img, H_img, border=0)
        output_psnr = util.calculate_psnr(E_img, H_img, border=0)
        input_ssim = util.calculate_ssim(L_img, H_img, border=0)
        output_ssim = util.calculate_ssim(E_img, H_img, border=0)
        row = dict(sequence_name=sequence_name, frame_idx=frame_idx,
                   input_psnr=input_psnr, output_psnr=output_psnr,
                   delta_psnr=output_psnr - input_psnr,
                   input_ssim=input_ssim, output_ssim=output_ssim,
                   delta_ssim=output_ssim - input_ssim)
        rows.append(row)
        logger.info('%s/%03d | %s', sequence_name, frame_idx,
                    ', '.join(f'{key}: {row[key]:.6f}' for key in METRIC_NAMES))

    # split 文件可以有自己的顺序；CSV 始终按名称和帧号稳定排序。
    rows.sort(key=lambda row: (row['sequence_name'], row['frame_idx']))
    with open(os.path.join(checkpoint_dir, 'metrics_per_frame.csv'),
              'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sequence_name', 'frame_idx', *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for frame_idx in range(9):
        frame_rows = [row for row in rows if row['frame_idx'] == frame_idx]
        summary = dict(frame_idx=frame_idx, count=len(frame_rows))
        for key in METRIC_NAMES:
            values = [row[key] for row in frame_rows]
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_std'] = float(np.std(values, ddof=0))
        summary_rows.append(summary)

    summary_fields = ['frame_idx', 'count'] + [
        f'{key}_{stat}' for key in METRIC_NAMES for stat in ('mean', 'std')
    ]
    with open(os.path.join(checkpoint_dir, 'frame_position_summary.csv'),
              'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # 先逐帧计算提升量，再对全部帧等权求均值。
    means = {key: float(np.mean([row[key] for row in rows])) for key in METRIC_NAMES}
    logger.info('<iter:%8d> %s', current_step,
                ', '.join(f'mean {key}: {means[key]:.6f}' for key in METRIC_NAMES))
    return means


def main(json_path=os.path.join(os.path.dirname(__file__),
                               'options/swinir/train_swinir_sar_single.json')):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true', default=None)

    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    if args.dist is not None:
        opt['dist'] = args.dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist(args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G',
                                                         pretrained_path=opt['path']['pretrained_netG'])
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E',
                                                         pretrained_path=opt['path']['pretrained_netE'])
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # 保存配置前确定实际种子，使随机生成的种子也能用于复现实验。
    seed = opt['train'].get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
    opt['train']['manual_seed'] = seed

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), len(train_loader)))

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:
                if opt['rank'] == 0:
                    # DDP 的 forward 会同步 buffer；仅主进程验证时直接使用底层网络。
                    training_net = model.netG
                    if opt['dist']:
                        model.netG = model.get_bare_model(training_net)
                    try:
                        evaluate(model, test_loader, opt['path']['images'], current_step, logger,
                                 save_images=opt['train']['save_test_images'])
                    finally:
                        model.netG = training_net
                if opt['dist']:
                    # 其余进程等待完整验证结束，再一起进入下一次训练。
                    torch.distributed.barrier()


if __name__ == '__main__':
    main()
