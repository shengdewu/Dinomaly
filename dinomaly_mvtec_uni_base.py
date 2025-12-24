# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import ConcatDataset

from models.uad import ViTill, ViTillv2
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2, ConvBlock, FeatureJitter
from dataset import MVTecDataset
from utils import evaluation_batch, global_cosine, regional_cosine_hm_percent, global_cosine_hm_percent, \
    WarmCosineScheduler
from functools import partial
from optimizers import StableAdamW, AdamW
import torch.optim as optim
import warnings
import logging
import math

from utils import visualize
from datetime import datetime
from dinov3.hub.backbones import load_dinov3_model

warnings.filterwarnings("ignore")


class WarmupCosineAnnealingLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0., last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段：线性增加学习率
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(item_list, save_path, image_size=512):
    setup_seed(1)

    total_iters = 25000
    check = 1000
    batch_size = 8
    lr = 2e-4
    save_iter = 10000

    data_transform, gt_transform = get_data_transforms(image_size)

    train_data_list = []
    test_data_list = []

    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        test_path = os.path.join(args.data_path, item)

        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]

        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder_name = 'dinov3_vitb16'
    encoder_weight = 'weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'

    encoder = load_dinov3_model(encoder_name, layers_to_extract_from=target_layers,
                                pretrained_weight_path=encoder_weight)

    if 'vits' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'vitb' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'vitl' in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise "Architecture not in vits, vitb, vitl."

    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)

    trainable = nn.ModuleList([bottleneck, decoder])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = AdamW([{'params': trainable.parameters()}],
                      lr=lr, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=False, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=lr, final_value=lr * 0.05, total_iters=total_iters,
                                       warmup_iters=100)

    # lr_scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=100, max_epochs=total_iters, eta_min=1e-7)

    total_epoch = int(np.ceil(total_iters / len(train_dataloader)))
    save_epoch = int(np.ceil(save_iter / len(train_dataloader)))

    print_fn('total epoch {} train image number:{}; save epoch {}'.format(total_epoch, len(train_data), save_epoch))

    it = 0
    for epoch in range(total_epoch):
        model.train()
        model.encoder.eval()

        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            # label = label.to(device)
            en, de = model(img)

            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            # is_nan = [torch.isnan(p).sum() for p in en]
            # assert sum(is_nan) == 0, 'en'

            # is_nan = [torch.isnan(p).sum() for p in de]
            # assert sum(is_nan) == 0, 'de'

            # # 1. 先看loss是不是nan,如果loss是nan,那么说明可能是在forward的过程中出现了第一条列举的除0或者log0的操作
            # assert torch.isnan(loss).sum() == 0,  f'{len(en)}'

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            # # 2. 如果loss不是nan,那么说明forward过程没问题，可能是梯度爆炸，所以用梯度裁剪试试
            # # 3.1 在step之前，判断参数是不是nan, 如果不是判断step之后是不是nan
            # is_nan = [torch.isnan(p).sum() for p in trainable.parameters()]
            # assert sum(is_nan) == 0

            optimizer.step()

            # # 3.2 在step之后判断，参数和其梯度是不是nan，如果3.1不是nan,而3.2是nan,
            # # 特别是梯度出现了Nan,考虑学习速率是否太大，调小学习速率或者换个优化器试试。
            # is_nan = [torch.isnan(p).sum() for p in trainable.parameters()]
            # assert sum(is_nan) == 0, print(model.mu)
            # is_nan = [torch.isnan(p.grad).sum() for p in trainable.parameters() if p.grad is not None]
            # assert sum(is_nan) == 0, print(model.mu.grad)

            loss_list.append(loss.item())
            lr_scheduler.step()

            if (it + 1) % check == 0:

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                  num_workers=4)
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

                model.train()
                model.encoder.eval()

            it += 1
            if it == total_iters:
                break

        current_lr = optimizer.param_groups[0]['lr']
        now = datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        print_fn('{} iter [{}/{}], loss:{:.4f}, lr:{:.6f}'.format(now, it, total_iters, np.mean(loss_list), current_lr))

        if epoch % save_epoch == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'model-{epoch}.pth'))

    model.eval()
    for item, test_data in zip(item_list, test_data_list):
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                      num_workers=2)

        visualize(model, test_dataloader, device, _class_=args.item_name, save_path=save_path)

    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/mnt/sda/datasets/皮带异常数据集合/MVTec-AD-Style')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='vitill_mvtec_uni_dinov3_base')
    parser.add_argument('--item_name', type=str, default='pdseg-clahe-region')
    args = parser.parse_args()

    # item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
    #              'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

    item_list = [args.item_name]

    image_size = (512, 512)

    now = datetime.now()
    save_time = now.strftime("%Y%m%d-%H%M%S")
    save_path = f'{args.save_dir}/{args.save_name}/{args.item_name}/{image_size}/{save_time}'
    os.makedirs(save_path, exist_ok=True)

    logger = get_logger(args.save_name, save_path)
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    train(item_list, save_path, image_size)
