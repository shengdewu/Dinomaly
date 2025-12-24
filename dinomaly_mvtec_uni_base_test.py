# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from dataset import get_data_transforms
import numpy as np
import random
import os
import math
from torch.nn import functional as F
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2, ConvBlock, FeatureJitter
from dataset import MVTecDataset
from utils import evaluation_batch
from functools import partial
import warnings
import logging

from utils import visualize
from datetime import datetime
from dinov3.hub.backbones import load_dinov3_model

warnings.filterwarnings("ignore")


def export_onnx(model, im, file, opset, dynamic, simplify):
    import onnx
    import onnx_graphsurgeon as gs

    print(onnx.__version__)

    f = file.replace('.pth', '.onnx')

    # output_names = ['output0', 'output1', 'output2', 'output3']
    output_names = ['output0']
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    print('modify onnx node!')
    graph = gs.import_onnx(model_onnx)

    flatten_node = None
    for node in graph.nodes:
        if node.name == '/rope_embed/Flatten_1':
            flatten_node = node
            print('flatten_node:', flatten_node)
            break

    tile_node = None
    for node in graph.nodes:
        if node.name == '/rope_embed/Tile':
            tile_node = node
            print('Tile_node:', tile_node)
            break

    tile_node.inputs[0] = flatten_node.outputs[0]

    constant_repeat = gs.Constant(name='/rope_embed/constant_repeat_1', values=np.array([1, 2], dtype=np.int64),
                                  export_dtype=np.int64)
    tile_node.inputs[1] = constant_repeat

    print('rt Tile_node:', tile_node)

    graph.cleanup().toposort()
    model_onnx = gs.export_onnx(
        graph,
        ir_version=10,  # 强制设置 IR 版本为 10，兼容 onnxruntime
    )

    f = f.replace('.onnx', '-modify.onnx')
    onnx.checker.check_model(model_onnx)
    # model_onnx = onnx.version_converter.convert_version(model_onnx, 10)
    onnx.save(model_onnx, f)
    print('modify onnx check success!')

    # Simplify
    if simplify:
        try:
            # 'onnx-simplifier>=0.4.1')
            import onnxsim

            print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            print(f'simplifier failure: {e}')
    return f, model_onnx


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


class DinomalyModel(nn.Module):
    def __init__(self, base_model: nn.Module, kernel_size=3, sigma=2, channels=1):
        super(DinomalyModel, self).__init__()
        self.base_model = base_model
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        self.gaussian_kernel = self.get_gaussian_kernel()
        return

    def cal_anomaly_maps(self, fs_list, ft_list, out_size=224):
        if not isinstance(out_size, tuple):
            out_size = (out_size, out_size)

        a_map_list = []
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
            a_map_list.append(a_map)
        anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
        return anomaly_map, a_map_list

    def get_gaussian_kernel(self):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(self.kernel_size)
        x_grid = x_coord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (self.kernel_size - 1) / 2.
        variance = self.sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)

        gaussian_filter = torch.nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                          kernel_size=self.kernel_size, groups=self.channels,
                                          bias=False, padding=self.kernel_size // 2)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    def forward(self, img):
        output = self.base_model(img)
        en, de = output[0], output[1]
        anomaly_map, _ = self.cal_anomaly_maps(en, de, img.shape[-1])
        anomaly_map = self.gaussian_kernel(anomaly_map)
        return anomaly_map


def test(item_list, image_size):
    setup_seed(1)

    if not os.path.exists(args.weight):
        return

    batch_size = 8

    data_transform, gt_transform = get_data_transforms(image_size)

    test_data_list = []

    for i, item in enumerate(item_list):
        test_path = os.path.join(args.data_path, item)

        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        test_data_list.append(test_data)

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

    model.eval()
    state_dict = torch.load(args.weight, map_location='cpu')
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    if isinstance(image_size, int):
        in_shape = [1, 3] + [image_size, image_size]
    else:
        in_shape = [1, 3] + list(image_size)

    im = torch.zeros(in_shape).to(device)
    export_onnx(model, im, args.weight, 17, False, False)

    now = datetime.now()
    save_time = now.strftime("%Y%m%d%H%M%S")

    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

    for item, test_data in zip(item_list, test_data_list):
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                      num_workers=1)
        # results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
        # auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

        # auroc_sp_list.append(auroc_sp)
        # ap_sp_list.append(ap_sp)
        # f1_sp_list.append(f1_sp)
        # auroc_px_list.append(auroc_px)
        # ap_px_list.append(ap_px)
        # f1_px_list.append(f1_px)
        # aupro_px_list.append(aupro_px)
        #
        # print(
        #     '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
        #         item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
        #
        # print(
        #     'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
        #         np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
        #         np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

        visualize(model, test_dataloader, device, _class_=item, save_path=f'{args.save_name}')

    return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/mnt/sda/datasets/皮带异常数据集合/MVTec-AD-Style')
    parser.add_argument('--save_name', type=str, default='saved_test')
    parser.add_argument('--weight', type=str,
                        default='saved_results/vitill_mvtec_uni_dinov3_base/pdseg-clahe-region/(512, 512)/20251223-114121/model.pth')
    args = parser.parse_args()

    item_list = ['pdseg-clahe-region']

    image_size = (512, 512)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    test(item_list, image_size)
