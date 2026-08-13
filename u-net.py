# -*- coding: utf-8 -*-
"""
U-Net 液膜厚度回归项目
======================
由培养皿俯拍 RGB 图像回归液膜厚度图（单位 cm）。

数据与流程
----------
1. 读取 pre_analysis/ 中已预处理的图像 "num_deg (x).png"（384×384，旋转/黑色
   补全已在 pre_analysis-1.py 完成），不再重采样，直接作为输入；
2. 对照同名厚度图 "num_deg.tiff"（384×384，皿外为 NaN，由 thickness.py 补齐），
   直接作为回归标签；
3. 按"角度"划分数据集（同一角度的 3 张图共享同一 tiff，避免数据泄漏）：
   80% 角度为训练集、20% 为测试集；再从训练集中划出 10% 作为验证集
   （用于选取最佳模型）；
4. 训练 U-Net：输入 3 通道 RGB，输出 1 通道厚度图；损失仅在皿内有效像素上
   计算（掩膜 MSE，厚度归一化到 [0,1] 后回归，再换算回 cm 评估）；
5. 将最佳模型保存到子文件夹 models/，并在测试集上给出 RMSE / MAE（cm）。

用法
----
训练（默认参数；直接读取 pre_analysis/ 中 384×384 的预处理数据，不重采样）：
    python u-net.py
    python u-net.py --epochs 60 --batch-size 8     # 自定义轮数 / 批大小

预测（用已训练模型对单张图像输出厚度图，若有同名 *_deg.tiff 还会生成对比图）：
    python u-net.py --predict "pre_analysis/3.0_deg (1).png"

主要参数（完整列表可用 python u-net.py --help 查看）：
    --data-dir    预处理后图像/标签目录   （默认 pre_analysis）
    --out-dir     模型与结果保存子文件夹 （默认 models）
    --size        模型输入尺寸，须为16倍数 （默认 384）
    --base        U-Net 基础通道数         （默认 16）
    --epochs      训练轮数                （默认 30）
    --batch-size  批大小                  （默认 8）
    --lr          学习率                  （默认 1e-3）
    --seed        随机种子                （默认 42）
    --rebuild-cache  强制重建预处理缓存
    --predict     预测模式：传入一张图像路径
"""
import os
import re
import json
import hashlib
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import tifffile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# matplotlib 中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 厚度归一化常数（cm）：全角度实测最大厚度 < 0.55 cm，取 0.6 保证不饱和
H_MAX = 0.6


# ============================ 1. 数据配对 ============================
def list_pairs(data_dir):
    """返回 [(png 文件名, 角度 key '0.0_deg'), ...]。
    仅匹配 pre_analysis 中 'num_deg (x).png'，并要求存在同名 'num_deg.tiff'
    作为标签。"""
    pairs = []
    for f in sorted(os.listdir(data_dir)):
        m = re.match(r'^([0-9.]+_deg) \((\d+)\)\.png$', f, re.I)
        if not m:
            continue
        tiff = os.path.join(data_dir, m.group(1) + '.tiff')
        if os.path.isfile(tiff):
            pairs.append((f, m.group(1)))
    return pairs


def resize_image_rgb(img, size):
    """PIL RGB → size×size 数组 (H,W,3) uint8，LANCZOS 高质量重采样。
    仅用于预测模式对单张任意尺寸输入做归一化；训练数据已是 384×384，
    预处理阶段不再调用。"""
    img = img.convert('RGB').resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def resize_gt_thickness(t, size):
    """厚度图 (H,W) float32（皿外为 NaN）→ size×size，NaN 感知双线性缩放。
    仅用于预测模式，将同名真实厚度图对齐到模型输入尺寸。"""
    t = t.astype(np.float32)
    valid = np.isfinite(t)
    t0 = np.nan_to_num(t, nan=0.0)
    img = Image.fromarray(t0).resize((size, size), Image.BILINEAR)
    m = Image.fromarray(valid.astype(np.float32)).resize((size, size),
                                                         Image.BILINEAR)
    mz = np.asarray(m, dtype=np.float32) > 0.5
    tz = np.asarray(img, dtype=np.float32)
    return np.where(mz, tz, np.nan).astype(np.float32)


def signature(pairs, data_dir, size):
    """缓存有效性签名（尺寸 + 全部配对文件名及其大小）。"""
    h = hashlib.md5()
    h.update(f'size={size}'.encode())
    for f, _ in pairs:
        h.update(f'{f}:{os.path.getsize(os.path.join(data_dir, f))};'.encode())
    return h.hexdigest()


def preprocess(data_dir, cache_dir, size, rebuild=False):
    """读取已预处理（384×384，不重采样）的样本并缓存到
    cache_dir/dataset_{size}.npz。
    返回 (images, thickness, mask, angle_keys, pairs)。
    images (N,3,S,S) uint8；thickness (N,1,S,S) float32（皿外 NaN）；
    mask (N,1,S,S) uint8（1=有效）；angle_keys (N,) S 数组。"""
    os.makedirs(cache_dir, exist_ok=True)
    pairs = list_pairs(data_dir)
    if not pairs:
        raise RuntimeError(f'在 {data_dir} 中未找到 "num_deg (x).png" 配对样本！')

    cache_path = os.path.join(cache_dir, f'dataset_{size}.npz')
    sig = signature(pairs, data_dir, size)

    if os.path.isfile(cache_path) and not rebuild:
        data = np.load(cache_path)
        if str(data['signature'].item()) == sig:
            print(f'[数据] 使用缓存: {cache_path}（{len(pairs)} 个样本）')
            return (data['images'], data['thickness'], data['mask'],
                    data['angle_keys'], pairs)
        print('[数据] 缓存失效（数据或尺寸变化），重新预处理…')

    print(f'[数据] 读取 {len(pairs)} 个样本（{size}×{size}，已预处理，不重采样）…')
    N = len(pairs)
    images = np.empty((N, 3, size, size), dtype=np.uint8)
    thickness = np.full((N, 1, size, size), np.nan, dtype=np.float32)
    mask = np.zeros((N, 1, size, size), dtype=np.uint8)
    angle_keys = np.array([k.encode('utf-8') for _, k in pairs], dtype='S32')
    for i, (f, key) in enumerate(pairs):
        img = Image.open(os.path.join(data_dir, f)).convert('RGB')
        if img.size != (size, size):
            raise ValueError(f'{f} 尺寸 {img.size} != {size}×{size}，请先在 '
                             'pre_analysis-1.py 中统一为 384×384。')
        images[i] = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
        t = tifffile.imread(os.path.join(data_dir, key + '.tiff'))
        if t.shape != (size, size):
            raise ValueError(f'{key}.tiff 尺寸 {t.shape} != {size}×{size}，'
                             '请先在 thickness.py 中补齐为 384×384。')
        t = t.astype(np.float32)
        thickness[i, 0] = t
        mask[i, 0] = np.isfinite(t).astype(np.uint8)
        angle_keys[i] = key
        if (i + 1) % 50 == 0 or i + 1 == N:
            print(f'    {i + 1}/{N} 完成')
    np.savez(cache_path, signature=sig, images=images, thickness=thickness,
             mask=mask, angle_keys=angle_keys)
    print(f'[数据] 已保存缓存: {cache_path}')
    return images, thickness, mask, angle_keys, pairs


# ============================ 2. 数据集与划分 ============================
class ThicknessDataset(Dataset):
    """RGB → 归一化厚度图。样本来自预处理的 numpy 数组。"""

    def __init__(self, images, thickness, mask, idxs):
        self.images = images[idxs]       # uint8 (N,3,S,S)
        self.targets = thickness[idxs]   # float32 (N,1,S,S) cm，皿外 NaN
        self.masks = mask[idxs]          # uint8 (N,1,S,S)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = self.images[i].astype(np.float32) / 255.0
        y = np.nan_to_num(self.targets[i], nan=0.0) / H_MAX   # -> [0,1]
        m = self.masks[i].astype(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(y),
                torch.from_numpy(m))


def split_indices(angle_keys, seed, train_frac, val_frac):
    """按角度划分 train / val / test 索引。
    同一角度的所有样本（3 张图共享同一 tiff）始终在同一集合，避免数据泄漏。"""
    uniq = sorted(set(angle_keys))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_test = int(round(len(uniq) * (1 - train_frac)))
    n_train_total = len(uniq) - n_test
    n_val = int(round(n_train_total * val_frac))
    n_train = n_train_total - n_val

    train_keys = set(uniq[i] for i in perm[:n_train])
    val_keys = set(uniq[i] for i in perm[n_train:n_train + n_val])
    test_keys = set(uniq[i] for i in perm[n_train + n_val:])

    by_key = {}
    for i, k in enumerate(angle_keys):
        by_key.setdefault(k, []).append(i)
    train_idx = [i for k in train_keys for i in by_key[k]]
    val_idx = [i for k in val_keys for i in by_key[k]]
    test_idx = [i for k in test_keys for i in by_key[k]]
    return train_idx, val_idx, test_idx, (train_keys, val_keys, test_keys)


# ============================ 3. U-Net 模型 ============================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """标准 U-Net：输入 in_ch 通道 → 输出 out_ch 通道。
    384×384 输入（4 次下采样到 24×24 瓶颈）。"""

    def __init__(self, in_ch=3, out_ch=1, base=16):
        super().__init__()
        f = base
        # 编码器
        self.e1 = DoubleConv(in_ch, f)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = DoubleConv(f, f * 2)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = DoubleConv(f * 2, f * 4)
        self.p3 = nn.MaxPool2d(2)
        self.e4 = DoubleConv(f * 4, f * 8)
        self.p4 = nn.MaxPool2d(2)
        # 瓶颈
        self.bottleneck = DoubleConv(f * 8, f * 16)
        # 解码器
        self.d4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.u4 = DoubleConv(f * 8 + f * 8, f * 8)
        self.d3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.u3 = DoubleConv(f * 4 + f * 4, f * 4)
        self.d2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.u2 = DoubleConv(f * 2 + f * 2, f * 2)
        self.d1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.u1 = DoubleConv(f + f, f)
        self.out = nn.Conv2d(f, out_ch, 1)

    def forward(self, x):
        c1 = self.e1(x)
        c2 = self.e2(self.p1(c1))
        c3 = self.e3(self.p2(c2))
        c4 = self.e4(self.p3(c3))
        b = self.bottleneck(self.p4(c4))
        u = self.u4(torch.cat([self.d4(b), c4], dim=1))
        u = self.u3(torch.cat([self.d3(u), c3], dim=1))
        u = self.u2(torch.cat([self.d2(u), c2], dim=1))
        u = self.u1(torch.cat([self.d1(u), c1], dim=1))
        return self.out(u)


# ============================ 4. 损失 / 评估 ============================
def masked_mse(pred, target, mask):
    """仅对有效像素（皿内）计算 MSE。"""
    d = (pred - target) ** 2
    return (d * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, loader, device):
    """在数据集上评估，返回 (RMSE_cm, MAE_cm)。"""
    model.eval()
    tot_mse = 0.0
    tot_mae = 0.0
    tot_w = 0.0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        pred = model(x)
        d = pred * H_MAX - y * H_MAX      # 换算回 cm
        w = m.sum().item()
        tot_mse += (d * d * m).sum().item()
        tot_mae += (d.abs() * m).sum().item()
        tot_w += w
    rmse = float(np.sqrt(tot_mse / tot_w))
    mae = float(tot_mae / tot_w)
    return rmse, mae


# ============================ 5. 训练 / 测试 / 预测 ============================
def plot_comparison(inp, gt, pr, err, save_path):
    """绘制 输入 / 真实 / 预测 / 误差 四联对比图。
    真实厚度与预测厚度共用同一颜色刻度（vmin/vmax），误差图则固定 0 为白色
    中心，以便直接判断正/负偏差。"""
    stack = np.concatenate([gt, pr])
    finite = stack[np.isfinite(stack)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0

    err_finite = err[np.isfinite(err)]
    err_norm = None
    if err_finite.size:
        err_min = float(err_finite.min())
        err_max = float(err_finite.max())
        if err_min < 0.0 and err_max > 0.0:
            err_norm = matplotlib.colors.TwoSlopeNorm(vcenter=0.0,
                                                     vmin=err_min,
                                                     vmax=err_max)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    axs[0].imshow(inp)
    axs[0].set_title('输入 RGB')
    im1 = axs[1].imshow(gt, cmap='viridis', vmin=lo, vmax=hi)
    axs[1].set_title('真实厚度 (cm)')
    im2 = axs[2].imshow(pr, cmap='viridis', vmin=lo, vmax=hi)
    axs[2].set_title('预测厚度 (cm)')
    im3 = axs[3].imshow(err, cmap='bwr', norm=err_norm)
    axs[3].set_title('误差 (cm)')
    plt.colorbar(im1, ax=axs[1])
    plt.colorbar(im2, ax=axs[2])
    plt.colorbar(im3, ax=axs[3])
    for a in axs:
        a.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close(fig)


def save_predictions(model, loader, device, out_dir, max_show=6):
    """保存若干测试样本的 输入 / 真实 / 预测 / 误差 可视化。"""
    model.eval()
    shown = 0
    with torch.no_grad():
        for x, y, m in loader:
            pred = model(x.to(device)).cpu()
            x, y, m = x.cpu(), y.cpu(), m.cpu()
            for i in range(x.shape[0]):
                if shown >= max_show:
                    return
                inp = x[i].permute(1, 2, 0).numpy()
                mk = m[i, 0].numpy() > 0.5
                gt = np.where(mk, y[i, 0].numpy() * H_MAX, np.nan)
                pr = np.where(mk, np.clip(pred[i, 0].numpy(), 0.0, None) * H_MAX,
                              np.nan)
                err = np.where(mk, pr - gt, np.nan)
                plot_comparison(inp, gt, pr, err,
                                os.path.join(out_dir, f'test_pred_{shown}.png'))
                shown += 1


def train(args, train_idx, val_idx, test_idx, images, thickness, mask):
    device = args.device
    train_ds = ThicknessDataset(images, thickness, mask, train_idx)
    val_ds = ThicknessDataset(images, thickness, mask, val_idx)
    test_ds = ThicknessDataset(images, thickness, mask, test_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers)

    model = UNet(in_ch=3, out_ch=1, base=args.base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=8, factor=0.5)

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, f'unet_{args.size}_best.pt')
    last_path = os.path.join(args.out_dir, f'unet_{args.size}_last.pt')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'[模型] 参数量: {n_params / 1e6:.2f} M  设备: {device}')
    print(f'[数据] 训练 {len(train_idx)} / 验证 {len(val_idx)} / 测试 {len(test_idx)}')

    best_val = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse_cm': [],
        'val_mae_cm': [],
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x)
            loss = masked_mse(pred, y, m)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run += loss.item()
            nb += 1
        train_loss = run / nb

        val_rmse, val_mae = evaluate(model, val_loader, device)
        val_loss = float((val_rmse / H_MAX) ** 2)
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse_cm'].append(val_rmse)
        history['val_mae_cm'].append(val_mae)
        print(f'  第{epoch:3d}/{args.epochs}轮  训练MSE={train_loss:.5f}  '
              f'验证RMSE={val_rmse * 10:.3f} mm  验证MAE={val_mae * 10:.3f} mm',
              flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'state_dict': model.state_dict(),
                        'config': {'in_ch': 3, 'out_ch': 1, 'base': args.base,
                                   'size': args.size, 'H_MAX': H_MAX,
                                   'seed': args.seed, 'lr': args.lr},
                        'history': history,
                        'val_rmse_cm': val_rmse, 'val_mae_cm': val_mae},
                       best_path)
            print(f'        -> 最佳模型已保存: {best_path}')
        torch.save({'state_dict': model.state_dict(),
                    'config': {'in_ch': 3, 'out_ch': 1, 'base': args.base,
                               'size': args.size, 'H_MAX': H_MAX,
                               'seed': args.seed, 'lr': args.lr},
                    'history': history,
                    'val_rmse_cm': val_rmse, 'val_mae_cm': val_mae},
                   last_path)

    # 载入最佳模型评估测试集
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    test_rmse, test_mae = evaluate(model, test_loader, device)
    print(f'\n[测试] RMSE = {test_rmse:.4f} cm（{test_rmse * 10:.3f} mm）  '
          f'MAE = {test_mae:.4f} cm（{test_mae * 10:.3f} mm）')

    save_predictions(model, test_loader, device, args.out_dir)
    print(f'[测试] 预测可视化已保存到 {args.out_dir}/test_pred_*.png')

    meta = {
        'best_val_rmse_cm': ckpt['val_rmse_cm'],
        'best_val_mae_cm': ckpt['val_mae_cm'],
        'test_rmse_cm': test_rmse,
        'test_mae_cm': test_mae,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'H_MAX': H_MAX,
        'config': ckpt['config'],
        'history': history,
    }
    with open(os.path.join(args.out_dir, 'metadata.json'), 'w',
              encoding='utf-8') as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)
    print(f'[模型] 元数据已保存: {os.path.join(args.out_dir, "metadata.json")}')


def predict(args, model_path):
    """用已训练模型对单张图像（PNG/JPG）预测厚度图；
    若同目录存在同名真实厚度图（*_deg.tiff），一并生成对比图
    （真实/预测共用同一颜色刻度）。"""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'未找到模型 {model_path}，请先运行训练。')
    device = args.device
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    hmax = cfg.get('H_MAX', H_MAX)
    model = UNet(in_ch=cfg['in_ch'], out_ch=cfg['out_ch'],
                 base=cfg['base']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    img = Image.open(args.predict)
    arr = resize_image_rgb(img, cfg['size'])
    x = torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32) / 255.0)
    x = x[None].to(device)
    with torch.no_grad():
        pred = model(x)[0, 0].cpu().numpy() * hmax
    pred = np.clip(pred, 0.0, None)   # 厚度不可能为负，截断底部伪影

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.predict))[0]
    out_tiff = os.path.join(args.out_dir, f'pred_{base}.tiff')
    tifffile.imwrite(out_tiff, pred.astype(np.float32))
    out_png = os.path.join(args.out_dir, f'pred_{base}.png')
    plt.imsave(out_png, pred, cmap='viridis')
    print(f'[预测] 输入: {args.predict}')
    print(f'[预测] 厚度图已保存: {out_tiff}  与  {out_png}')
    print(f'[预测] 厚度范围: {np.nanmin(pred):.4f} ~ {np.nanmax(pred):.4f} cm')

    # 若同目录存在同名真实厚度图（如 9.4_deg.tiff 对应 9.4_deg (3).png），
    # 生成与 save_predictions 相同的四联对比图
    key = re.sub(r'\s*\(\d+\)$', '', base)
    gt_path = os.path.join(os.path.dirname(args.predict), key + '.tiff')
    if os.path.isfile(gt_path):
        gt = tifffile.imread(gt_path).astype(np.float32)
        if gt.shape != (cfg['size'], cfg['size']):
            gt = resize_gt_thickness(gt, cfg['size'])
        mk = np.isfinite(gt)
        gt = np.where(mk, gt, np.nan)
        pr = np.where(mk, pred, np.nan)
        err = np.where(mk, pr - gt, np.nan)
        compare_path = os.path.join(args.out_dir, f'pred_{base}_compare.png')
        plot_comparison(arr, gt, pr, err, compare_path)
        print(f'[预测] 对比图已保存: {compare_path}（真实/预测共用同一颜色刻度）')
    else:
        print(f'[预测] 未找到同名真实厚度图 {gt_path}，跳过对比图。')


# ============================ 6. 主程序 ============================
def main():
    ap = argparse.ArgumentParser(description='U-Net 液膜厚度回归项目')
    ap.add_argument('--data-dir', default='pre_analysis', help='预处理后图像/标签所在目录')
    ap.add_argument('--cache-dir', default='preprocessed', help='预处理缓存目录')
    ap.add_argument('--out-dir', default='models', help='模型与结果保存子文件夹')
    ap.add_argument('--size', type=int, default=384, help='模型输入尺寸（须为16的倍数，需与预处理一致）')
    ap.add_argument('--base', type=int, default=16, help='U-Net 基础通道数')
    ap.add_argument('--epochs', type=int, default=30, help='训练轮数')
    ap.add_argument('--batch-size', type=int, default=8, help='批大小')
    ap.add_argument('--lr', type=float, default=1e-3, help='学习率')
    ap.add_argument('--seed', type=int, default=42, help='随机种子')
    ap.add_argument('--split-ratio', type=float, default=0.8,
                    help='训练集角度占比（其余为测试）')
    ap.add_argument('--val-ratio', type=float, default=0.1,
                    help='训练集中验证集角度占比')
    ap.add_argument('--workers', type=int, default=0, help='DataLoader 线程数')
    ap.add_argument('--rebuild-cache', action='store_true',
                    help='强制重建预处理缓存')
    ap.add_argument('--predict', type=str, default=None,
                    help='预测模式：传入一张图像路径（如 pre_analysis/*.png）')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[设备] {args.device}')

    images, thickness, mask, angle_keys, pairs = preprocess(
        args.data_dir, args.cache_dir, args.size, args.rebuild_cache)
    angle_keys = [k.decode() if isinstance(k, bytes) else k for k in angle_keys]

    if args.predict:
        predict(args, os.path.join(args.out_dir, f'unet_{args.size}_best.pt'))
        return

    train_idx, val_idx, test_idx, sets = split_indices(
        angle_keys, args.seed, args.split_ratio, args.val_ratio)
    n_train_a, n_val_a, n_test_a = (len(s) for s in sets)
    print(f'[数据] 角度划分: 训练 {n_train_a} / 验证 {n_val_a} / 测试 {n_test_a}')
    print(f'[数据] 样本划分: 训练 {len(train_idx)} / 验证 {len(val_idx)} '
          f'/ 测试 {len(test_idx)}')

    train(args, train_idx, val_idx, test_idx, images, thickness, mask)


if __name__ == '__main__':
    main()