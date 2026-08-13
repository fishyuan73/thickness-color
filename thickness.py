import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imwrite

R = 1.586      # 培养皿半径
D = 0.114       # 培养皿厚度
V = 2.3      # 加入液体体积

d = V / (np.pi * R**2)  # 液体厚度

resolution = 109.7  # 空间分辨率（像素/单位长度）
ROT_DEG = 6.0     # 深度图顺时针旋转角度（度），使倾斜轴与横轴成 6° 夹角
OUT_DIR = 'pre_analysis'   # 输出子文件夹

def rotate_cw(img, deg, cval=0.0):
    """将二维数组 img 顺时针旋转 deg 度（双线性插值，画布自动扩展以包含全部内容）。

    屏幕坐标系（y 向下）中顺时针旋转 deg 度的正变换：
        x' =  x·cosθ - y·sinθ
        y' =  x·sinθ + y·cosθ
    用其逆变换（输出→输入）对源图像采样实现旋转。
    """
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)

    H, W = img.shape
    # 输入四角（x 向右，y 向下），顺时针旋转后的角点
    corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], dtype=float)
    rot = corners @ np.array([[c, s], [-s, c]])      # x'=c·x−s·y, y'=s·x+c·y
    xmin, ymin = rot[:, 0].min(), rot[:, 1].min()
    xmax, ymax = rot[:, 0].max(), rot[:, 1].max()
    W2 = max(int(round(xmax - xmin)) + 1, 1)
    H2 = max(int(round(ymax - ymin)) + 1, 1)

    # 输出画布网格（旋转后的坐标）
    X2, Y2 = np.meshgrid(np.linspace(xmin, xmax, W2),
                         np.linspace(ymin, ymax, H2))
    # 逆变换得到源坐标
    xs = c * X2 + s * Y2
    ys = -s * X2 + c * Y2

    # 双线性插值
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = xs - x0
    fy = ys - y0

    def sample(dx, dy):
        x = x0 + dx
        y = y0 + dy
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        xc = np.clip(x, 0, W - 1)
        yc = np.clip(y, 0, H - 1)
        return np.where(ok, img[yc, xc], cval)

    return (sample(0, 0) * (1 - fx) * (1 - fy)
            + sample(1, 0) * fx * (1 - fy)
            + sample(0, 1) * (1 - fx) * fy
            + sample(1, 1) * fx * fy)


os.makedirs(OUT_DIR, exist_ok=True)

for theta_deg in np.arange(0, 9.7, 0.1):
    theta = np.radians(theta_deg)

    # 从正上方看，培养皿是椭圆：a=R*cos(theta)，b=R
    a = R * np.cos(theta)  # 椭圆半短轴（x方向，倾斜方向）
    b = R                  # 椭圆半长轴（y方向，垂直于倾斜方向）

    # 图像尺寸（像素）
    width = max(int(2 * a * resolution), 1)
    height = max(int(2 * b * resolution), 1)

    # 生成坐标网格，x为距离最左端的距离
    x_vals = np.linspace(0, 2 * a, width)   # x从最左端(0)到最右端(2a)
    y_vals = np.linspace(-b, b, height)
    X, Y = np.meshgrid(x_vals, y_vals)

    # 椭圆掩膜（中心在(a, 0)）
    mask = ((X - a)**2 / a**2 + Y**2 / b**2) <= 1

    # 液面深度（不修改已有公式）
    h = (d + D) * np.cos(theta) + R * np.sin(theta) \
        - X * np.tan(theta) - D / np.cos(theta)  # 液面深度

    # 椭圆外用NaN填充
    depth_map = np.where(mask, h, np.nan)

    # 顺时针旋转 6°（先填 NaN 再旋转，再按掩膜恢复 NaN，
    # 避免插值把 NaN 拖入皿内区域）
    valid = np.isfinite(depth_map)
    filled = np.where(valid, depth_map, 0.0)
    rotated = rotate_cw(filled, ROT_DEG)
    depth_map = np.where(rotate_cw(valid.astype(np.float64), ROT_DEG) > 0.5,
                         rotated, np.nan)

    # 统一补齐到 TARGET_SIZE×TARGET_SIZE（=384，与预处理图像画布一致）：
    # 不改动皿内分辨率（不缩放），仅在四周用 NaN 填充空值。
    TARGET_SIZE = 384
    ph = TARGET_SIZE - depth_map.shape[0]
    pw = TARGET_SIZE - depth_map.shape[1]
    if ph < 0 or pw < 0:
        raise ValueError(f'theta={theta_deg:.1f}°: 深度图 '
                         f'{depth_map.shape} 大于目标 {TARGET_SIZE}×{TARGET_SIZE}！')
    depth_map = np.pad(depth_map,
                       ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
                       mode='constant', constant_values=np.nan)

    # 保存存储了深度信息的TIFF到 pre_analysis 子文件夹
    fname = os.path.join(OUT_DIR, f'{theta_deg:.1f}_deg.tiff')
    imwrite(fname, depth_map.astype(np.float32))
    print(f'theta={theta_deg:.1f}°: 保存 {fname}, '
          f'尺寸={depth_map.shape[0]}×{depth_map.shape[1]}, a={a:.3f}, b={b:.3f}')