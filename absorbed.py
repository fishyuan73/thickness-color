# -*- coding: utf-8 -*-
"""
基于 thickness.py 的颜色仿真程序（6500K 白光 + 罗丹明 B 吸收 → RGB → JPG）
--------------------------------------------------------------------
物理过程：
    1. 底部白光由 6500K 色温的白光源产生（Planck 黑体辐射近似日光/白光，
       见 illuminant_spd，光谱平滑、蓝端略强）。
    2. 白光向上穿过含有罗丹明 B (Rhodamine B) 溶液的液膜，发生
       Beer-Lambert 吸收： I(λ) = I0(λ) · 10^(-ε(λ)·c·h)
    3. 到达相机的光谱经 CIE 1931 2° 标准观察者色匹配函数耦合（积分）到
       XYZ，再做白点适配（白场映射到 D65）与 sRGB 矩阵变换，得到 RGB 像素值。
    4. 复用 thickness.py 的楔形液膜几何（培养皿椭圆掩膜 + 深度场），为每个
       倾斜角生成一张模拟相机拍摄的 JPG 图像；并顺时针旋转 6°（与 thickness.py
       一致），使倾斜轴与横轴成 6° 夹角。
    5. 由于实际相机白平衡/校正问题，"白光（无吸收）"被记录为 sRGB(164,167,202)
       而非纯白 (255,255,255)。本程序以该背景色作为颜色学白点：无液/背景区域
       （h=0，透射率=1）恰好渲染为 (164,167,202)，与实际拍摄一致。

输出：
    figure/spectra_6500k_rhodamine.png        —— 6500K 光源光谱、罗丹明 B 消光系数、透射率、透射光谱
    figure/color_vs_thickness_rhodamine.png   —— sRGB 颜色随液膜厚度的色带（标定曲线）
    figure/rhodamine_6500k_theta_XX.X.jpg     —— 各倾斜角对应的模拟相机图像
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# matplotlib 中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========================= 实验几何参数（与 thickness.py 一致） =========================
R = 1.586           # 培养皿半径 (cm)
D = 0.114           # 培养皿底部厚度 (cm)
V = 2.3             # 液体体积 (cm^3 = mL)
d = V / (np.pi * R ** 2)   # 水平放置时的液深 (cm)

c_dye = 8e-6        # 罗丹明 B 浓度 (mol/L)，可调
resolution = 109.7   # 空间分辨率（像素/单位长度）
theta_range = np.arange(0.0, 9.7, 0.1)   # 倾斜角扫描（度）
ROT_DEG = 6.0       # 图像顺时针旋转角度（度），使倾斜轴与横轴成 6° 夹角
OUT_DIR = 'pre_analysis'
INFO_DIR='figure'

# ========================= 光谱波长网格（5 nm，覆盖可见光 380~780 nm） =========================
LAM = np.arange(380, 781, 5, dtype=float)


def illuminant_spd(lam, T=6500.0):
    """6500K 白光光源的相对光谱（Planck 黑体辐射，近似日光/白光）。
    色温 6500K 的光谱平滑、蓝端略强，是常见"白光"标准。lam 单位 nm，
    返回相对辐射功率。"""
    lam_um = lam / 1000.0                 # nm → µm
    c2 = 1.438776877e4                    # hc/k (µm·K)
    return lam_um ** -5.0 / np.expm1(c2 / (lam_um * T))


def rhodamine_epsilon(lam):
    """罗丹明 B (Rhodamine B) 在水中的摩尔消光系数近似 ε(λ) (M^-1·cm^-1)。
    特征：绿光区强吸收（主峰 ~553 nm，肩峰 ~520 nm），红光区 (>600 nm)
    与蓝光区吸收均很弱 —— 因此溶液呈玫红/品红色。
    说明：实际使用时可用分光光度计实测数据替换本函数。"""
    g = lambda l0, s: np.exp(-0.5 * ((lam - l0) / s) ** 2)
    return (100000 * g(553, 22)   # 单体主峰（水中 ~553 nm）
            + 45000 * g(520, 20)  # 振动肩峰
            + 12000 * g(460, 40)  # 蓝绿区宽带吸收
            + 9000 * g(360, 45))  # 近紫外区微量吸收


# ================ CIE 1931 2° 标准观察者色匹配函数（5 nm 间隔，380~780 nm） ================
CIE_X = np.array([
    0.001368, 0.002236, 0.004243, 0.00765, 0.01431, 0.02319, 0.04351, 0.07763,
    0.13438, 0.21477, 0.2839, 0.3285, 0.34828, 0.34806, 0.3362, 0.3187, 0.2908,
    0.2511, 0.19536, 0.1421, 0.09564, 0.05795, 0.03201, 0.0147, 0.0049, 0.0024,
    0.0093, 0.0291, 0.06327, 0.1096, 0.1655, 0.22575, 0.2904, 0.3597, 0.43345,
    0.51205, 0.5945, 0.6784, 0.7621, 0.8425, 0.9163, 0.9786, 1.0263, 1.0567,
    1.0622, 1.0456, 1.0026, 0.9384, 0.85445, 0.7514, 0.6424, 0.5419, 0.4479,
    0.3608, 0.2835, 0.2187, 0.1649, 0.1212, 0.0874, 0.0636, 0.04677, 0.0329,
    0.0227, 0.01584, 0.011359, 0.008111, 0.00579, 0.004109, 0.002899, 0.002049,
    0.00144, 0.001, 0.00069, 0.000476, 0.000332, 0.000235, 0.000166, 0.000117,
    0.000083, 0.000059, 0.000042])
CIE_Y = np.array([
    0.000039, 0.000064, 0.00012, 0.000217, 0.000396, 0.00064, 0.00121, 0.00218,
    0.004, 0.0073, 0.0116, 0.01684, 0.023, 0.0298, 0.038, 0.048, 0.06, 0.0739,
    0.09098, 0.1126, 0.13902, 0.1693, 0.20802, 0.2586, 0.323, 0.4073, 0.503,
    0.6082, 0.71, 0.7932, 0.862, 0.91485, 0.954, 0.9803, 0.99495, 1.0, 0.995,
    0.9786, 0.952, 0.9154, 0.87, 0.8163, 0.757, 0.6949, 0.631, 0.5668, 0.503,
    0.4412, 0.381, 0.321, 0.265, 0.217, 0.175, 0.1382, 0.107, 0.0816, 0.061,
    0.04458, 0.032, 0.0232, 0.017, 0.01192, 0.00821, 0.005723, 0.004102,
    0.002929, 0.002091, 0.001484, 0.001047, 0.00074, 0.00052, 0.000361,
    0.000249, 0.000172, 0.00012, 0.000085, 0.00006, 0.000042, 0.00003,
    0.000021, 0.000015])
CIE_Z = np.array([
    0.00645, 0.01055, 0.02005, 0.03621, 0.06785, 0.1102, 0.2074, 0.3713, 0.6456,
    1.03905, 1.3856, 1.62296, 1.74706, 1.7826, 1.77211, 1.7441, 1.6692, 1.5281,
    1.28764, 1.0419, 0.81295, 0.6162, 0.46518, 0.3533, 0.272, 0.2123, 0.1582,
    0.1117, 0.07825, 0.05725, 0.04216, 0.02984, 0.0203, 0.0134, 0.00875,
    0.00575, 0.0039, 0.00275, 0.0021, 0.0018, 0.00165, 0.0014, 0.0011, 0.001,
    0.0008, 0.0006, 0.00034, 0.00024, 0.00019, 0.0001, 0.00005, 0.00003,
    0.00002, 0.00001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0])
#人眼对不同波长的敏感度，CIE 1931 2° 标准观察者色匹配函数
# ========================= 光谱 → XYZ → sRGB 转换 =========================
# 6500K 白光（未吸收）的 XYZ
S_white = illuminant_spd(LAM)
Xw = float(np.sum(S_white * CIE_X))
Yw = float(np.sum(S_white * CIE_Y))
Zw = float(np.sum(S_white * CIE_Z))

SRGB_M = np.array([[3.2406, -1.5372, -0.4986],
                   [-0.9689, 1.8758, 0.0415],
                   [0.0557, -0.2040, 1.0570]])


def srgb_to_xyz(rgb):
    """sRGB(0~1) → XYZ（sRGB 线性化 + 逆 sRGB 矩阵，白点按 D65）。
    与 xyz_to_srgb 的 rgb = xyz @ SRGB_M.T 行向量约定一致，
    故逆变换用 c @ inv(SRGB_M).T。"""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return c @ np.linalg.inv(SRGB_M).T


# ================== 实际拍摄背景色（相机白平衡/校正偏移） ==================
# 真实相机把"白光（无吸收）"记录为 sRGB(164,167,202) 而非 (255,255,255)。
# 以该背景色作为本仿真的颜色学白点：无液/背景区域（h=0，T=1）将恰好渲染为
# (164,167,202)；液膜吸收色也按同一白点计算，与实际拍摄一致。
# 如需其它底色，仅需修改 BG_SRGB8。
BG_SRGB8 = np.array([164, 167, 202])      # 背景 sRGB（8-bit，0~255）
BG_XYZ = srgb_to_xyz(BG_SRGB8 / 255.0)    # 背景色对应的 XYZ

# 白点归一化系数：把 6500K 白光映射到背景色 BG_XYZ（而非标准 D65 白）
WHITE_SCALE = BG_XYZ / np.array([Xw, Yw, Zw])


def spd_to_xyz(spd):
    """逐行光谱(..., N) → XYZ(..., 3)，N 与 LAM 等长。"""
    return np.stack([np.sum(spd * CIE_X, axis=-1),
                     np.sum(spd * CIE_Y, axis=-1),
                     np.sum(spd * CIE_Z, axis=-1)], axis=-1)


def xyz_to_srgb(xyz):
    """XYZ(..., 3) → 线性 sRGB(..., 3)，含 sRGB 伽马编码。"""
    rgb = xyz @ SRGB_M.T
    return np.where(rgb <= 0.0031308,
                    12.92 * rgb,
                    1.055 * np.maximum(rgb, 0.0) ** (1.0 / 2.4) - 0.055)


def transmittance_of(h):
    """给定液膜厚度 h (cm) 的透射光谱 T(λ) = 10^(-ε·c·h)。
    标量 h 返回 (N,)，数组 h 返回 (..., N)。"""
    h = np.asarray(h, dtype=float)
    k = rhodamine_epsilon(LAM) * c_dye   # (N,)
    if h.ndim == 0:
        return 10.0 ** (-k * h)         # (N,)
    return 10.0 ** (-k[None, :] * h[..., None])   # (..., N)


def thickness_to_srgb(h):
    """液膜厚度 h (cm) → sRGB(0~1)。h<=0 视为无液（透射率=1，即白光）。"""
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    T = transmittance_of(h)              # (..., N)
    xyz = spd_to_xyz(S_white * T) * WHITE_SCALE
    rgb = xyz_to_srgb(xyz)
    return np.clip(rgb, 0.0, 1.0)


# ========================= 图像顺时针旋转（与 thickness.py 一致） =========================
def rotate_cw(img, deg, cval=0.0):
    """将二维数组 img 顺时针旋转 deg 度（双线性插值，画布自动扩展以包含全部内容）。

    屏幕坐标系（y 向下）中顺时针旋转 deg 度的正变换：
        x' =  x·cosθ - y·sinθ
        y' =  x·sinθ + y·cosθ
    用其逆变换（输出→输入）对源图像采样实现旋转。"""
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


def rotate_cw_rgb(img, deg, cval=0.0):
    """将 RGB 图像 (H,W,3) 顺时针旋转 deg 度（逐通道旋转，边界填充 cval）。
    cval 可为标量或长度=3 的数组（每通道各一个填充值）。"""
    cval = np.broadcast_to(np.asarray(cval, dtype=float), (3,))
    return np.stack([rotate_cw(img[..., ch], deg, cval[ch])
                     for ch in range(img.shape[-1])], axis=-1)


# ========================= 渲染单帧（复用 thickness.py 的几何） =========================
def render_frame(theta_deg):
    """按倾斜角渲染相机图像，返回 (RGB uint8 图像 (H,W,3), 深度场 h (H,W))。"""
    theta = np.radians(theta_deg)
    # 从正上方看，培养皿是椭圆：a = R·cosθ（x 方向），b = R（y 方向）
    a = R * np.cos(theta)
    b = R
    width = max(int(2 * a * resolution), 1)
    height = max(int(2 * b * resolution), 1)

    x_vals = np.linspace(0, 2 * a, width)   # x：距最左端的距离
    y_vals = np.linspace(-b, b, height)
    X, Y = np.meshgrid(x_vals, y_vals)
    mask = ((X - a) ** 2 / a ** 2 + Y ** 2 / b ** 2) <= 1   # 椭圆掩膜（皿内）

    # 液面深度（与 thickness.py 公式完全一致，单位 cm）
    h = ((d + D) * np.cos(theta) + R * np.sin(theta)
         - X * np.tan(theta) - D / np.cos(theta))
    # 皿外与干区（h<0，无液）均视为 0 光程，直接看到白光源
    h = np.where(mask, h, 0.0)
    h = np.clip(h, 0.0, None)

    # 分批计算透射光谱→sRGB，控制峰值内存
    h_flat = h.ravel()
    rgb = np.empty((h.size, 3), dtype=np.float64)
    n_chunk = 100_000
    for i0 in range(0, h.size, n_chunk):
        hc = h_flat[i0:i0 + n_chunk]
        T = 10.0 ** (-(rhodamine_epsilon(LAM) * c_dye)[None, :] * hc[:, None])
        xyz = spd_to_xyz(S_white[None, :] * T) * WHITE_SCALE
        rgb[i0:i0 + n_chunk] = np.clip(xyz_to_srgb(xyz), 0.0, 1.0)

    img = (rgb.reshape(height, width, 3) * 255).astype(np.uint8)
    # 顺时针旋转 ROT_DEG 度（与 thickness.py 一致），使倾斜轴与横轴成 6° 夹角
    # 皿外为 0 光程（透射=白光），RGB 边界用实际背景色 BG_SRGB8 填充；
    # 深度场边界用 0 填充
    img = np.clip(rotate_cw_rgb(img.astype(np.float64), ROT_DEG, cval=BG_SRGB8),
                  0, 255).astype(np.uint8)
    h = rotate_cw(h, ROT_DEG)
    return img, h


# ========================= 光谱诊断图 =========================
def plot_spectra():
    h_list = [0.03, d, 0.20]   # 薄 / 平均水平 / 厚
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('6500K 白光 + 罗丹明 B 吸收 → 透射光谱（Beer-Lambert）', fontsize=14)

    # (1) 6500K 白光光谱
    ax[0, 0].plot(LAM, S_white, 'k-', lw=1.5)
    ax[0, 0].fill_between(LAM, S_white, color='tab:blue', alpha=0.3)
    ax[0, 0].set_title('6500K 黑体白光光谱 S(λ)')
    ax[0, 0].set_xlabel('波长 λ (nm)'); ax[0, 0].set_ylabel('相对强度')

    # (2) 罗丹明 B 消光系数
    ax[0, 1].plot(LAM, rhodamine_epsilon(LAM), 'b-', lw=1.5)
    ax[0, 1].set_title(f'罗丹明 B 摩尔消光系数 ε(λ)，c={c_dye:.0e} mol/L')
    ax[0, 1].set_xlabel('波长 λ (nm)'); ax[0, 1].set_ylabel(r'ε (M$^{-1}$·cm$^{-1}$)')

    # (3) 不同厚度的透射率
    for hh in h_list:
        ax[1, 0].plot(LAM, transmittance_of(hh), lw=1.5,
                      label=f'h = {hh:.3f} cm')
    ax[1, 0].set_ylim(0, 1.05)
    ax[1, 0].set_title('透射率 T(λ) = 10^(-ε·c·h)')
    ax[1, 0].set_xlabel('波长 λ (nm)'); ax[1, 0].set_ylabel('T')
    ax[1, 0].legend(fontsize=9)

    # (4) 到达相机的透射光谱 + 对应 sRGB 色块
    for hh in h_list:
        ax[1, 1].plot(LAM, S_white * transmittance_of(hh), lw=1.5,
                      label=f'h = {hh:.3f} cm')
    ax[1, 1].set_title('相机采样到的光谱 S(λ)·T(λ)')
    ax[1, 1].set_xlabel('波长 λ (nm)'); ax[1, 1].set_ylabel('相对强度')
    ax[1, 1].legend(fontsize=9)
    # 叠加 sRGB 色块
    y0 = ax[1, 1].get_ylim()[1]
    for i, hh in enumerate(h_list):
        c = thickness_to_srgb(hh)
        ax[1, 1].add_patch(plt.Rectangle((380 + i * 130, y0 * 0.82), 110, y0 * 0.14,
                                         facecolor=c, edgecolor='k'))
        ax[1, 1].text(380 + i * 130 + 55, y0 * 0.99, f'{hh:.3f} cm',
                      ha='center', fontsize=9)

    plt.tight_layout()
    return fig


def plot_color_strip():
    """sRGB 颜色随液膜厚度的色带（标定曲线）。"""
    h_axis = np.linspace(0, 0.25, 256)
    rgb = thickness_to_srgb(h_axis)          # (256, 3)
    strip = np.tile(rgb[None, :, :], (60, 1, 1))
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.imshow(strip, aspect='auto', extent=[0, 0.25, 0, 1])
    ax.set_xticks(np.arange(0, 0.251, 0.025))
    ax.set_yticks([])
    ax.set_title('sRGB 颜色 ~ 液膜厚度 h（罗丹明 B，Beer-Lambert + 光谱耦合）')
    ax.set_xlabel('液膜厚度 h (cm)')
    plt.tight_layout()
    return fig


# ========================= 主程序 =========================
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'培养皿: R={R} cm, D={D} cm, V={V} mL, 水平液深 d={d:.4f} cm')
    print(f'罗丹明 B 浓度 c={c_dye:.1e} mol/L')
    print(f'光源: 6500K 黑体白光（Planck）')
    print(f'罗丹明 B 吸收峰: ~553 nm (ε≈100000 M^-1 cm^-1)')
    print(f'背景色（实际相机白场）: sRGB{tuple(int(v) for v in BG_SRGB8)}')
    print(f'图像顺时针旋转: {ROT_DEG:.0f}°')

    # 校验：若干代表性厚度的 sRGB 值
    print('\n厚度→sRGB 校验:')
    for hh in [0.0, d / 3, d, 2 * d]:
        rgb8 = np.round(thickness_to_srgb(hh) * 255).astype(int)
        print(f'  h={hh:.4f} cm  ->  R={rgb8[0]:3d}  G={rgb8[1]:3d}  B={rgb8[2]:3d}')

    # 光谱诊断图
    plot_spectra().savefig(os.path.join(INFO_DIR, 'spectra_6500k_rhodamine.png'), dpi=150)
    plot_color_strip().savefig(os.path.join(INFO_DIR, 'color_vs_thickness_rhodamine.png'), dpi=150)
    print(f'\n已保存光谱图: {INFO_DIR}/spectra_6500k_rhodamine.png')
    print(f'已保存色带图: {INFO_DIR}/color_vs_thickness_rhodamine.png')

    # 逐角度渲染并保存 JPG
    print('\n渲染 JPG:')
    for theta_deg in theta_range:
        img, _ = render_frame(theta_deg)
        fname = os.path.join(OUT_DIR, f'{theta_deg:.1f}_deg.jpg')
        Image.fromarray(img).save(fname, quality=95)
        print(f'  theta={theta_deg:.1f}° -> {fname}  ({img.shape[0]}x{img.shape[1]})')
