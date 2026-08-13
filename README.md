# thickness-color
基于培养皿俯拍 RGB 图像进行液膜厚度回归的深度学习项目。将图像数据与仿真/真实厚度图对应，并使用 U-Net 网络预测每个像素点的液膜厚度。
## 项目概述
主要流程：
- 预处理拍摄的图像与深度图，形成训练样本
- 生成或准备液膜几何与厚度标签数据
- 使用 U-Net 回归液膜厚度图
- 使用训练好的模型进行单张图像预测
- 可选使用物理信息约束版本 `u-net_phy.py`

本项目中的核心脚本：
- `absorbed.py`：光谱/颜色仿真，计算吸收和 RGB 颜色映射
- `thickness.py`：生成液膜厚度图/深度图
- `u-net.py`：主 U-Net 训练与推理脚本
- `u-net_phy.py`：物理信息 U-Net 版本

首先需要使用`thickness.py`自行生成深度图对应的tiff到`pre_analysis`对应文件夹下，具体参见示意图。

## 需求
见requirement.txt
 python==3.12.13
 tifffile==2023.7.10
 numpy==2.5.1
 scipy==1.18.0
 pandas==3.0.5
 Pillow==12.3.0
 matplotlib==3.11.1
 colorcet==3.2.1
 tensorflow==2.21.0
 ipython==9.15.0
 torch==2.13.0
## 用法
主训练脚本：

```bash
python u-net.py
```

可选自定义参数：

```bash
python u-net.py --epochs 60 --batch-size 8
python u-net.py --data-dir pre_analysis --out-dir models --size 384
```

脚本会：
- 读取图像和标签对
- 按角度进行数据集划分
- 训练 U-Net 回归网络
- 保存最佳模型权重到 `models/`
- 输出训练过程中的评估指标（如 RMSE / MAE）

使用训练好的模型对单张图像做推理：
```bash
python u-net.py --predict "pre_analysis/3.0_deg (1).png"
```
