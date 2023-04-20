# encoding: utf-8
"""
@author: Justin
@time: 2023/4/20 20:59
@file: config.py
@function: 所有配置信息
"""
from typing import *
import numpy as np

# --------------------------------- #
#                                   #
#         训练需要的配置              #
#                                   #
# --------------------------------- #

# ==========必须配置================
# 1. 是否使用gpu
cuda = False
# 2. 训练的类别
num_classes = 21
# 3. 训练世代
UnFreeze_Epoch = 100
# 4. 批次大小
Unfreeze_batch_size = 4
# 5. 是否使用预训练权重
pretrained = False
# 6. 预训练权重位置
model_path = ""
# 7. 输入图片大小
input_shape = [512, 512]
# 8. 主干网络mobilenet、xception
backbone = "mobilenet"
# 9. 下采样倍率（8或者16倍）
downsample_factor = 16
# 10. 优化器，可选的有adam、sgd
optimizer_type = "sgd"
# 11. 模型最大学习率，当使用Adam优化器时建议设置  Init_lr=5e-4 当使用SGD优化器时建议设置   Init_lr=7e-3
Init_lr = 7e-3
# 12. 模型的最小学习率，默认为最大学习率的0.01
Min_lr = Init_lr * 0.01
# 13. 优化器内部使用到的momentum参数
momentum = 0.9
# 14. 权值衰减，可防止过拟合  adam会导致weight_decay错误，使用adam时建议设置为0。
weight_decay = 1e-4
# 15. 学习率下降方式，可选的有'step'、'cos'
lr_decay_type = 'cos'
# 16. 多少个epoch保存一次权值
save_period = 5
# 17. 权值与日志文件保存的文件夹
save_dir = 'logs'
# 18. 是否在训练时进行评估，评估对象为验证集
eval_flag = True
# 19. 代表多少个epoch评估一次，不建议频繁的评估
eval_period = 5
# 20. VOC devkit_path  数据集路径
VOCdevkit_path = 'VOCdevkit'
# 21. 建议选项：
#  种类少（几类）时，设置为True
#  种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
#  种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
dice_loss = False
# 22. 是否使用多线程读取数据  1代表关闭多线程
num_workers = 4
# ==========可选配置================
# 1. 是否采用分布式
distributed = False
# 2. 是否使用多卡
sync_bn = False
# 3. 是否使用混合精度训练
fp16 = False
# 4.冻结训练
Freeze_Train = False
# 5. 冻结训练参数
Init_Epoch = 0
Freeze_Epoch = 50
Freeze_batch_size = 8
# 6. 是否使用focal loss来防止正负样本不平衡
focal_loss = False
# 7. 是否给不同种类赋予不同的损失权值，默认是平衡的。
#   设置的话，注意设置成numpy形式的，长度和num_classes一样。
#   如：
#   num_classes = 3
#   cls_weights = np.array([1, 2, 3], np.float32)
cls_weights = np.ones([num_classes], np.float32)

