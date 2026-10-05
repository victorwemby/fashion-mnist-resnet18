# 可解释性与消融实验

## Grad-CAM

Grad-CAM 用最后一个卷积层的梯度生成热力图，显示模型用于分类的图像区域。

```powershell
.\.venv\Scripts\python.exe src\gradcam.py --checkpoint artifacts\best.pt --image artifacts\fashion_sample.png --output artifacts\gradcam.png
```

输出：`artifacts/gradcam.png`。

## 消融实验

同一数据划分、同一训练轮数下比较：

```powershell
.\.venv\Scripts\python.exe src\ablation.py --runs 1 --output-dir artifacts\ablation
```

三组设置：

- `full_finetuning`：完整微调 + 数据增强
- `no_augmentation`：完整微调，不使用随机翻转和裁剪
- `frozen_backbone`：冻结 ResNet-18，只训练最后分类层

结果文件：`artifacts/ablation/ablation_results.csv`。

解释时关注验证 Accuracy 和 Macro-F1 的差异，并说明实验只改变一个因素。正式报告应使用多次运行或更多 epoch 后的真实结果。
