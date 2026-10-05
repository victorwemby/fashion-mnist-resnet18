# Grad-CAM 案例：d02d510d1a241b34bb4ef477ea5bdf9b

## 输入

桌面图片：`d02d510d1a241b34bb4ef477ea5bdf9b.png`

## 生成命令

```powershell
.\.venv\Scripts\python.exe src\gradcam.py `
  --checkpoint artifacts\best.pt `
  --image "C:\Users\wxhdq\Desktop\d02d510d1a241b34bb4ef477ea5bdf9b.png" `
  --output artifacts\gradcam_d02d510d1a241b34bb4ef477ea5bdf9b.png
```

## 如何解读

Grad-CAM 将 ResNet-18 最后卷积层的梯度投影回图片空间。红色区域表示对当前预测贡献较大的区域，蓝色区域表示贡献较小的区域。分析时检查热力图是否集中在服饰主体，而不是背景。

## 申请材料写法

`I used Grad-CAM to inspect whether the classifier focused on the clothing object when making its prediction. The visualization provided a qualitative check of the model's decision process and exposed potential sensitivity to background and input distribution.`

## 限制

Fashion-MNIST 训练图片是 28×28 灰度图，因此彩色商品图的热力图只能作为分布偏移案例，不能证明模型具备通用商品识别能力。
