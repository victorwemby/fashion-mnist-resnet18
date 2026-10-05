# 实验结果

## 实验设置

- 数据集：Fashion-MNIST，10 个服饰类别
- 模型：ImageNet 预训练 ResNet-18
- 训练轮数：1 epoch
- Batch size：64
- 设备：CPU
- 数据划分：训练集 80%，验证集 10%，测试集 10%

## 测试结果

| 指标 | 结果 |
|---|---:|
| Test Accuracy | 92.3% |
| Test Macro-F1 | 92.2% |
| Validation Accuracy | 93.1% |
| Validation Macro-F1 | 93.0% |

## 可操作演示

运行：

```powershell
.\.venv\Scripts\python.exe src\demo.py --checkpoint artifacts\best.pt
```

选择一张图片后，程序显示原图和整图预测；点击 `Analyze 4 regions` 可以查看四个区域的预测结果。

## 限制与改进方向

当前模型训练数据是 28×28 灰度单物体图片，因此对彩色商品图、复杂背景和多个物体的识别能力有限。四区域功能用于演示多区域推理，不等同于目标检测。

后续可以使用彩色鞋类数据集和 YOLO/SSD 等目标检测模型，输出每个物体的边界框、类别和置信度。
