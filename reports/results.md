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

## 训练曲线显示说明

本项目的记录结果来自一次快速的 **1 epoch CPU run**。因此，`artifacts/history.csv`
中只有一行训练记录，训练曲线图中每条曲线只有一个数据点。单个数据点没有相邻
epoch 可以连接，所以旧版图像可能看起来像没有曲线或没有明显的点；这不表示训练
失败，也不影响 `history.csv` 中保存的数值结果。

当前 `src/train.py` 已为曲线增加圆点标记、坐标轴和网格。若需要展示更完整的
变化趋势，可以在已有数据和权重的基础上重新运行 3 个 epoch：

```powershell
.\.venv\Scripts\python.exe src\train.py --epochs 3 --batch-size 64 --output-dir artifacts
```

重新运行后，`artifacts/training_curves.png` 会显示多个 epoch 的训练损失、验证损失
和准确率变化。申请材料中使用的 92.3% 测试准确率和 92.2% 测试 Macro-F1 来自
已保存的测试预测文件，与曲线图是否显示多个点是两个独立问题。

## 限制与改进方向

当前模型训练数据是 28×28 灰度单物体图片，因此对彩色商品图、复杂背景和多个物体的识别能力有限。四区域功能用于演示多区域推理，不等同于目标检测。

后续可以使用彩色鞋类数据集和 YOLO/SSD 等目标检测模型，输出每个物体的边界框、类别和置信度。
