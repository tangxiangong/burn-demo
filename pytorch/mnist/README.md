# MNIST 手写数字识别 - PyTorch 实现

这是一个使用 PyTorch 实现的 MNIST 手写数字识别卷积神经网络。该实现支持 Apple Metal (MPS) 加速，并提供了训练时间统计。

## 模型结构

模型结构如下：
- conv1: 卷积层 (3x3)
- conv2: 卷积层 (3x3)
- pool: 自适应平均池化 (8x8)
- dropout: Dropout层 (p=0.5)
- linear1: 全连接层 (1024->512)
- linear2: 全连接层 (512->10)
- activation: ReLU

## 使用方法

从项目根目录运行：

```bash
python -m pytorch.main
```

### 命令行参数

可以通过命令行参数自定义训练过程：

- `--batch-size`: 批次大小 (默认: 64)
- `--epochs`: 训练轮数 (默认: 10)
- `--lr`: 学习率 (默认: 0.01)
- `--momentum`: SGD 动量 (默认: 0.5)

例如：

```bash
python -m pytorch.main --batch-size 128 --epochs 20 --lr 0.005
```

## 特性

- 自动检测并使用 Apple Metal (MPS) 加速
- 详细的训练和评估日志
- 训练时间统计
- 模型保存功能

## 输出

训练完成后，模型将被保存为 `mnist_cnn.pt`，并输出最终准确率和总训练时间。 