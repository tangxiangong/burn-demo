import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore
import time
from typing import Tuple, Dict, Any

from .model import MNISTConvNet


def get_device() -> torch.device:
    """
    获取可用的设备，优先使用Apple Metal（MPS）

    Returns:
        torch.device: 训练设备
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 Apple Metal (MPS) 加速")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用 CUDA 加速")
    else:
        device = torch.device("cpu")
        print("使用 CPU 进行训练")
    return device


def load_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """
    加载MNIST数据集

    Args:
        batch_size: 批次大小

    Returns:
        Tuple[DataLoader, DataLoader]: 训练数据加载器和测试数据加载器
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """
    训练模型一个epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 训练设备
        epoch: 当前epoch

    Returns:
        float: 平均损失
    """
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"训练 Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                f" ({100.0 * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}"
            )

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    评估模型

    Args:
        model: 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 训练设备

    Returns:
        Tuple[float, float]: 测试损失和准确率
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        f"\n测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)}"
        f" ({accuracy:.1f}%)\n"
    )

    return test_loss, accuracy


def train_mnist(
    epochs: int = 10, batch_size: int = 64, lr: float = 0.01, momentum: float = 0.5
) -> Dict[str, Any]:
    """
    训练MNIST模型的主函数

    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        momentum: 动量

    Returns:
        Dict[str, Any]: 包含训练结果的字典
    """
    # 获取设备
    device = get_device()

    # 加载数据
    train_loader, test_loader = load_data(batch_size)

    # 创建模型
    model = MNISTConvNet().to(device)
    print(f"模型结构:\n{model}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # 训练和评估
    train_losses = []
    test_losses = []
    accuracies = []

    # 记录总训练时间
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, device, epoch)
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)

        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch} 完成，用时: {epoch_time:.2f} 秒")

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

    total_time = time.time() - start_time
    print(f"总训练时间: {total_time:.2f} 秒")

    # 保存模型
    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("模型已保存为 mnist_cnn.pt")

    return {
        "model": model,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "accuracies": accuracies,
        "total_time": total_time,
    }
