import argparse
from mnist import train_mnist


def main():
    """
    主函数，解析命令行参数并调用MNIST训练函数
    """
    parser = argparse.ArgumentParser(description="PyTorch MNIST 训练")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="输入批次大小 (默认: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, metavar="N", help="训练轮数 (默认: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="学习率 (默认: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD 动量 (默认: 0.5)"
    )
    args = parser.parse_args()

    print("开始 MNIST 手写数字识别训练...")

    # 调用训练函数
    results = train_mnist(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
    )

    print(f"训练完成！最终准确率: {results['accuracies'][-1]:.2f}%")
    print(f"总训练时间: {results['total_time']:.2f} 秒")


if __name__ == "__main__":
    main()
