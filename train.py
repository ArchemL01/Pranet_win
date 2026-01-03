import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.PraNet_Res2Net import PraNet
from utils.dataloader import PolypDataset
from utils.utils import AvgMeter


def main():
    # =========================
    # 1. 基本配置
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据路径
    image_root = os.path.join(
        "data", "TrainDataset", "Kvasir-SEG", "images"
    )
    gt_root = os.path.join(
        "data", "TrainDataset", "Kvasir-SEG", "masks"
    )

    # 预训练权重（Res2Net）
    backbone_weight = os.path.join(
        "pretrained", "res2net50_v1b_26w_4s-3cf99910.pth"
    )

    # 模型保存路径
    save_dir = os.path.join("snapshots", "PraNet_Windows")
    os.makedirs(save_dir, exist_ok=True)

    # 训练参数
    batch_size = 2
    lr = 1e-4
    epochs = 100
    trainsize = 352

    # =========================
    # 2. 数据集 & DataLoader
    # =========================
    train_dataset = PolypDataset(
        image_root=image_root,
        gt_root=gt_root,
        trainsize=trainsize
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # =========================
    # 3. 模型构建
    # =========================
    model = PraNet().to(device)

    # —— 只加载 Res2Net backbone 权重 ——
    print("Loading Res2Net pretrained weights...")
    state_dict = torch.load(backbone_weight, map_location="cpu")
    model.resnet.load_state_dict(state_dict, strict=False)
    print("Backbone loaded.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # =========================
    # 4. 训练循环
    # =========================
    model.train()
    for epoch in range(1, epochs + 1):
        loss_record = AvgMeter()

        for i, pack in enumerate(train_loader):
            images, gts = pack
            images = images.to(device)
            gts = gts.to(device)

            optimizer.zero_grad()

            # PraNet 多尺度输出
            pred1, pred2, pred3, pred4 = model(images)

            loss = (
                criterion(pred1, gts) +
                criterion(pred2, gts) +
                criterion(pred3, gts) +
                criterion(pred4, gts)
            )

            loss.backward()
            optimizer.step()

            loss_record.update(loss.item(), images.size(0))

        print(
            f"Epoch [{epoch:03d}/{epochs}] | "
            f"Loss: {loss_record.show():.4f}"
        )

        # 每 10 个 epoch 保存一次
        if epoch % 10 == 0:
            save_path = os.path.join(
                save_dir, f"PraNet_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
