import os
import numpy as np
from PIL import Image


# =========================
# Metric functions
# =========================

def dice_score(pred, gt, eps=1e-7):
    intersection = (pred * gt).sum()
    return (2.0 * intersection + eps) / (pred.sum() + gt.sum() + eps)


def iou_score(pred, gt, eps=1e-7):
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return (intersection + eps) / (union + eps)


def mae_score(pred, gt):
    return np.mean(np.abs(pred - gt))


# =========================
# Evaluation
# =========================

def evaluate_dataset(pred_dir, gt_dir, threshold=0.5):
    dice_list = []
    iou_list = []
    mae_list = []

    names = os.listdir(pred_dir)
    names.sort()

    for name in names:
        pred_path = os.path.join(pred_dir, name)
        gt_path = os.path.join(gt_dir, name)

        if not os.path.exists(gt_path):
            continue

        pred = Image.open(pred_path).convert('L')
        gt = Image.open(gt_path).convert('L')

        pred = np.array(pred, dtype=np.float32) / 255.0
        gt = np.array(gt, dtype=np.float32) / 255.0

        # MAE (continuous)
        mae_list.append(mae_score(pred, gt))

        # Dice / IoU (binary)
        pred_bin = (pred >= threshold).astype(np.uint8)
        gt_bin = (gt >= 0.5).astype(np.uint8)

        dice_list.append(dice_score(pred_bin, gt_bin))
        iou_list.append(iou_score(pred_bin, gt_bin))

    return np.mean(dice_list), np.mean(iou_list), np.mean(mae_list), len(dice_list)


# =========================
# Main
# =========================

if __name__ == '__main__':

    pred_dir = 'results/CVC-300/'
    gt_dir = 'data/TestDataset/CVC-300/masks/'

    dice, iou, mae, num = evaluate_dataset(pred_dir, gt_dir)

    print('======== Evaluation Result:CVC-300=================')
    print(f'Test images : {num}')
    print(f'Dice        : {dice:.4f}')
    print(f'IoU         : {iou:.4f}')
    print(f'MAE         : {mae:.4f}')
    print('===================================================')
