import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset


# =========================
# 配置
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

testsize = 352
model_path = os.path.join('snapshots', 'PraNet_Windows', 'PraNet_epoch_100.pth')

test_datasets = {
    'ETIS-LaribPolypDB': {
        'image_root': os.path.join('data', 'TestDataset', 'ETIS-LaribPolypDB', 'images') + os.sep,
        'gt_root': os.path.join('data', 'TestDataset', 'ETIS-LaribPolypDB', 'masks') + os.sep
    }
}

save_root = 'results'
os.makedirs(save_root, exist_ok=True)


# =========================
# 加载模型
# =========================
print('Loading PraNet...')
model = PraNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print('Model loaded.')


# =========================
# 测试
# =========================
with torch.no_grad():
    for dataset_name, paths in test_datasets.items():
        print(f'Testing on {dataset_name}...')

        save_path = os.path.join(save_root, dataset_name)
        os.makedirs(save_path, exist_ok=True)

        test_loader = test_dataset(
            paths['image_root'],
            paths['gt_root'],
            testsize
        )

        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            image = image.to(device)

            # PraNet forward（只取最终预测）
            res5, res4, res3, res2 = model(image)
            res = res2

            # resize 回原图大小
            res = F.interpolate(
                res,
                size=gt.shape,
                mode='bilinear',
                align_corners=False
            )
            res = res.sigmoid().cpu().numpy().squeeze()

            # 归一化并保存
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255).astype(np.uint8)

            Image.fromarray(res).save(os.path.join(save_path, name))

        print(f'{dataset_name} done.')

print('All tests finished.')
