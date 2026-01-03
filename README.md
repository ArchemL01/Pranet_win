# PraNet 论文复现

## 一、项目简介

本项目复现了论文 **PraNet: Parallel Reverse Attention Network for Polyp Segmentation**（MICCAI 2020），在 **Windows** 环境下完成了模型训练、测试与评估，用于课程作业验收

PraNet 通过并行反向注意力机制（Parallel Reverse Attention），逐级细化预测结果，在医学息肉分割任务中具有良好性能。

---

## 二、实验环境

- **操作系统**：Windows 10  
- **Python**：3.8  
- **深度学习框架**：PyTorch  
- **GPU**：NVIDIA RTX 3060

主要依赖如下：

```bash
pip install torch torchvision numpy pillow opencv-python thop
```

---

## 三、数据集

实验使用以下公开息肉分割数据集：

- Kvasir-SEG  
- CVC-ClinicDB  
- CVC-ColonDB  
- CVC-300  
- ETIS-LaribPolypDB  

### 数据目录结构（本地）

```text
data/
├── TrainDataset/
│   └── Kvasir-SEG/
│       ├── images/
│       └── masks/
└── TestDataset/
    ├── Kvasir/
    ├── CVC-ClinicDB/
    ├── CVC-ColonDB/
    ├── CVC-300/
    └── ETIS-LaribPolypDB/
        ├── images/
        └── masks/
```

---

## 四、使用说明（Usage）

### 1. 克隆仓库
### 2. 下载预训练权重

下载 Res2Net 预训练权重()，并放置在如下路径：

```text
pretrained/
└── res2net50_v1b_26w_4s-3cf99910.pth
```

---

### 3. 模型训练

```bash
python train.py
```

训练过程中会自动加载 Res2Net 预训练权重，并按 epoch 保存模型：

```text
snapshots/PraNet_Windows/
├── PraNet_epoch_10.pth
├── ...
└── PraNet_epoch_100.pth
```

---

### 4. 模型测试（生成预测结果）

在 `test.py` 中指定模型权重路径，例如：

```python
model_path = 'snapshots/PraNet_Windows/PraNet_epoch_100.pth'
```

运行：

```bash
python test.py
```

预测结果将保存在：

```text
results/
└── DatasetName/
```

---

### 5. 批量评估（Dice / IoU / MAE）

直接运行评估脚本：

```bash
python eval.py
```

评估结果将直接输出在终端。

---

## 五、说明（关于模型权重与数据集）

由于模型权重文件（`.pth`）和数据集体积较大，本仓库 **仅包含完整的代码实现与实验配置**，并未上传训练好的模型参数和原始数据。

### 未包含内容

- 训练得到的模型权重文件（`.pth`）
- 原始训练 / 测试数据集 （https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view）（https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view）
- 预训练模型 （https://drive.google.com/file/d/1FjXh_YG1hLGPPM6j-c8UxHcIWtzGGau5/view）
  
## 六、训练与测试设置

- **Backbone**：Res2Net-50 v1b（ImageNet 预训练）
- **输入尺寸**：352 × 352  
- **优化器**：Adam（初始学习率 1e-4，step decay）  
- **训练轮数**：100 epochs  
- **损失函数**：Structure Loss（Weighted BCE + IoU）

---

## 七、实验结果

在多个公开测试集上的定量评估结果如下：

| Dataset           | Images | Dice ↑ | IoU ↑ | MAE ↓ |
|-------------------|--------|--------|-------|-------|
| Kvasir-SEG        | 100    | 0.8625 | 0.8057 | 0.0437 |
| CVC-ClinicDB      | 62     | 0.8749 | 0.8231 | 0.0139 |
| CVC-ColonDB       | 380    | 0.6323 | 0.5697 | 0.0412 |
| CVC-300           | 60     | 0.8678 | 0.7998 | 0.0089 |
| ETIS-LaribPolypDB | 196    | 0.4892 | 0.4391 | 0.0256 |

---

## 八、个人工作说明

本人在本次作业中独立完成了以下工作：

1. 在 Windows 系统下成功配置 PraNet 训练与测试环境  
2. 修复官方代码在新版本 PyTorch 下的兼容性问题  
3. 完成模型完整训练并保存多 epoch 权重  
4. 编写并完善测试与评估流程（Dice / IoU / MAE）  
5. 在多个公开数据集上进行实验评估并分析结果  

---

## 十、总结

本项目在 Windows 单卡 GPU 环境下，完整复现了 PraNet 的训练、测试与评估流程，实验结果合理可靠，初步达到原论文中结果。

---

## 十一、参考

- Deng-Ping Fan et al., *PraNet: Parallel Reverse Attention Network for Polyp Segmentation*  
- 官方代码：https://github.com/DengPingFan/PraNet

