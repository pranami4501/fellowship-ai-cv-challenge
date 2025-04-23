# Fellowship.AI CV Challenge

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pranami4501/fellowship-ai-cv-challenge/blob/main/FellowshipProjectCV.ipynb)

This notebook fine-tunes a pretrained **ResNet-50** on the **Oxford Flowers-102** dataset using PyTorch. It covers:

1. **Setup & Dependencies**  
2. **Dataset Preparation** (download, split, folder organization)  
3. **Data Pipeline** (transforms, DataLoaders)  
4. **Model Setup** (freeze backbone, new 102-way head)  
5. **Training & Validation** (loss, accuracy tracking)  
6. **Results & Analysis** (test accuracy, confusion matrix, error examples)  
7. **Conclusion & Next Steps**

## 📦 Repository Contents

- `FellowshipProjectCV.ipynb` – Colab notebook with full code and narrative  
- `resnet50_flowers102.pth` – Trained model weights

## 🚀 How to Run

1. Click the **“Open in Colab”** badge above.  
2. In Colab, set **Runtime → Change runtime type → GPU**.  
3. Run cells top to bottom.  
4. Download `resnet50_flowers102.pth` and reload the model:
   ```python
   from torchvision import models
   from torch import nn
   import torch

   model = models.resnet50(pretrained=False)
   model.fc = nn.Linear(model.fc.in_features, 102)
   model.load_state_dict(torch.load('resnet50_flowers102.pth'))
   model.eval()
