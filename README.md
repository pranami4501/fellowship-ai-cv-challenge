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
- *(Model weights are generated by running the notebook end-to-end.)*  

## 🚀 How to Run

1. Click the **“Open in Colab”** badge above.  
2. In Colab, set **Runtime → Change runtime type → GPU**.  
3. Run cells top to bottom.  
4.  Run all cells; the notebook will train the model and save `resnet50_flowers102.pth` in your Colab workspace.
