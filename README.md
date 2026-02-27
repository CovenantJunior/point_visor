# PointVisor: Point-Supervised Semantic Segmentation
### Project: LandVisor Technical Assessment -- DLRSD Analysis

PointVisor is a deep learning solution for **Weakly-Supervised Semantic Segmentation**. It solves the "challenging problem" of remote sensing classification using sparse point annotations instead of dense pixel masks.

## 🛠️ Environment Setup

### 1. Install Jupyter
```bash
pip install notebook jupyterlab

```

### 2\. Install Core Dependencies

Bash

```
# Core AI & Models
pip install torch torchvision torchaudio
pip install segmentation-models-pytorch

# Image Processing & Style Stability
pip install opencv-python scikit-image pillow

# Data Management
pip install pandas matplotlib tqdm

```

* * * * *

🔬 Methodology
--------------

### 1\. Style Stability (Histogram Matching)

To ensure the **"stability of the picture style"** across different remote sensing tiles, the pipeline matches the color distribution of every image to a reference tile. This prevents atmospheric or sensor-driven style shifts from confusing the model.

### 2\. Partial Focal Cross-Entropy (pfCE)

The model trains using a custom **Partial Focal Loss**. Unlike standard CE, it applies a binary mask to the gradient, forcing the model to learn *only* from verified points while ignoring unmarked pixels.

$$pfCE = \frac{\sum (Focal\_loss(pre, GT) \times MASK_{labeled})}{\sum MASK_{labeled}}$$

* * * * *

📂 Dataset & Simulation
-----------------------

-   **Source:** DLRSD (2,100 images, 256x256, 17 classes).

-   **Simulation:** The system simulates "incomplete tagging" by randomly sampling $N$ points (5, 15, or 30) per land-cover class.

* * * * *

🏃 Execution Instructions
-------------------------

1.  **Data:** Place `DLRSD/Images` and `DLRSD/Labels` in the project root.

2.  **Run:** Open `dots_to_full_segmentation.ipynb`.

3.  **Experimental Battery:**

    -   The script re-initializes the **ResNet34-UNet** for every run.

    -   It compares Point Density vs. Loss Function performance.

    -   **Note:** `num_workers=0` is set for Windows compatibility to prevent hangs.

* * * * *

📝 Technical Report
-------------------

### Purpose & Hypothesis

**Hypothesis:** Partial Focal Loss will show superior convergence stability over standard Cross-Entropy because it effectively weights hard-to-classify sparse points against the background "noise" of unlabeled pixels.

### Results Comparison (Example Data)

| **Pts/Class** | **Loss Type** | **Final Loss** | **Performance** |
| --- | --- | --- | --- |
| 5 | Focal (pfCE) | 0.0124 | Stable |
| 5 | Vanilla CE | 0.0451 | High Error |
| 30 | Focal (pfCE) | 0.0082 | Optimized |

### Conclusion

The **pfCE** approach allows LandVisor to leverage minimal human annotation effort while maintaining high segmentation accuracy, fulfilling the "Weakly Supervised" requirement of the project.

* * * * *