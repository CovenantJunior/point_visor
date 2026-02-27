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

**Author:** [Your ID]

**Category:** Remote Sensing / Deep Learning

**Date:** February 2026

```

Would you like me to help you summarize the "Final Results" table based on the actual numbers you get from your current run?

```PointVisor: Quick Setup & Execution Guide
-----------------------------------------

This procedure ensures your environment is ready to handle the **DLRSD** dataset and the **Point-Supervised** training logic.

* * * * *

### 1\. Tooling & Environment

Install the core deep learning and image processing stack.

Bash

```
# Core AI & Models
pip install torch torchvision torchaudio
pip install segmentation-models-pytorch

# Image Processing & Style Stability
pip install opencv-python scikit-image pillow

# Data & Progress Tracking
pip install pandas matplotlib tqdm

```

### 2\. Dataset Architecture

Organize your project folder so the code can locate the 4,200 total files.

-   **Root Folder:** `DLRSD/`

-   **Subfolder 1:** `Images/` (Place `.tif` or `.jpg` remote sensing tiles here)

-   **Subfolder 2:** `Labels/` (Place `.png` ground truth masks here)

### 3\. Core Implementation Logic

In your `dots_to_full_segmentation.ipynb`, ensure these three "LandVisor" requirements are met:

-   **Label Normalization:** Map raw pixel values (like 149, 160) to a $0--16$ range inside `__getitem__` to avoid `IndexError`.

-   **Dynamic Point Simulation:** Use `np.unique()` in the `simulate_points` function to ensure points are sampled even from sparse high-value labels.

-   **Partial Focal Loss:** Initialize the `PartialFocalLoss` with `gamma=2.0` to force the model to focus on labeled dots rather than the background.

### 4\. Running the Experiment Battery

Execute the training loop for the following configurations to generate your final report:

1.  **Point Counts:** Test with 5, 15, and 30 points per class.

2.  **Loss Types:** Toggle `use_focal` between `True` and `False`.

3.  **Batching:** Use `batch_size=8` and `num_workers=0` (for Windows stability).

* * * * *