## Set up your coding playground (tools)

Use Python + Jupyter Notebook (easiest for experiments)

Ensure Python is install.

### Install Jupyter:

```bash
pip install notebook
# or to install the newer JupyterLab interface:
pip install jupyterlab
```

### Launch Jupyter:

```bash
jupyter notebook
# or
jupyter lab
```

### Install these (run in terminal / command prompt):

```bash
pip install torch torchvision torchaudio  # GPU version if you have NVIDIA
pip install segmentation-models-pytorch   # magic ready models
pip install opencv-python pillow matplotlib pandas tqdm
```

> **Note:** If you use Google Colab → most are already there, just add `!pip install segmentation-models-pytorch`

### Create a new notebook

Create a new notebook called `dots_to_full_segmentation.ipynb`