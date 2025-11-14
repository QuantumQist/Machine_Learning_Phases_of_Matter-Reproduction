# ml_phases_of_matter_reproduction

This repository reproduces the results corresponding to the **square-lattice Ising model** from the paper:

> **Carrasquilla, J., & Melko, R. G. (2017).**  
> *Machine learning phases of matter.* *Nature Physics*, 13(5), 431–434.  
> [https://doi.org/10.1038/nphys4035](https://doi.org/10.1038/nphys4035)

---

## 1. Training Data

The Monte Carlo training data used in this reproduction can be obtained from Juan Carrasquilla’s official data repository:  
👉 [https://github.com/carrasqu/data_nature_phy_paper](https://github.com/carrasqu/data_nature_phy_paper)

> *Note:* This repository is not directly referenced in the paper.  
> I located it independently to ensure the use of the original datasets.

Before running the notebook, all datasets should be downloaded from that repository and extract them into a local folder named `data/`.  
The expected directory structure is: `data/L_20/Xtrain.txt`.

The data is not provided in this notebook as I do not own it. 

## 2. Files. 

### `models` folder

This folder contains pretrained models for different lattice sizes.

### `training_notebook.ipynb`

The notebook where we perform training of neural networks.

### `model_evaluation.ipynb`

This notebook contains the reproduction of figures from the original paper.

### `engine.py`

This file implements the method training the neural networks.

### `helper_functions.py`

This file contains methods helping with evaluation, e.g., providing model accuracy or making plots. 

### `utils.py`

This notebook implements the function loading the data used for training and testing. 

---

## Comments

This notebook represents my **best effort to faithfully reproduce** the results of Carrasquilla & Melko (2017) using **PyTorch** rather than TensorFlow used in the paper.  
The central qualitative result — **phase classification across the critical temperature** — is successfully reproduced.  

---

**Author:** Robert Czupryniak

**License:** MIT

**Acknowledgment:** This reproduction uses data made publicly available by Juan Carrasquilla.
