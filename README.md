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

---

## Comments and Limitations

This notebook represents my **best effort to faithfully reproduce** the results of Carrasquilla & Melko (2017) using **PyTorch** rather than TensorFlow.  
The central qualitative result — **phase classification across the critical temperature** — is successfully reproduced.  
However, several implementation details remain ambiguous, as the paper does not specify them explicitly.  

### Known ambiguities
- The exact **L2 regularization coefficient** (weight-decay strength) is not reported.  
- The **number of training epochs** is not stated; here, training continues until convergence is visually observed.  
- The **label format** is unspecified. This implementation assumes integer binary labels (`0` or `1`), not one-hot vectors.  
- The **spin representation** in the dataset is not discussed (`±1` vs. `0/1`).  
  Using Carrasquilla’s released data resolved this uncertainty.

These minor ambiguities do not affect the qualitative conclusions, but they should be noted for anyone attempting exact numerical reproduction.

---

**Author:** Robert Czupryniak

**License:** MIT

**Acknowledgment:** This reproduction uses data made publicly available by Juan Carrasquilla.
