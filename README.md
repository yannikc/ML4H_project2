# ML4H_project 2

Repository for project 2 (Explainability and Interpretability) the course "Machine Learning for Health Care" 
(Spring Semester 2025) at ETH Zurich.

**Team members:**  
1. Yannik Collenberg (19-915-354)
2. Melanie Rieff (20-949-574)
3. Yumi Kim (20-920-815)

## Installation

We provide two files for setting up the environment:
- `environment.yml` - for conda environment setup
- `requirements_conda.txt` - for pip-based installation

### Option 1: Using conda (recommended)

1. Create and activate the conda environment from the YAML file:
```bash
conda env create -f environment.yml
conda activate ML4H_project2
```

### Option 2: Using pip

1. Create a new conda environment with Python 3.10:
```bash
conda create -n ML4H_project2 python=3.10
conda activate ML4H_project2
```

2. Install PyTorch (adjust for your CUDA version if using GPU):
```bash
# For CPU only
conda install pytorch torchvision
```

3. Install the remaining requirements:
```bash
pip install -r requirements_conda.txt
```

Note: If you encounter issues with the `nam` package and sklearn, install scikit-learn first:
```bash
pip install scikit-learn
pip install nam --no-deps #nam depends on sklearn, which is deprecated
```


## References
[1] Lundberg, S. M., & Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874). In *Advances in Neural Information Processing Systems 30 (NIPS 2017)*.\
[2] Agarwal, R., Melnick, L., Frosst, N., Zhang, X., Lengerich, B., Caruana, R., & Hinton, G. (2021). [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/abs/2004.13912). In *Advances in Neural Information Processing Systems 34 (NeurIPS 2021)*.