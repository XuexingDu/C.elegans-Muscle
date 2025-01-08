# Hodgkin-Huxley Based Model for C.elegans Body Wall Muscle Cell 

Code for paper [**Hodgkin-Huxley Based Model for C.elegans Body Wall Muscle Cell **](https://www.biorxiv.org/content/10.1101/2024.07.15.603498v1).

![paper-summary](PCB-news.png)

## Requirements

### Python

- neuronal simulation: numpy, brian2
- network reconstruction: numpy, pandas matplotlib, scipy, sci-kit learn, seaborn, 


### Install Python utilities

```bash
conda create -n causal4 python=3.11 --file requirements.txt -c conda-forge
conda activate causal4
pip install -e .
```

Install additional packages for running body-wall muscle cell model:
```bash
pip install --upgrade "jax[cpu]"
pip install -U brainpy
pip install brainpylib
pip install -U "ray[default]"

## Reproduce figures in the maintext

1. Figure 2:
    ```bash
    ./code4paper/pm_scan_kl_HH10.py
    ```
2. Figure 3:
    ```bash
    ./code4paper/pm_scan_kl_HH10.py
    ```
3. Figure 4:
    ```bash
    ./code4paper/HH100_recon_pnas.py
    ```