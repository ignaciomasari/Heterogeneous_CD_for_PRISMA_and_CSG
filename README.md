# Heterogeneous_CD for PRISMA and Cosmo-SkyMed data
This repository contains the software developed for Heterogeneous Change Detection from image domains with a great difference in dimensionality. This work was partially supported by ASI in the framework of the project PRISMA_Learn – ASI no. 2022-12-U.0; the support is gratefully acknowledged.

## Data
Three datasets were mainly used for this project.
The images used were COSMO-SkyMed Second Generation and PRISMA Products, both © of the Italian Space Agency (ASI), delivered under a license to use.

## License

The code is released under the GPL-3.0-only license. See `LICENSE` for more details.

## Installation

The code was built on a virtual environment running on Python 3.10

### Clone the repository

```
git clone --recursive https://github.com/ignaciomasari/Heterogeneous_CD_for_PRISMA_and_CSG.git
```

## Project structure

```
semantic_segmentation
├── dataset - contains the data loader
├── input - images to train and test the network 
├── net - contains the loss, the network, and the training and testing functions
├── CFC-CRF - contains the approximation of the ideal fully connected CRF (refer to the README in that folder for more informations)
├── output - should contain the results of the training / inference
|   ├── exp_name
|   └── model.pth
├── utils - misc functions
└── main.py - program to run
```



