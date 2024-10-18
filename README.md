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
source
├── Code_Aligned_Autoencoders.py - contains an implementation of the method from [1] used for comparison.
├── I2Inet.py - contains one of the sub-networks used for comparison. The translation is only done to Y's domain. Network usd for ablation purposes.
├── I2Inet2.py - contains one of the sub-networks used for comparison. The translation is only done to X's domain. Network usd for ablation purposes.
├── SCCN.py - contains an implementation of the method from [2] used for comparison.
├── XNet.py - contains an implementation of the purposed method.
├── change_detector.py, change_priors.py, datasets.py, decorators.py, filtering.py, image_translation.py, matrics.py contain useful classes and functions.
├── config.py contains dictionaries with the configuration settings.
└── experiments.py - runs the experiments used for the paper. Runs all different networks, with the different dimensionality reduction techniques and different number of bands. All expeiments are run N=5 times. 
```

[1] X. Niu, M. Gong, T. Zhan, and Y. Yang, “A conditional adversarial network for change detection in heterogeneous images,” IEEE Geosci. Remote Sens. Lett., vol. 16, no. 1, pp. 45–49, 2018.
[2] J. Liu, M. Gong, K. Qin, and P. Zhang, “A deep convolutional coupling network for change detection based on heterogeneous optical and radar images,” IEEE Trans. Neural Netw. Learn. Syst., vol. 29, no. 3, pp. 545–559, 2016.


