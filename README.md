### Implementation of [Domain Adaptation for the Segmentation of Confidential Medical Images](https://arxiv.org/abs/2101.00522)

# Requirements

Python 3.8; Tensorflow 2.2.0; CUDA 460 driver

# Data

The MMWHS data can be downloaded from the PnP-AdaNet repository [here](https://github.com/carrenD/Medical-Cross-Modality-Domain-Adaptation). The CHAOS data can be downloaded directly from the challenge [website](https://chaos.grand-challenge.org/). Alternatively, processed data is also available at this [link](https://drive.google.com/file/d/1vvukECVjFJ93HyaEcERbuNdrG7pXvKPk/view?usp=sharing). Data should be placed in a **data** folder in the repository root. 

# Running the code

If data pre-processing is needed, please refer to the **data-preprocessing-\*** notebooks. Once data is processed, to run the code simply run **training** and then **adaptation** .ipynb notebooks for either problem. 
