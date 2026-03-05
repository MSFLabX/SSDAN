# S<sup>2</sup>-Differential Feature Awareness Network for Hyperspectral Image Fusion(SSDAN)
![Language](https://img.shields.io/badge/language-python-brightgreen) 

Our model are trained on a NVIDIA A800-SXM4-80GB GPU.

<div align="center">
    <img src="SSDAN.png" alt="overall framework" width="800"/>
</div>

## 👉 Data
We conduct experiments on three publicly available datasets and use a data simulation strategy to generate training and test image pairs.
* [Cave](https://cave.cs.columbia.edu/repository/Multispectral)

* [Harvard](http://vision.seas.harvard.edu/hyperspec/)

* [KAIST](https://vclab.kaist.ac.kr/siggraphasia2017p1/)

## 🌈 Results

| Dataset  | PSNR | SAM | RMSE |
|----------|--------|--------|-----------|
| Cave    | 49.0519 |  2.2188 |    1.1347  |
| Harvard   | 47.4242 |  2.8470 |    1.6968  |
| KAIST  | 45.8791 |  2.4655 |    1.3743  |

## 🌿 Getting Started

### Environment Setup

To get started, we recommend setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment.
    
    conda create -n SSDAN python==3.8.19
    
    conda activate SSDAN
    
    pip install -r requirements.txt

### Train and Test
    python train_SSDAN.py

### Citation
If this code is useful for your research, please cite this paper.
    
    Update later

