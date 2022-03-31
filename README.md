* Contents
    * [Introduction](#introduction)
	* [Requirements](#requirements)
	* [Usage](#usage)
    * [Citations](#citations)
	* [License](#license)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Introduction
Just for fun.

## Requirements
* python = 3.6 
* torchvision = 0.6.1
* tabulate = 0.8.7
* overrides = 3.0.0
* opencv_python = 4.3.0.36
* numpy = 1.16.4
* pandas = 1.0.5
* allennlp = 1.0.0
* torchtext = 0.6.0
* tqdm = 4.47.0
* torch = 1.5.1shapely
* scikit-image
* imgaug==0.4.0
* pyclipper
* lmdb
* tqdm
* numpy
* visualdl
* python-Levenshtein
* opencv-contrib-python==4.4.0.46
* cython
* lxml
* premailer
* openpyxl
* paddle
```bash
pip install -r requirements.txt
```
Are you having fun with installing the required environment variables? Here's a good news: besides the above stuffs, you also have to install paddle-gpu (by visiting [this link](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html))

## Usage
In order to run smoothly, you have to:
* Configurating your weight and dataset folders in ``config.yml`` file. (for each module please read the README file in that folder)
* Run the ``predictor.py`` file

## Citations
If you find this code useful please cite the authors, not us (we just reimplement) 

* [paper](https://arxiv.org/abs/2004.07464):
```bibtex
@inproceedings{Yu2020PICKPK,
  title={{PICK}: Processing Key Information Extraction from Documents using 
  Improved Graph Learning-Convolutional Networks},
  author={Wenwen Yu and Ning Lu and Xianbiao Qi and Ping Gong and Rong Xiao},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  year={2020}
}
```

* **PaddleOCR**
- Many thanks to [Khanh Tran](https://github.com/xxxpsyduck) and [Karl Horky](https://github.com/karlhorky) for contributing and revising the English documentation.
- Many thanks to [zhangxin](https://github.com/ZhangXinNan) for contributing the new visualize function、add .gitignore and discard set PYTHONPATH manually.
- Many thanks to [lyl120117](https://github.com/lyl120117) for contributing the code for printing the network structure.
- Thanks [xiangyubo](https://github.com/xiangyubo) for contributing the handwritten Chinese OCR datasets.
- Thanks [authorfu](https://github.com/authorfu) for contributing Android demo  and [xiadeye](https://github.com/xiadeye) contributing iOS demo, respectively.
- Thanks [BeyondYourself](https://github.com/BeyondYourself) for contributing many great suggestions and simplifying part of the code style.
- Thanks [tangmq](https://gitee.com/tangmq) for contributing Dockerized deployment services to PaddleOCR and supporting the rapid release of callable Restful API services.
- Thanks [lijinhan](https://github.com/lijinhan) for contributing a new way, i.e., java SpringBoot, to achieve the request for the Hubserving deployment.
- Thanks [Mejans](https://github.com/Mejans) for contributing the Occitan corpus and character set.
- Thanks [LKKlein](https://github.com/LKKlein) for contributing a new deploying package with the Golang program language.
- Thanks [Evezerest](https://github.com/Evezerest), [ninetailskim](https://github.com/ninetailskim), [edencfc](https://github.com/edencfc), [BeyondYourself](https://github.com/BeyondYourself) and [1084667371](https://github.com/1084667371) for contributing a new data annotation tool, i.e., PPOCRLabel。

## License
This project is licensed under the MIT License. See LICENSE for more details.

## Acknowledgements
This project structure takes example by [PyTorch Template Project](https://github.com/victoresque/pytorch-template).
