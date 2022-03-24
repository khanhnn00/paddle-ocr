## Introduction

PaddleOCR aims to create multilingual, awesome, leading, and practical OCR tools that help users train better models and apply them into practice. For more details, please visit the original repository at [this link](https://github.com/PaddlePaddle/PaddleOCR).

## Quick start
- Clone this repository to your local machine
- Run the command ``pip install -r requirement.txt`` in order to install needed environment variables.
- Visit [this link](https://drive.google.com/drive/folders/1e7ug1WLVtxqj9f6kFCDLw1Bvwvo_KDzS?usp=sharing) and download the weights of two models, including recognition and detection.
- You are asking me where should I put these weights? Good question, now you can put it anywhere you want, just remember to specify the path in your config file, in ``Global.pretrained_model`` section.
- When running, **MAKE SURE THAT YOUR POINTER IS IN THE PARENT FOLDER**. You can run by the command ``bash ./bash/your-file.sh``. I have created many .sh files for many running cases: run det only (``infer_det.sh``), run recognition only (``infer_rec.sh``) and run det + rec (``infer_full.sh``). You can also specify the weight folder and the image to infer inside the .sh file.
- Result image should appear in the parent folder, named ``result.jpg``.

## License
This project is released under <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>

## Contribution
We welcome all the contributions to PaddleOCR and appreciate for your feedback very much.

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
