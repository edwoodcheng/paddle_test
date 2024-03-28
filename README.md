# This repo is runing on CPU, **NOT GPU**. Please install the corresponding library for GPU

OS: Windows, Python version: 3.11.5

**1.1 Install PaddlePaddle**

If you do not have a Python environment, please refer to Environment Preparation.


If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install

	python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple


If you have no available GPU on your machine, please run the following command to install the CPU version

	python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple


1.2 Install PaddleOCR Whl Package

    # Install paddleocr, version 2.6 is recommended
	pip3 install "paddleocr>=2.6.0.3"

    # Install the image direction classification dependency package paddleclas (if you do not use the image direction classification, you can skip it)
	pip3 install paddleclas>=2.4.3
