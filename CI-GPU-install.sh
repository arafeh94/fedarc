#!/bin/bash
sudo apt-get update -y
sudo apt install python3-pip
sudo apt install libopenmpi-dev
sudo apt-get install python3-venv
python3 -m venv venv
source venv/bin/activate
pip3 install wheel
pip3 install mpi4py
pip3 install scikit-learn
pip3 install numpy
pip3 install h5py
pip3 install setproctitle
pip3 install networkx
pip3 install psutil
pip3 install paho-mqtt
pip3 install mysql-connector-python
pip3 install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#default cuda on ubuntu 20.04 is 10.1, you can change it
