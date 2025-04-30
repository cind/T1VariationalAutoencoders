#!/bin/bash

/opt/python/3.10.5/bin/python3.10 -m venv venv-tf
source venv-tf/bin/activate

pip install -U pip
pip install -r reqs_tf.txt
