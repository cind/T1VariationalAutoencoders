#!/bin/bash

/opt/python/3.11.6/bin/python3.11 -m venv venv-ae
source venv-ae/bin/activate

pip install -U pip
pip install -r requirements.txt
