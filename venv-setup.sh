#!/bin/bash

#alias python=/opt/python/3.10.5/bin/python3.10
#python -m venv venv
/opt/python/3.10.5/bin/python3.10 -m venv venv
source venv/bin/activate
#unalias python

pip install -U pip
pip install -r requirements.txt
