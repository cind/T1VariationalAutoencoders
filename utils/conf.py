import json
import logging
import os
import smtplib

LOGGER = logging.getLogger(__name__)

config = {}

def load(config_file):
    LOGGER.info("Loading configuration from " + config_file)
    with open(config_file, 'r') as cf:
        global config
        config = json.loads(cf.read())

local = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pijp.conf"))

if os.path.exists(local):
    load(local)
elif os.path.exists('/etc/pijp.conf'):
    load('/etc/pijp.conf')
else:
    LOGGER.error("No configuration file found.")

