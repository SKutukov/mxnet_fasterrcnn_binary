import logging
import sys

# set up logger
logging.basicConfig(filename="train.log")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
