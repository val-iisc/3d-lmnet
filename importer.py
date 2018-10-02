import argparse
import json
import os
import random
import re
import sys
import time

from itertools import product
from os import listdir, makedirs
from os.path import join, exists, isdir, dirname, abspath, basename

import cv2
import numpy as np
import tensorflow as tf
import tflearn

from utils.data_loader import *
from utils.checkpoint_loader import *
from utils.metrics_utils import *
from utils.summary_utils import *
from utils.encoders_decoders import *

random.seed(1024)
np.random.seed(1024)
tf.set_random_seed(1024)
