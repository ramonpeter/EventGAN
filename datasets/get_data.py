""" Get datasets """

import os
import wget

# Make ttbar DIR
PARENT_DIR = "../datasets"
DIR = "../datasets/ttbar"

PATH = os.path.abspath(os.path.join(PARENT_DIR, DIR))

if not os.path.exists(PATH):
    os.makedirs(PATH)

URLS = [
    "https://www.dropbox.com/s/b5g80zt3lkbj8tn/ttbar_6f_train.h5?dl=1",
    "https://www.dropbox.com/s/wxb9vdt2ueidb9l/ttbar_6f_test.h5?dl=1",
    "https://www.dropbox.com/s/8m5tm67tw0a4815/README.txt?dl=1",
]

NAMES = ["ttbar_6f_train.h5", "ttbar_6f_test.h5", "README.txt"]

for url, name in zip(URLS, NAMES):
    wget.download(url, f"ttbar/{name}")
