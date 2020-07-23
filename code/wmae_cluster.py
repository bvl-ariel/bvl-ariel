import os
import sys
from predictive_models import *
try:
    from tqdm import tqdm, trange
except ImportError:
    print("tqdm ...")
from utils import load_data
import random
from data_evaluation import multi_score
import itertools
import matplotlib.pyplot as plt
from typing import List
from cluster_targets import create_all


XRSL = """&
(executable="pozeni.sh")
(jobname="{0}")
(stdout = "mojLog.log")
(join = yes)
(walltime = "3 days")
(gmlog = "log")
(memory = "8000")
(count="1")
(countpernode="1")
(inputfiles = ("data.tar.gz" "/media/sf_Matej/ariel/database/csv/csv.tar.gz")
              ("py.tar.gz" "/media/sf_Matej/ariel/code/py.tar.gz")
              ("clus.jar" "/media/sf_Matej/ariel/code/clus.jar"))
(outputfiles = ("results.tar.gz" ""))
(runTimeEnvironment = "APPS/BASE/PYTHON-E8")
(queue != "gridgpu")"""

SH = """tar -xzf py.tar.gz
tar -xzf data.tar.gz
python3 training_trees.py {} {} {}
tar -czf results.tar.gz predictions*
"""


def create_jos():
    for i in range(55):
        exp_dir_ch = "../experiments/wmae/channel{}".format(i)
        trees = 10
        max_trees = 250
        assert max_trees % trees == 0
        for j in range(1, max_trees, trees):
            exp_dir = exp_dir_ch + "/tree{}".format(j)
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, "krog1.xrsl"), "w", newline="") as f:
                print(XRSL.format("ch{}".format(i)), file=f)
            with open(os.path.join(exp_dir, "pozeni.sh"), "w", newline="") as f:
                print(SH.format(i, j, j + trees - 1), file=f)


create_jos()