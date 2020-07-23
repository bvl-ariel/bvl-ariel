import random
from utils import convert_stacked_to_mtr
# from cluster_search import create_all
from cluster_targets import create_all_clusters
from tqdm import tqdm
import sys
from predictive_models import SemiMTR
from data_evaluation import multi_score
import os
import numpy as np

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
              ("grid_cluster_rf.txt" "/media/sf_Matej/ariel/optimisation/grid_cluster_rf.txt")
              ("py.tar.gz" "/media/sf_Matej/ariel/code/py.tar.gz"))
(outputfiles = ("results.tar.gz" ""))
(runTimeEnvironment = "APPS/BASE/PYTHON-E8")
(queue != "gridgpu")"""


def create(subdir, xrsl, feature_type):
    for c in range(55):
        experiment_dir_full = "../experiments/cross_val/{}/channel{}/".format(subdir, c)
        os.makedirs(experiment_dir_full, exist_ok=True)
        # sh
        with open(os.path.join(experiment_dir_full, "pozeni.sh"), "w", newline="") as f:
            print("tar -xzf py.tar.gz", file=f)
            print("tar -xzf data.tar.gz", file=f)
            print("python3 training_trees.py {} {}".format(c, feature_type), file=f)
            print("tar -czf results.tar.gz RandomForest* true_indices*", file=f)
        # xrsl
        with open(os.path.join(experiment_dir_full, xrsl), "w", newline="") as f:
            print(XRSL.format("bagung{}".format(c)), file=f)


def collect_the_results(cluster_file):
    experiments = "../experiments/semi_mtr/smarter"
    collected_results = {}
    keys = ["mse", "mae", "wmae"]
    for i, experiment in enumerate(os.listdir(experiments)):
        if not experiment.startswith("clusters"):
            continue
        conf = os.path.join(experiments, experiment, "conf.txt")
        results = os.path.join(experiments, experiment, "results/results.txt")
        if os.path.exists(results):
            with open(conf) as f:
                clustering = eval(f.readline())
            with open(results) as f:
                for l in f:
                    if l.startswith("PARAMETERS"):
                        clustering2 = eval(l[l.find(";") + 1:])
                        assert clustering == clustering2, results
                        break
                metrics = eval(f.readline())
            assert len(metrics) == len(clustering)
            for performances, cluster in zip(metrics, clustering):
                for target in performances:
                    for key in keys:
                        if key not in collected_results:
                            collected_results[key] = [[] for _ in range(55)]
                        # old_value = collected_results[key][target][0]
                        current_value = performances[target][key]
                        collected_results[key][target].append((current_value, cluster))
        else:
            print(results, "missing")
    best = {key: [None for _ in range(55)] for key in keys}
    cs = [[0, 1], [0, 1], [2, 4], [3, 4], [2, 4], [5, 7], [6, 12], [7, 14], [8, 9], [8, 9], [10, 46], [11, 12], [6, 12],
          [12, 13], [14, 44], [15, 16], [16, 17], [17, 19], [18, 19], [17, 19], [20, 41], [21, 51], [22, 25], [23, 50],
          [23, 24], [25, 27], [26, 49], [25, 27], [27, 28], [28, 29], [30, 48], [29, 31], [32, 38], [33, 35], [34, 37],
          [33, 35], [36, 54], [37, 38], [37, 38], [38, 39], [31, 40], [20, 41], [15, 42], [14, 43], [14, 44], [45, 46],
          [45, 46], [9, 47], [24, 48], [26, 49], [50, 51], [50, 51], [32, 52], [53, 54], [53, 54]]
    for key in collected_results:
        if key != "mse":
            continue
        print("measure", key)
        for target, pair in enumerate(collected_results[key]):
            pair.sort()
            i_mtr, i_str, i_opt = None, None, None
            for i in range(len(pair)):
                if pair[i][1] == [target]:
                    i_str = i
                elif pair[i][1] == list(range(55)):
                    i_mtr = i
                elif pair[i][1] == cs[target]:
                    i_opt = i
            if i_opt is not None:
                print(target, pair[0], i_str, pair[i_str][0], i_mtr, pair[i_mtr][0], i_opt, pair[i_opt][0])
            best[key][target] = pair[0][1]
    exit(0)
    with open(cluster_file, "w") as f:
        print(best, file=f)
        # for elt in collected_results[key][:100]:
        #     print(elt)
        # str_index, mtr_index = None, None
        # for i in range(len(collected_results[key])):
        #     c = collected_results[key][i][-1]
        #     if c == str:
        #         str_index = i
        #     elif c == mtr:
        #         mtr_index = i
        # print("Index of str and mtr:", str_index, mtr_index)
        # print()


if __name__ == "__main__":
    create("random_forest/classic", "cl1.xrsl", "classic")
    # perform_cluster_operation("../experiments/semi_mtr/smarter/clusters0/", "../database/csv/")
    # collect_the_results("../optimisation/str_mtr_clusters.txt")
