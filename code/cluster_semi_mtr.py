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
              ("py.tar.gz" "/media/sf_Matej/ariel/code/py.tar.gz")
              ("clus.jar" "/media/sf_Matej/ariel/code/clus.jar")
              ("conf.txt" "/media/sf_Matej/ariel/experiments/{1}/conf.txt"))
(outputfiles = ("results.txt" ""))
(runTimeEnvironment = "APPS/BASE/PYTHON-E8")
(queue != "gridgpu")"""


def find_for(subdir, xrsl):
    j0 = 0
    jobs = 500
    total_sum = 0
    clusters = create_all_clusters()
    for c in clusters:
        total_sum += len(c)
    package = []
    package_w = 0
    for i, c in tqdm(enumerate(clusters)):
        package_w += len(c)
        package.append(c)
        if package_w > total_sum / jobs or i == len(clusters) - 1:
            path_template = "../experiments/semi_mtr/{}/clusters{{}}/".format(subdir)
            experiment_dir_full = path_template.format(j0)
            j0 += 1
            # while os.path.exists(experiment_dir_full):
            #     j0 += 1
            #     experiment_dir_full = path_template.format(j0)
            os.makedirs(experiment_dir_full, exist_ok=True)
            # configurations
            with open(os.path.join(experiment_dir_full, "conf.txt"), "w", newline="") as f:
                print(package, file=f)
            # sh
            with open(os.path.join(experiment_dir_full, "pozeni.sh"), "w", newline="") as f:
                print("tar -xzf py.tar.gz", file=f)
                print("tar -xzf data.tar.gz", file=f)
                print("python3 cluster_semi_mtr.py conf.txt", file=f)  # TODO TODO TODO
            # xrsl
            with open(os.path.join(experiment_dir_full, xrsl), "w", newline="") as f:
                print(XRSL.format("semi{}".format(i),
                                  experiment_dir_full),
                      file=f)
            package = []
            package_w = 0
    print(j0, "packages created")


def perform_cluster_operation(is_denoised, path0="", path1="./"):
    with open(path0 + "conf.txt") as f:
        clusters = eval(f.readline())
    evaluate_options(False, clusters, path1, is_denoised)


def evaluate_options(is_test, clustering, path, denoised):
    # load dataset
    data, c_features, o_features, first_target_index = convert_stacked_to_mtr(is_test, True, path, denoised)
    print("Loaded data")
    n_examples, columns = data.shape
    i_features = list(range(first_target_index))
    i_targets = list(range(first_target_index, columns))
    # folds
    random.seed(2506)
    i_examples = list(range(n_examples))
    random.shuffle(i_examples)
    n_folds = 4
    q, r = n_examples // n_folds, n_examples % n_folds
    portions = [q + (i < r) for i in range(n_folds)]
    starting_fold_indices = [0]
    for i in range(n_folds):
        starting_fold_indices.append(starting_fold_indices[-1] + portions[i])
    train_test_indices = [([], []) for _ in range(n_folds)]
    for i in range(n_folds):
        for j in range(starting_fold_indices[i], starting_fold_indices[i + 1]):
            i_example = i_examples[j]
            for k in range(n_folds):
                train_test_indices[k][i == k].append(i_example)
    # evaluate
    results = []
    parameters = {}
    for cluster in clustering:
        predictions_full = np.zeros((0, 55))
        target_full = np.zeros((0, 55))
        for i in range(n_folds):
            indices_train, indices_test = train_test_indices[i]
            d_train = data[indices_train, :]
            d_test = data[indices_test, :]
            features_train, target_train = d_train[:, i_features], d_train[:, i_targets]
            features_test, target_test = d_test[:, i_features], d_test[:, i_targets]
            predictive_model = SemiMTR(target_group=cluster)
            parameters = predictive_model.__dict__
            predictive_model.fit(features_train, target_train, features_test)
            y_hat = predictive_model.predict(features_test)
            predictions_full = np.concatenate((predictions_full, y_hat))
            target_full = np.concatenate((target_full, target_test))
        results.append({tar: multi_score(target_full[:, tar], predictions_full[:, tar]) for tar in cluster})
    with open("results.txt", "w") as f:
        print("This are the results for", file=f)  # something that circumvents problems with missing first byte
        print("MODEL;{};{}".format(SemiMTR.__name__, parameters), file=f)
        print("PARAMETERS;{}".format(clustering), file=f)
        print(results, file=f)
    return results


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
    if len(sys.argv) == 1:
        # find_for("corr", "smart1.xrsl")
        # perform_cluster_operation("../experiments/semi_mtr/smarter/clusters0/", "../database/csv/")
        collect_the_results("../optimisation/str_mtr_clusters.txt")
    else:
        additional_arguments = sys.argv[1:]
        if "conf.txt" not in additional_arguments[0]:
            raise ValueError("Wrong argument")
        is_denoised = eval(additional_arguments[1])
        perform_cluster_operation(is_denoised)
