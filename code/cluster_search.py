import os
import sys
from predictive_models import *
from tqdm import tqdm, trange
from utils import load_data
import random
from data_evaluation import multi_score
import itertools
try:
    import matplotlib.pyplot as plt
except:
    print("import matplotlib ...")
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
              ("conf.txt" "/media/sf_Matej/ariel/experiments/{1}/conf.txt"))
(outputfiles = ("results.txt" ""))
(runTimeEnvironment = "APPS/BASE/PYTHON-E8")
(queue != "gridgpu")"""


def evaluate_options(is_test, model_class, parameter_combinations):
    # load dataset
    data, i_target, instance_names, feature_names = load_data(is_test, True, "./")
    print("Loaded data")
    rows, columns = data.shape
    assert rows % 55 == 0
    n_examples = rows // 55
    i_features = [i for i in range(columns) if i != i_target]
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
    results = [[[] for _ in parameter_combinations] for _ in range(55)]
    for channel in tqdm(list(range(55 if not is_test else 2)), desc="channel"):
        d = data[channel::55, :]
        for i, parameters in tqdm(enumerate(parameter_combinations), desc="parameters"):
            r = evaluate_one(d, i_features, i_target, train_test_indices, model_class, parameters)
            results[channel][i].append(r)
    with open("results.txt", "w") as f:
        print("This are the results for", file=f)  # something that circumvents problems with missing first byte
        print("MODEL;" + model_class.__name__, file=f)
        print("PARAMETERS;{}".format(parameter_combinations), file=f)
        print(results, file=f)
    return results


def evaluate_one(data, i_features, i_target, train_test_indices, model_class, algorithm_parameters):
    n_folds = len(train_test_indices)
    results = []
    for fold in range(n_folds):
        train_test = [data[indices, :] for indices in train_test_indices[fold]]
        xs = [d[:, i_features] for d in train_test]
        y = [d[:, i_target] for d in train_test]
        # train
        model = model_class(**algorithm_parameters)  # type: PredictiveModel
        model.fit(xs[0], y[0])
        y_prediction = model.predict(xs[1])
        results.append(multi_score(y[1], y_prediction))
    return results


def test_search():
    a1 = evaluate_options(True, GradientBoosting, [{"eta": 0.01, "max_depth": 2, "subsample": 0.8,
                                                    "colsample_bytree": 1.0, "min_child_weight": 1.0, "n_trees": 2},
                                                   {"eta": 0.7, "max_depth": 3, "subsample": 1.0,
                                                    "colsample_bytree": 1.0, "min_child_weight": 1.0, "n_trees": 150}])
    a2 = evaluate_options(True, RandomForest, [{"max_features": 1.0, "min_samples_leaf": 5, "n_estimators": 50},
                                               {"max_features": 0.3, "min_samples_leaf": 10, "n_estimators": 2}])
    print(a1)
    print(a2)


def find_for_boosting(xrsl, is_test):
    def complexity(parameter_values):
        return parameter_values["n_trees"] * parameter_values["max_depth"]  # Yeah, I know ...

    parameter_names = ["eta", "max_depth", "subsample", "colsample_bytree", "min_child_weight", "n_trees"]
    # etas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    etas = [0.03, 0.05, 0.07, 0.1, 0.3, 0.6]
    # dephts = [4, 6, 8, 10, 12, 14, 20, 25, 30, 35]
    dephts = [10, 12, 14, 20, 25, 30, 35, 40, 45]
    subsamples = [0.6, 0.7, 0.8, 0.9, 1.0]
    colsamples = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    # child_w = [1, 2, 3, 5, 10, 15, 20]
    child_w = [1, 2, 3, 5]
    # n_trees = [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
    n_trees = [200, 225, 250, 300]
    parameters = []
    total_complexity = 0
    n_jobs = 1000
    for combination in itertools.product(etas, dephts, subsamples, colsamples, child_w, n_trees):
        e, d, s, c, w, n = combination
        # remove weird combinations
        if e < 0.05 and n < 10:
            continue
        elif d <= 6 and (w <= 10 or n <= 10):
            continue
        configuration = dict(zip(parameter_names, combination))
        parameters.append(configuration)
        total_complexity += complexity(configuration)
    print("Configurations:", len(parameters))
    if is_test:
        return
    find_for("GradientBoosting", "refined1", xrsl, total_complexity, n_jobs, parameters, complexity)


def find_for_light_boosting(xrsl, is_test):
    def complexity(parameter_values):
        return 1  # Yeah, I know ...

    parameter_names = ["boosting_type"]
    ensemble_types = ["dart", "goss"]
    parameters = []
    total_complexity = 0
    n_jobs = 2
    for combination in itertools.product(ensemble_types):
        configuration = dict(zip(parameter_names, combination))
        parameters.append(configuration)
        total_complexity += complexity(configuration)
    print("Configurations:", len(parameters))
    if is_test:
        return
    find_for("LightGradientBoosting", "boost_type", xrsl, total_complexity, n_jobs, parameters, complexity)


def find_for_forest(xrsl, is_test):
    def complexity(parameter_values):
        return parameter_values["n_estimators"]  # Yeah, I know ...

    parameter_names = ["max_features", "min_samples_leaf", "n_estimators", "criterion"]
    features = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    min_samples_leaf = [1, 2, 3, 4, 6, 8, 10, 12, 14, 20, 25, 30, 35]
    n_estimators = [10, 50, 100, 150, 200, 250, 300, 350, 400, 500]
    criterions = ["mse", "mae"]
    parameters = []
    total_complexity = 0
    n_jobs = 300
    for combination in itertools.product(features, min_samples_leaf, n_estimators, criterions):
        configuration = dict(zip(parameter_names, combination))
        parameters.append(configuration)
        total_complexity += complexity(configuration)
    print("Configurations:", len(parameters))
    if is_test:
        return
    find_for("RandomForest", "basic", xrsl, total_complexity, n_jobs, parameters, complexity)


def find_for_semi_mtr(xrsl, is_test):
    def complexity(parameter_values):
        return len(parameter_values["target_groups"])  # Yeah, I know ...

    parameter_names = ["ensemble_method", "feature_subset", "min_leaf_size", "n_trees"  "target_groups"]
    ensemble_methods = ["RForest"]
    features = [0.8, 0.9, 1.0]
    min_samples_leaf = [1]
    n_estimators = [200]
    subgroups = [list(range(55)), [[i] for i in range(55)]]  # mtr, str
    subgroups += create_all()
    random.seed(123)
    parameters = []
    total_complexity = 0
    n_jobs = 300
    for combination in itertools.product(ensemble_methods, features, min_samples_leaf, n_estimators, subgroups):
        configuration = dict(zip(parameter_names, combination))
        parameters.append(configuration)
        total_complexity += complexity(configuration)
    print("Configurations:", len(parameters))
    if is_test:
        return
    find_for("RandomForest", "basic", xrsl, total_complexity, n_jobs, parameters, complexity)


def find_for(model_name, experiment_dir, xrsl, total_complexity, n_jobs, parameters, complexity):
    packages = [[]]
    current_complexity = 0.0
    for configuration in parameters:
        c = complexity(configuration)
        if current_complexity + c > total_complexity / n_jobs:
            assert packages[-1]
            packages.append([])
            current_complexity = 0
        packages[-1].append(configuration)
        current_complexity += c
    for i, package in tqdm(enumerate(packages)):
        experiment_dir_full = "../experiments/grid_search/{}/{}/package{}".format(model_name, experiment_dir, i)
        os.makedirs(experiment_dir_full, exist_ok=True)
        # configurations
        with open(os.path.join(experiment_dir_full, "conf.txt"), "w", newline="") as f:
            print(model_name, file=f)
            print(package, file=f)
        # sh
        with open(os.path.join(experiment_dir_full, "pozeni.sh"), "w", newline="") as f:
            print("tar -xzf py.tar.gz", file=f)
            print("tar -xzf data.tar.gz", file=f)
            print("python3 cluster_search.py conf.txt", file=f)
        # xrsl
        with open(os.path.join(experiment_dir_full, xrsl), "w", newline="") as f:
            print(XRSL.format("_".join([model_name, experiment_dir, str(i)]),
                              experiment_dir_full),
                  file=f)


def perform_cluster_operation():
    with open("conf.txt") as f:
        model_class = eval(f.readline())
        parameters = eval(f.readline())
    evaluate_options(False, model_class, parameters)


def load_results_file(results_file, should_flatten):
    def maybe_flatten(l):
        if should_flatten:
            return l[0]
        else:
            return l

    def line_iterator(file_name):
        with open(file_name) as f_handle:
            for _ in range(3):
                f_handle.readline()
            for line in f_handle:
                yield [maybe_flatten(parameter_results) for parameter_results in eval(line)]

    # first, count lines
    lines = 0
    with open(results_file) as f:
        for _ in f:
            lines += 1
    first_line = "This are the results for"
    with open(results_file) as f:
        first = f.readline().strip()
        if first != first_line:
            print(results_file, first, "is weird")
        model, parameters = [f.readline().strip().split(';')[1] for _ in range(2)]
        parameters = eval(parameters)
        if lines == 4:
            # all results in one line
            results = eval(f.readline())
            flattened = iter([[maybe_flatten(parameter_results) for parameter_results in channel_results]
                              for channel_results in results])
        elif lines == 58:
            flattened = line_iterator(results_file)
        else:
            raise ValueError("Wrong number of lines: {}".format(lines))

        return model, parameters, flattened


def collect_results(experiment_dir, should_flatten):
    """
    Joins the results from different packages into a single file.
    :param experiment_dir:
    :param should_flatten: True if the files in the experiment_dir are from the cluster experiments where
    the results are one level too deep. In the joined file, we get rid of the redundant dimension.
    :return:
    """
    joined_parameters = []
    with open(os.path.join(experiment_dir, "results.txt"), "w") as f:
        for channel in trange(55):
            model = "?"
            joined_results = []
            for package in os.listdir(experiment_dir):
                results_file = os.path.join(experiment_dir, package, "results", "results.txt")
                if not os.path.exists(results_file):
                    # print(package, "has no results.txt")
                    continue

                model, parameters, flattened = load_results_file(results_file, should_flatten)
                n_items = 0
                if channel == 0:
                    joined_parameters += parameters
                # for joined_parameter_results, parameter_results in zip(joined_results, flattened):
                for i, channel_results in enumerate(flattened):
                    if i == channel:
                        joined_results += channel_results
                    n_items += 1
                assert n_items == 55
            if channel == 0:
                print("This are the results for", file=f)
                print("MODEL;{}".format(model), file=f)
                print("PARAMETERS;{}".format(joined_parameters), file=f)
            print(joined_results, file=f)
    print("Results collected")


def analyze_the_results(experiment_dir, should_plot):
    """
    For each parameter, we plot its marginal distribution, for every metric.
    :param experiment_dir:
    :return:
    """
    result_file = os.path.join(experiment_dir, "results.txt")
    graph_dir = os.path.join(experiment_dir, "plots")
    model, parameters, _ = load_results_file(result_file, False)
    parameter_values = {p: set() for p in parameters[0]}
    for ps in parameters:
        for p, v in ps.items():
            parameter_values[p].add(v)
    parameter_values = {p: sorted(vs) for p, vs in parameter_values.items()}
    measures = []  # performances is an iterator, so do not touch it in advance
    if should_plot:
        # for each parameter and each channel, plot the distribution of the parameter
        for p, vs in tqdm(parameter_values.items(), desc="parameter"):
            x_labels = [str(v) for v in vs]
            x_positions = list(range(len(vs)))
            v_to_x = {x: i for i, x in enumerate(vs)}

            ig, axs = plt.subplots(55, 3, sharex=True, figsize=(20, 160))
            # colors = dict(zip(measures, "rgbk"))
            model, parameters, performances = load_results_file(result_file, False)
            for i, channel_performances in tqdm(enumerate(performances), desc="channel"):
                if i == 0:
                    measures = sorted(channel_performances[0][0])
                for j, measure in enumerate(measures):
                    axis = axs[i, j]
                    ys = [[] for _ in x_positions]  # type: List[List[float]]
                    assert len(parameters) == len(channel_performances)
                    for params, performances_fold in zip(parameters, channel_performances):
                        x = v_to_x[params[p]]
                        y = np.mean([performance[measure] for performance in performances_fold])  # type: float
                        # xs.append(x)
                        ys[x].append(y)
                    # plotting all xs and ys might be too costly, we will plot min, max and percentiles.
                    x_optimal = -1
                    y_optimal = float("inf")
                    y_max = float("-inf")
                    for x, ys_x in enumerate(ys):
                        ys_x.sort()
                        if ys_x[0] < y_optimal:
                            x_optimal = x
                            y_optimal = ys_x[0]
                        if ys_x[-1] > y_max:
                            y_max = ys_x[-1]
                        indices = sorted(
                            {index for index in {len(ys_x)} | {int(p / 100 * len(ys_x)) for p in range(101)}
                             if index < len(ys_x)}
                        )
                        ys_to_show = [ys_x[index] for index in indices]
                        xs_to_show = [x for _ in ys_to_show]
                        axis.scatter(xs_to_show, ys_to_show, marker=',', c="b")
                    # find optimal value
                    axis.scatter([x_optimal], [y_optimal], marker="x", c="r")

                    axis.set_title("Channel {}: {} with arg min = {}".format(i + 1, measure, x_labels[x_optimal]))
                    axis.set_xticks(x_positions)
                    axis.set_xticklabels(x_labels)
                    axis.set_ylim([y_optimal * 0.95, y_max * 1.05])
                    # Hide x labels and tick labels for top plots and y ticks for right plots.
                    # for ax in axs.flat:
                    #     ax.label_outer()
            plt.tight_layout()
            os.makedirs(graph_dir, exist_ok=True)
            plt.savefig(os.path.join(graph_dir, "{}_{}.pdf".format(model, p)))
            plt.close()
    model, parameters, performances = load_results_file(result_file, False)
    with open(os.path.join(experiment_dir, "best.txt"), "w") as f:
        for channel, channel_performances in enumerate(performances):
            measures = sorted(channel_performances[0][0])
            best_performances = {m: (float("inf"), None) for m in measures}
            for params, performances_fold in zip(parameters, channel_performances):
                y = np.mean([[performance[m] for m in measures] for performance in performances_fold], axis=0)
                assert len(y) == len(measures)
                for p, m in zip(y, measures):
                    if p < best_performances[m][0]:
                        best_performances[m] = (p, params)
            print(channel, best_performances, file=f)


def fast_join_of_best(files, out_f):
    measures = ["mae", "wmae", "mse"]
    best_results = [{measure: (float("inf"), None) for measure in measures} for _ in range(55)]
    for file in files:
        with open(file) as f:
            for i, line in enumerate(f):
                space = line.find(" ")
                channel = int(line[:space])
                assert channel == i
                performances = eval(line[space + 1:])
                for m in measures:
                    e_best = best_results[channel][m][0]
                    e_now = performances[m][0]
                    if e_now < e_best:
                        best_results[channel][m] = (e_now, performances[m][1])
    with open(out_f, "w") as f:
        for channel, best in enumerate(best_results):
            print("{} {}".format(channel, best), file=f)


if __name__ == "__main__":
    additional_arguments = sys.argv[1:]
    if len(additional_arguments) == 0:
        print("No additional arguments ...")
        # find_for_forest("krog1.xrsl", False)
        # find_for_boosting("krog1.xrsl", False)
        find_for_light_boosting("bu1.xrsl", False)
        if 0:
            # experiment_dir = "../experiments/grid_search/{}/refined1".format(["GradientBoosting", "RandomForest"][0])
            experiment_dir = "../experiments/grid_search/GradientBoosting/"
            # collect_results(experiment_dir, True)
            # analyze_the_results(experiment_dir, False)
            fast_join_of_best([experiment_dir + "basic/best.txt", experiment_dir + "refined1/best.txt"],
                              experiment_dir + "best.txt")

    else:
        print("Probably, this is cluster :)")
        if "conf.txt" not in additional_arguments[0]:
            raise ValueError("Wrong argument")
        perform_cluster_operation()
