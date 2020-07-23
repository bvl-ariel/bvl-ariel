from typing import Dict, List, Any, Union

try:
    from tqdm import tqdm, trange
except ImportError:
    print("tqdm could not be imported")
import os
import random
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from data_evaluation import wmae, mse
import itertools
from predictive_models import *
from utils import load_data, convert_stacked_to_mtr
import sys


def train_cross_val(predictive_model_classes, is_test, opt_params_files, measure, channels=None, out_dir=None,
                    classic_fresh="classic", **alg_options):
    def get_train_test(i_fold):
        i_test = [partition[part] for part in range(i_fold, len(partition), k)]
        i_train = sorted(set(range(len(partition))) - set(i_test))
        return i_train, i_test

    opt_parameters = [[{} for _ in range(55)] for _ in predictive_model_classes]  # type: List[List[Dict[str, Any]]]
    for i, opt_params_file in enumerate(opt_params_files):
        if opt_params_file is not None:
            opt_parameters[i] = load_optimal_params_file(opt_params_file, measure)
    if alg_options != {}:
        if len(predictive_model_classes) > 1:
            raise ValueError("Cannot have additional parameters for different models")
        for i in range(55):
            opt_parameters[0][i] = {**opt_parameters[0][i], **alg_options}

    data, i_target, instance_names, feature_names = load_data(is_test, True, path="./", classic_fresh=classic_fresh)
    assert len(instance_names) % 55 == 0
    instance_names = instance_names[::55]
    rows, columns = data.shape
    i_features = [i for i in range(columns) if i != i_target]
    assert rows % 55 == 0
    partition = list(range(rows // 55))
    random.seed(123)
    random.shuffle(partition)
    k = 10 if not is_test else 3
    maes = [[] for _ in predictive_model_classes]
    is_partial = channels is not None
    if not is_partial:
        channels = list(range(55)) if not is_test else list(range(4))
    true_values = [[] for _ in range(55)]
    predictions = [[[] for _ in range(55)] for _ in predictive_model_classes]
    test_indices_sorted = []
    is_first_round = True
    for channel in tqdm(channels):
        d = data[channel::55, :]
        for fold in range(k):
            indices_train, indices_test = get_train_test(fold)
            if is_first_round:
                test_indices_sorted += indices_test
            d_train = d[indices_train, :]
            d_test = d[indices_test, :]
            features_train, target_train = d_train[:, i_features], d_train[:, i_target]
            features_test, target_test = d_test[:, i_features], d_test[:, i_target]
            for i, predictive_model_class in enumerate(predictive_model_classes):
                opt_parameters_channel = opt_parameters[i][channel]  # type: Dict[str, Any]
                predictive_model = predictive_model_class(**opt_parameters_channel)  # type: PredictiveModel
                predictive_model.fit(features_train, target_train, features_test)
                y_hat = predictive_model.predict(features_test)
                if i == 0:
                    true_values[channel] += list(target_test)
                predictions[i][channel] += list(y_hat)
                maes[i].append(wmae(target_test, y_hat))
                # print("mse and wmae for fold", fold, mse(target_test, y_hat), wmae(target_test, y_hat))
        is_first_round = False

    n = len(predictive_model_classes)
    if is_partial:
        test_indices = [i for i_fold in range(k) for i in get_train_test(i_fold)[1]]
        inverse_permutation = [pair[1] for pair in sorted([(x, i) for i, x in enumerate(test_indices)])]
        os.makedirs(out_dir, exist_ok=True)
        for i in range(n):
            file = os.path.join(out_dir, "{}_{}.csv".format(predictive_model_classes[i].__name__,
                                                            "-".join([str(c) for c in channels])
                                                            )
                                )
            with open(file, "w") as f:
                ps = [[predictions[i][channel][j] for j in inverse_permutation] for channel in channels]
                print(channels, file=f)
                print(ps, file=f)
        # np.savetxt(os.path.join("../optimisation/true_values.csv"),
        #            np.array([[true_values[channel][j] for j in inverse_permutation] for channel in channels]).T,
        #            delimiter='\t',
        #            fmt='%.18f')
        with open(os.path.join(out_dir, "true_indices_{}.csv".format("-".join(str(c) for c in channels))), "w") as f:
            print(list(instance_names[test_indices_sorted]), file=f)
    else:
        # join by channel
        predictions = [[y for channel_predictions in model_predictions for y in channel_predictions]
                       for model_predictions in predictions]
        predictions = np.array(predictions)
        for k in range(1, n + 1):
            for combination in itertools.combinations(range(n), k):
                if k == 2:
                    steps = 10
                    weights = [[p / steps, 1 - p / steps] for p in range(1, steps)]
                else:
                    weights = [[1 / k] * k]
                for ws in weights:
                    ensemble_p = np.average(predictions[list(combination), :], axis=0, weights=ws)
                    mae_e = wmae(np.array(true_values), ensemble_p)
                    mse_e = mean_squared_error(np.array(true_values), ensemble_p)
                    models = [predictive_model_classes[i].__name__ for i in combination]
                    print("Ensemble of", models, "with ws", ws, "weird mae", mae_e, "mse", mse_e)


def join_prediction_files(exp_dir, model_class, channel_groups):
    files = [os.path.join(exp_dir, "{}_{}.csv".format(model_class.__name__, "-".join(str(t) for t in channel_group)))
             for channel_group in channel_groups]
    all_channels = sorted([x for c in channel_groups for x in c])
    results = [[] for _ in all_channels]
    for file in files:
        with open(file) as f:
            channels = eval(f.readline())
            predictions = eval(f.readline())
            for c, p in zip(channels, predictions):
                results[c] = p
    np.savetxt(os.path.join(exp_dir, model_class.__name__ + ".csv"), np.array(results).T, delimiter='\t', fmt='%.18f')


def train_models_real(channel_indices, experiment_dir, predictive_model_class, is_test, opt_params_file,
                      measure_to_optimize, **kwargs):
    data, i_target, instance_names, feature_names = load_data(is_test, False)
    rows, columns = data.shape
    i_features = [i for i in range(columns) if i != i_target]
    feature_names = [feature_names[i] for i in i_features]
    # find test instances
    channels = 55
    first_test = rows
    while np.isnan(data[first_test - channels, i_target]):
        first_test -= channels
    data_train = data[:first_test, :]
    data_test = data[first_test:, :]
    predictions = []

    opt_parameters = [{} for _ in range(55)]  # type: List[Dict[str, Any]]
    if opt_params_file is not None:
        opt_parameters = load_optimal_params_file(opt_params_file, measure_to_optimize)

    for channel in tqdm(channel_indices):
        d_train = data_train[channel::channels, :]
        d_test = data_test[channel::channels, :]

        features_train, target_train = d_train[:, i_features], d_train[:, i_target]
        features_test = d_test[:, i_features]
        opt_parameters_channel = opt_parameters[channel]
        predictive_model = predictive_model_class(**opt_parameters_channel, **kwargs)
        predictive_model.fit(features_train, target_train, features_test)
        predictions.append(list(predictive_model.predict(features_test)))
    os.makedirs(experiment_dir, exist_ok=True)

    np.savetxt(os.path.join(experiment_dir, "predictions_test{}.csv".format(channel_indices[0])),
               np.array(predictions).T, delimiter='\t',
               fmt='%.18f')
    np.savetxt(os.path.join(experiment_dir, "features.csv"), np.array([-1]), header=','.join(feature_names))
    # np.savetxt(os.path.join(experiment_dir, "feature_importances.csv"), np.array(feature_importances),
    #            header=','.join(feature_names), delimiter=',')
    with open(os.path.join(experiment_dir, "meta.txt"), "w") as f:
        print(predictive_model.__dict__, file=f)


def train_models_real_multi(experiment_dir, predictive_model_class, is_test, **kwargs):
    data, c_features, o_features, first_target_index = convert_stacked_to_mtr(is_test, False)
    print("Loaded data")
    n_examples, columns = data.shape
    i_features = list(range(first_target_index))
    i_targets = list(range(first_target_index, columns))
    rows, _ = data.shape
    first_test = rows
    while np.isnan(data[first_test - 1, first_target_index]):
        first_test -= 1
    d_train = data[:first_test, :]
    d_test = data[first_test:, :]

    features_train, target_train = d_train[:, i_features], d_train[:, i_targets]
    features_test = d_test[:, i_features]

    predictive_model = predictive_model_class(**kwargs)
    predictive_model.fit(features_train, target_train, features_test)
    ys = predictive_model.predict(features_test)
    os.makedirs(experiment_dir, exist_ok=True)

    np.savetxt(os.path.join(experiment_dir, "predictions_test.csv"), np.array(ys), delimiter='\t',
               fmt='%.18f')
    # np.savetxt(os.path.join(experiment_dir, "feature_importances.csv"), np.array(feature_importances),
    #            header=','.join(feature_names), delimiter=',')
    with open(os.path.join(experiment_dir, "meta.txt"), "w") as f:
        print(predictive_model.__dict__, file=f)


def train_models_real_multi_str(experiment_dir, is_test, clusters_file, measure_to_optimize, **kwargs):
    data, c_features, o_features, first_target_index = convert_stacked_to_mtr(is_test, False)
    print("Loaded data")
    n_examples, columns = data.shape
    i_features = list(range(first_target_index))
    i_targets = list(range(first_target_index, columns))
    rows, _ = data.shape
    first_test = rows
    while np.isnan(data[first_test - 1, first_target_index]):
        first_test -= 1
    d_train = data[:first_test, :]
    d_test = data[first_test:, :]

    features_train, target_train = d_train[:, i_features], d_train[:, i_targets]
    features_test = d_test[:, i_features]

    if clusters_file is not None:
        with open(clusters_file) as f:
            optimal_clusters = eval(f.readline())
    else:
        cs = [list(range(55)) for _ in range(55)]
        optimal_clusters = {"mse": cs, "mae": cs, "wmae": cs}

    predictions = np.zeros((d_test.shape[0], 0))
    parameters = {}
    for channel in tqdm(range(55)):
        cluster = optimal_clusters[measure_to_optimize][channel]
        predictive_model = RandomForestSTRviaMTR(target=channel, target_group=cluster, **kwargs)
        if channel == 0:
            parameters = predictive_model.__dict__
        predictive_model.fit(features_train, target_train, features_test)
        ys = predictive_model.predict(features_test)
        predictions = np.concatenate([predictions, ys], axis=1)
    os.makedirs(experiment_dir, exist_ok=True)
    np.savetxt(os.path.join(experiment_dir, "predictions_test.csv"), predictions, delimiter='\t',
               fmt='%.18f')
    with open(os.path.join(experiment_dir, "meta.txt"), "w") as f:
        print(parameters, file=f)


def train_models_real_channel(experiment_dir, predictive_model_class, is_test, opt_params_file, measure_to_optimize,
                              channel, **kwargs):
    data, i_target, instance_names, feature_names = load_data(is_test, False, ".")
    rows, columns = data.shape
    i_features = [i for i in range(columns) if i != i_target]
    feature_names = [feature_names[i] for i in i_features]
    channels = 55
    # find test instances
    first_test = rows
    while np.isnan(data[first_test - channels, i_target]):
        first_test -= channels
    data_train = data[:first_test, :]
    data_test = data[first_test:, :]
    predictions = []

    opt_parameters = [{} for _ in range(55)]  # type: List[Dict[str, Any]]
    if opt_params_file is not None:
        opt_parameters = load_optimal_params_file(opt_params_file, measure_to_optimize)

    d_train = data_train[channel::channels, :]
    d_test = data_test[channel::channels, :]

    features_train, target_train = d_train[:, i_features], d_train[:, i_target]
    features_test = d_test[:, i_features]
    if len(kwargs) == 0:
        opt_parameters_channel = opt_parameters[channel]
    else:
        opt_parameters_channel = kwargs
    predictive_model = predictive_model_class(**opt_parameters_channel)
    predictive_model.fit(features_train, target_train, features_test)
    predictions.append(list(predictive_model.predict(features_test)))
    os.makedirs(experiment_dir, exist_ok=True)

    np.savetxt(os.path.join(experiment_dir, "predictions{}_test.csv".format(channel)), np.array(predictions).T,
               delimiter='\t',
               fmt='%.18f')
    np.savetxt(os.path.join(experiment_dir, "features{}.csv".format(channel)), np.array([-1]),
               header=','.join(feature_names))
    # np.savetxt(os.path.join(experiment_dir, "feature_importances.csv"), np.array(feature_importances),
    #            header=','.join(feature_names), delimiter=',')
    with open(os.path.join(experiment_dir, "meta{}.txt".format(channel)), "w") as f:
        print(predictive_model.__dict__, file=f)


def grid_search(is_test):
    def create_new_interval(pair, best_value, parameter):
        is_round = False  # parameter == m_depth
        q = 0.8
        a, b = pair
        l_new = (b - a) * q
        if l_new <= min_lengths[parameter]:
            return [best_value, best_value]
        a_new = max(a, best_value - 0.5 * l_new)
        b_new = min(b, best_value + 0.5 * l_new)
        if is_round:
            a_new = round(a_new)
            b_new = round(b_new)
        return [a_new, b_new]

    data, i_target, instance_names, feature_names = load_data(is_test, True)
    _, columns = data.shape
    i_features = [i for i in range(columns) if i != i_target]
    best_params_channels = []
    # m_depth = 'max_depth'
    # m_leaf = 'min_samples_leaf'
    for channel in tqdm(list(range(55 if not is_test else 2))):
        print("Channel", channel)
        d = data[channel::55, :]
        features, target = d[:, i_features], d[:, i_target]
        # estimator = GradientBoosting()
        # intervals = {'eta': [0.01, 1.0], 'max_depth': [1, 20], "subsample": [0.1, 1.0],
        #              "colsample_bytree": [0.1, 1.0]}
        # min_lengths = {'eta': 0.01, 'max_depth': 1, "subsample": 0.01, "colsample_bytree": 0.05}
        # intervals = {"max_features": [0.1, 1.0], "min_samples_leaf": [1, 100]}
        # min_lengths = {"max_features": 0.05, "min_samples_leaf": 1}

        # features = np.random.rand(100, 3)
        # target = features[:, 0] + np.random.rand(100) * 0.05
        # intervals = {'eta': [0.01, 1.0], 'max_depth': [1, 20], "subsample": [0.1, 1.0],
        #              "colsample_bytree": [0.1, 1.0]}
        # min_lenghts = {'eta': 0.01, 'max_depth': 1, "subsample": 0.01, "colsample_bytree": 0.05}
        intervals = {'epsilon': [1.0, 1.0], "C": [0.001, 1000.0], "gamma": [2 ** -5, 2 ** 15]}
        min_lengths = {'epsilon': 0.001, "C": 0.001, "gamma": 2 ** -5}
        best_params = None
        best_score = float("-inf")  # because python local score manipulation
        for _ in range(2):
            grid = {}
            for key, interval in intervals.items():
                left, right = interval
                if right - left <= min_lengths[key]:
                    n_points = 1
                else:
                    n_points = 20 if not is_test else 3
                grid[key] = list(np.geomspace(left, right, n_points))
                random.shuffle(grid[key])
                # grid[key] = list(np.linspace(left, right, n_points))
                # random.shuffle(grid[key])
            # grid[m_depth] = list(np.around(sorted(set(round(depth) for depth in grid[m_depth]))).astype(int))
            # grid[m_leaf] = list(np.around(sorted(set(round(depth) for depth in grid[m_leaf]))).astype(int))
            if max(len(values) for values in grid.values()) == 1:
                break
            # grid[m_depth] = list(np.around(sorted(set(round(depth) for depth in grid[m_depth]))).astype(int))
            print("Grid", grid)
            my_scorer = make_scorer(mean_squared_error, greater_is_better=False)
            # make_scorer(wmae, greater_is_better=False)
            # estimator = BMachine()
            estimator = RandomForest()
            search = GridSearchCV(estimator=estimator,
                                  param_grid=grid,
                                  scoring=my_scorer,
                                  iid=False,
                                  cv=3,
                                  refit=False,
                                  n_jobs=6)
            search.fit(features, target)
            print(search.cv_results_["mean_test_score"], search.cv_results_["rank_test_score"])
            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_params = search.best_params_
                print("Found better", best_score, best_params)
            for key in intervals:
                intervals[key] = create_new_interval(intervals[key], best_params[key], key)
        best_params_channels.append(best_params)
    return best_params_channels


def grid_search_boosting(is_test):
    def create_new_interval(pair, best_value, parameter):
        is_round = False  # parameter == m_depth
        q = 0.8
        a, b = pair
        l_new = (b - a) * q
        if l_new <= min_lengths[parameter]:
            return [best_value, best_value]
        a_new = max(a, best_value - 0.5 * l_new)
        b_new = min(b, best_value + 0.5 * l_new)
        if is_round:
            a_new = round(a_new)
            b_new = round(b_new)
        return [a_new, b_new]

    data, i_target, instance_names, feature_names = load_data(is_test, True)
    _, columns = data.shape
    i_features = [i for i in range(columns) if i != i_target]
    best_params_channels = []
    m_depth = 'max_depth'
    m_leaf = 'min_child_weight'
    for channel in tqdm(list(range(55 if not is_test else 2))):
        print("Channel", channel)
        d = data[channel::55, :]
        features, target = d[:, i_features], d[:, i_target]
        intervals = {'eta': [0.01, 1.0], 'max_depth': [1, 20], "subsample": [0.5, 1.0],
                     "colsample_bytree": [0.1, 1.0], "min_child_weight": [1, 20]}
        min_lengths = {'eta': 0.01, 'max_depth': 1, "subsample": 0.05, "colsample_bytree": 0.05,
                       "min_child_weight": 1}

        best_params = None
        best_score = float("-inf")  # because python local score manipulation
        for _ in range(1):
            grid = {}
            for key, interval in intervals.items():
                left, right = interval
                if right - left <= min_lengths[key]:
                    n_points = 1
                else:
                    n_points = 10 if not is_test else 2
                grid[key] = list(np.linspace(left, right, n_points))
                # random.shuffle(grid[key])
            grid[m_depth] = list(np.around(sorted(set(round(x) for x in grid[m_depth]))).astype(int))
            grid[m_leaf] = list(np.around(sorted(set(round(x) for x in grid[m_leaf]))).astype(int))
            if max(len(values) for values in grid.values()) == 1:
                break
            print("Grid", grid)
            my_scorer = make_scorer(mean_squared_error, greater_is_better=False)
            estimator = GradientBoosting()
            search = GridSearchCV(estimator=estimator,
                                  param_grid=grid,
                                  scoring=my_scorer,
                                  iid=False,
                                  cv=3,
                                  refit=False,
                                  n_jobs=6)
            search.fit(features, target)
            print("Mean test scores, ranks and parameter tuples")
            print(list(search.cv_results_["mean_test_score"]), list(search.cv_results_["rank_test_score"]),
                  search.cv_results_["params"], sep="\n")
            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_params = search.best_params_
                print("Found better", best_score, best_params)
            for key in intervals:
                intervals[key] = create_new_interval(intervals[key], best_params[key], key)
        best_params_channels.append(best_params)
    return best_params_channels


def load_optimal_params_file(file, measure: Union[None, str] = None) -> List[Dict]:
    should_save = False
    last_lines = []
    with open(file) as f:
        for l in f:
            if l.startswith('0'):
                should_save = True
            if should_save:
                dictionary = eval(l[l.find(' ') + 1:])
                if measure is not None:
                    dictionary = dictionary[measure][1]
                last_lines.append(dictionary)
    assert len(last_lines) == 55
    return last_lines


if __name__ == '__main__':
    # train_models_real_multi("../experiments/semi_mtr_bagg250/", SemiMTR, False,
    #                         target_groups=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 41, 42, 43, 44, 45, 46, 47],
    #                                        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 48, 49, 50, 51, 52, 53, 54]])
    # train_models_real_multi_str("../experiments/submission29_str_via_mtr_250/",
    #                             False, "../optimisation/str_mtr_clusters.txt",
    #                             "mse")

    # train_random_forest()
    # experiment_dir, predictive_model_class, is_test, opt_params_file, measure_to_optimize, **kwargs

    # if len(sys.argv) == 1:
    #     train_cross_val([GradientBoosting, LightGradientBoosting], False,
    #                     ["../optimisation/grid_cluster_xgb.txt", None], "wmae",
    #                     list(range(0, 55)), "../explore/xgb_lgb")
    #     exit(0)
    #
    model = sys.argv[1]
    indices = eval(sys.argv[2])
    if model == "xgb":
        assert indices == list(range(55))
        train_models_real(indices, "../experiments/xgb_latest_feat_optimised", GradientBoosting, False,
                          "../optimisation/grid_cluster_xgb.txt", "mse")
    elif model == "rf":
        train_models_real(indices, "../experiments/bagging_latest_feat_optimised2", RandomForest, False,
                          "../optimisation/grid_cluster_rf.txt", "mse")
    # else:
    #     raise ValueError("Wong model")
    # train_models_real(list(range(55)), "../experiments/weird_mae_bagging_250", ClusForest, False, None, None)

    # train_cross_val([GradientBoosting, RandomForest], False, [None, None])
    # train_cross_val([ClusForest],
    #                 False, [None], "wmae")
    # prvi = int(sys.argv[1])
    # drugi = prvi + 1
    # feature_type = sys.argv[2]
    # train_models_real(list(range(prvi, drugi)), "../experiments/bagging_of_200boosting", BaggingBoosting, False,
    #                   "../optimisation/grid_cluster_xgb.txt", "mse", n_bags=200)
    # train_cross_val([GradientBoosting], False, ["../optimisation/grid_cluster_xgb.txt"], "mse",
    #                 list(range(1)),
    #                 "../experiments/xgb_outlier")  # "../optimisation/grid_cluster_xgb.txt"
    # train_models_real_multi("../optimisation/clus_ros", SemiMTR, False, target_groups=[list(range(55))], ensemble_method="ExtraTrees", n_trees=150,  ROSTargetSubspaceSize=0.75, ROSAlgorithmType="FixedSubspaces")
    # train_cross_val([RandomForest], False, ["grid_cluster_rf.txt"], "mse", list(range(prvi, drugi)), "./", feature_type)
    # train_cross_val([GradientBoosting, RandomForest], False, ["../optimisation/grid_cluster_xgb.txt", "../optimisation/grid_cluster_rf.txt"], "wmae", list(range(22, 33)), "../optimisation/best_w")
    # train_cross_val([GradientBoosting, RandomForest], False, ["../optimisation/grid_cluster_xgb.txt", "../optimisation/grid_cluster_rf.txt"], "wmae", list(range(33, 44)), "../optimisation/best_w")
    # train_cross_val([GradientBoosting, RandomForest], False, ["../optimisation/grid_cluster_xgb.txt", "../optimisation/grid_cluster_rf.txt"], "wmae", list(range(44, 50)), "../optimisation/best_w")
    # train_cross_val([GradientBoosting, RandomForest], False, ["../optimisation/grid_cluster_xgb.txt", "../optimisation/grid_cluster_rf.txt"], "wmae", list(range(50, 55)), "../optimisation/best_w")
    # train_cross_val([GradientBoosting], False, [None], "wmae", list(range(55)), "../optimisation/best_w")
    # train_cross_val([RandomForest, GradientBoosting, BMachine, KNN], False,
    #                 [None, "../optimisation/grid_xgb.txt", None, None])  # "../optimisation/grid_svm.txt"
    # train_cross_val([BMachine], False, ["../optimisation/grid_svm.txt"])
    # train_cross_val([BaggingSVM], False, ["../optimisation/grid_svm.txt"])
    # for ii, x in enumerate(grid_search_boosting(False)):
    #     print(ii, x)
    # join_prediction_files("../optimisation/test_best_w", GradientBoosting, [[0, 2], [1]])
    # join_prediction_files("../experiments/xgb_from_10_sub_or_so", GradientBoosting, [list(range(0, 11)),
    #                                                                             list(range(11, 22)),
    #                                                                             list(range(22, 33)),
    #                                                                             list(range(33, 44)),
    #                                                                             list(range(44, 55))])
    # join_prediction_files("../experiments/bagging_sub10_cv", RandomForest, [[i] for i in range(55)])
    # c = int(sys.argv[1])
    # tree0 = int(sys.argv[2])
    # tree1 = int(sys.argv[3])
    # bag_selection = "[{},{}]".format(tree0, tree1)
    # train_models_real_channel("./", ClusForest, False, None, None, c, bag_selection=bag_selection, min_leaf_size=1,
    #                           split_heuristic="WeirdMAE")
