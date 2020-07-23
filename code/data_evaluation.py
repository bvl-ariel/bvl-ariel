import numpy as np


# @DeprecationWarning
def weird_mean_absolute_error(predictions: np.ndarray, true_values: np.ndarray):
    return np.mean(true_values * np.abs(predictions - true_values))


def mse(t, p, axis=None):
    return np.mean(np.square(t - p), axis=axis)


def mae(t, p, axis=None):
    return np.mean(np.abs(t - p), axis=axis)


def wmae(t, p, axis=None):
    return np.mean(t * np.abs(t - p), axis=axis)


def multi_score(t, p, axis=None):
    return {s.__name__: s(t, p, axis) for s in [mse, mae, wmae]}


def read_csv(file):
    try:
        lines = []
        with open(file) as f:
            f.readline()
            for l in f:
                lines.append(l.strip())
        return np.array(eval("".join(lines))).T
    except SyntaxError:
        return np.genfromtxt(file, delimiter='\t')


def evaluate_from_file(file, file_true, axis=None):
    matrix = read_csv(file)
    true = read_csv(file_true)
    r = multi_score(true, matrix, axis)
    if not axis:
        for x, y in r.items():
            print(x, list(y) if isinstance(y, np.ndarray) else y)
    return r["mse"]


if __name__ == "__main__":
    pattern = "../explore/xgb_lgb/{}_" + "-".join([str(t) for t in range(55)]) + ".csv"
    # evaluate_from_file(pattern.format("GradientBoosting"), "../optimisation/ttrue_values.csv", 0)
    bag_pred = "../experiments/bagging_best_feats_cv/RandomForest.csv"
    rez = evaluate_from_file(bag_pred,
                             "../optimisation/ttrue_values.csv", axis=None)
    # with open("../experiments/xgb_outlier/true_indices_11-12-13-14-15-16-17-18-19-20-21.csv") as f:
    #     indices = eval(f.readline())
    # if len(rez) != len(indices):
    #     print(len(rez), len(indices))
    # a = list(zip(indices, rez))
    # a.sort(key=lambda t: t[1], reverse=True)
    # print(a)
    # import matplotlib.pyplot as plt
    # plt.hist([x for x in rez if x < 0.0002], bins=100)
    # plt.show()
