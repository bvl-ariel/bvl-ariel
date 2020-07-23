import matplotlib.pyplot as plt
from data_evaluation import *
import numpy as np


def plot_predictions(true_csv, predictions_csv, id_file):
    ys_true = read_csv(true_csv)
    ys_pred = read_csv(predictions_csv)
    with open(id_file) as f:
        ids = eval(f.readline())
    ae = ys_true * np.abs(ys_pred - ys_true)
    # planet_errors = multi_score(ys_true, ys_pred, axis=1)["wmae"]
    mean_per_planet = np.mean(ae, axis=1)
    max_per_planet = np.max(ae, axis=1)
    assert len(ids) == len(mean_per_planet)
    planets_mean = sorted(zip(ids, mean_per_planet), key=lambda t: -t[1])
    planets_max = sorted(zip(ids, max_per_planet), key=lambda t: -t[1])
    print(planets_mean)
    print(planets_max)
    # plt.scatter(planets_mean)
    # plt.show()




#
# def rearrange_true(current_true, current_ids, new_ids):
#     ys_true = read_csv(current_true)
#     with open(current_ids) as f:
#         c = eval(f.readline())
#     with open(new_ids) as f:
#         n = eval(f.readline())
#
#     new_indices = [c.index(x) for x in n]
#     ys_true = ys_true[new_indices, :]
#     np.savetxt("../optimisation/linux_true.csv", ys_true, delimiter='\t', fmt='%.18f')
#
#
# rearrange_true("../optimisation/ttrue_values.csv", )

bag_pred = "../experiments/bagging_best_feats_cv/RandomForest.csv"
xgb_pred = "../experiments/xgb_outlier/GradientBoosting.csv"
plot_predictions("../optimisation/ttrue_values.csv", xgb_pred, "../optimisation/true_indices_new.csv")
