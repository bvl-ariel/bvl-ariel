import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def filter_ccssvv(x):
    csv = x[:x.rfind(".")]
    with open(csv, "w") as f:
        with open(x) as g:
            for i, l in enumerate(g):
                if i % 100 == 0:
                    print(l.strip(), file=f)


# filter_ccssvv("../experiments/submission13_rf_xgb_cnn/predictions_test.csv.ccssvv")


def get_cnn():
    cnn_rf_xgb = "../experiments/submission13_rf_xgb_cnn/predictions_test.csv"
    rf = "../experiments/submission10_bagging_str_500_optimised/predictions_test.csv"
    xgb = "../experiments/submission11_xgb_cluster_optimised_str_optimised_wmae/predictions_test.csv"
    c_r_x = np.genfromtxt(cnn_rf_xgb, delimiter='\t')
    r = np.genfromtxt(rf, delimiter='\t')
    x = np.genfromtxt(xgb, delimiter='\t')
    c = (c_r_x - (0.2 * x + 0.3 * r)) / 0.5
    np.savetxt("../experiments/best_cnn_submission/predictions_test.csv", c, delimiter='\t', fmt='%.18f')


# get_cnn()


def parse_opt(o_file):
    ws = [[] for _ in range(55)]
    es = [[] for _ in range(55)]
    i = -1
    with open(o_file) as f:
        for l in f:
            if l.startswith("Optimising"):
                i += 1
            elif "new best" in l:
                i0 = l.find("[")
                i1 = l.find("]")
                ws[i] = eval(l[i0: i1 + 1])
                es[i] = eval(l[i1 + 2:])
    with open(o_file + ".parsed", "w") as f:
        print(ws, file=f)
        print(es, file=f)


# parse_opt("../optimisation/best_w/opt1.txt")

def best_weights(o_files):
    ws = []
    es = []
    for o in o_files:
        with open(o) as f:
            ws.append(eval(f.readline()))
            es.append(eval(f.readline()))
    best = np.argmin(es, axis=0)
    print([ws[best[i]][i] for i in range(55)])


# best_weights(["../optimisation/best_w/opt1.txt.parsed", "../optimisation/best_w/opt2.txt.parsed"])


def compare(csv1, csv2):
    a = np.genfromtxt(csv1, delimiter='\t')
    b = np.genfromtxt(csv2, delimiter='\t')
    print(np.max(np.abs(a - b)))


def show_clean_vs_ugly(variance_explained):
    file = "../database/csv/basic_mean_median{}.csv"
    ugly = pd.read_csv(file.format(""), sep=',', index_col=0)
    clean = pd.read_csv(file.format("_cleaned_{}".format(variance_explained)), sep=',', index_col=0)
    time_series_colums = ["m{}".format(i) for i in range(1, 301)]
    planet_ids = ugly.index.tolist()
    ugly = np.array(ugly[time_series_colums])
    clean = np.array(clean[time_series_colums])
    for i in range(44, len(planet_ids), 55):
        planet_index = planet_ids[i]
        y_ugly = ugly[i, :]
        y_clean = clean[i, :]
        plt.plot(y_ugly, "r")
        plt.plot(y_clean, "g--")
        plt.title("Planet " + planet_index)
        plt.show()
        plt.clf()
        print()


# compare("../experiments/best_cnn_submission/predictions_test.csv",
#           "../experiments/best_cnn_submission/pred_cnn.csv")
if 0:
    with open("../database/csv/basic_mean_median.csv") as f:
        print(f.readline().strip())
        for x in f:
            if x.startswith("1234_9"):
                print(x.strip())
                break


show_clean_vs_ugly(50)
