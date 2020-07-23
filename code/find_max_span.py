from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from feature_construction import NOISE_ORDER


def join_csv():
    with open("../database/csv/new_maxima_and_stuff.csv", "w") as f:
        for i in range(5):
            with open("../database/csv/new_maxima_and_stuff{}.csv".format(i)) as g:
                ls = [l.strip() for l in g.readlines()]
            print("\n".join(ls), file=f)


def error(ys_true: np.ndarray, xs, is_mse):
    """
    :param ys_true: [y0, ... , y299]
    :param xs: [x0, x1, x2, x3], indices, 0 <= x0 < x1 < x2 < x3 <= 299
    :param is_mse: If False, mae
    :return:
    """
    def aggregate(ys) -> float:
        if is_mse:
            return np.mean(ys)
        else:
            return np.median(ys)

    indices0 = list(range(xs[0] + 1)) + list(range(xs[3], 300))
    indices1 = list(range(xs[1], xs[2] + 1))
    # both constant parts
    ys0_true = ys_true[indices0]
    ys1_true = ys_true[indices1]
    y0 = aggregate(ys0_true)
    y1 = aggregate(ys1_true)

    ys01 = np.linspace(y0, y1, xs[1] - xs[0] + 1)
    ys10 = np.linspace(y1, y0, xs[3] - xs[2] + 1)

    diff0 = ys0_true - y0
    diff1 = ys1_true - y1

    diff01 = ys_true[xs[0]: xs[1] + 1] - ys01
    diff10 = ys_true[xs[2]: xs[3] + 1] - ys10

    concatenated = np.concatenate((diff0, diff1, diff01, diff10))
    if is_mse:
        return np.square(concatenated).sum()
    else:
        return np.abs(concatenated).sum()


def optimize(ys, xs0, is_mse):
    def reflect(left_xs):
        return [len(ys) - 1 - left_x for left_x in left_xs[::-1]]
    opt_config = None
    opt_error = float("inf")
    changed_something = True
    xs = xs0[:]
    dx = 5
    while changed_something:
        changed_something = False
        for i in range(len(xs)):
            x = xs[i]
            opt_x = x
            lower = 0 if i == 0 else xs[i - 1] + 1
            upper = len(ys) // 2 if i == len(xs) - 1 else xs[i + 1]
            for j in range(max(lower, x - dx), min(upper, x + dx)):
                xs[i] = j
                e = error(ys, xs + reflect(xs), is_mse)
                if e < opt_error:
                    opt_config = xs[:]
                    opt_error = e
                    opt_x = x
                    if j != x:
                        changed_something = True
            xs[i] = opt_x
    return opt_config, opt_error


def create(csv, i0):
    def reflect(left_xs):
        return [300 - 1 - left_x for left_x in left_xs[::-1]]

    def generate_x0():
        # for j in range(m):
        #     yield sorted(random.sample(range(ys.shape[1] // 2), 2)) if j > 0 else [100, 120]
        for x0 in range(50, 120, 5):
            for x1 in range(x0 + 1, 140, 4):
                yield [x0, x1]

    data = pd.read_csv(csv, sep=',', index_col=0)
    time_series_colums = ["m{}".format(i) for i in range(1, 301)]
    planet_ids = data.index.tolist()
    data = data[time_series_colums]
    ys = np.array(data)
    random.seed(132)
    header = "ID,middleMedian,middleParabola,x0x1,x1x2"
    matrix = []
    new_csv = open("../database/csv/new_maxima_and_stuff{}.csv".format(i0), "w")

    n_instances = ys.shape[0]
    assert n_instances % 5 == 0
    q = n_instances // 5
    for i in tqdm(list(range(q * i0, q * (i0 + 1)))):
        planet_index = planet_ids[i]
        y_now = ys[i, :]
        xs_opt = None
        e_opt = float("inf")
        for xs0 in generate_x0():
            xs = xs0 + reflect(xs0)
            e = error(y_now, xs, False)
            # , e = optimize(ys[i, :], xs0, is_mse)
            if e < e_opt:
                # print("--->", xs, e)
                e_opt = e
                xs_opt = xs[:]
        # print(xs_opt)
        # plt.plot(xs_opt, [0] * 4, "ro")
        # plt.plot(y_now, '.')
        median = np.median(y_now[xs_opt[1]: xs_opt[2] + 1])
        # plt.plot([0, y_now.shape[0]], [median, median], 'r--')

        # parabola
        # x_middle = list(range(xs_opt[1], xs_opt[2] + 1))
        # y_middle = y_now[xs_opt[1]: xs_opt[2] + 1]
        p = np.polyfit(list(range(xs_opt[1], xs_opt[2] + 1)), y_now[xs_opt[1]: xs_opt[2] + 1], 2)
        x0 = -p[1] / (2 * p[0])
        # x_p = np.linspace(xs_opt[1], xs_opt[2], 40)
        y0 = np.polyval(p, x0)
        # plt.plot(x_p, y0, 'g--')
        # plt.title("{} {}".format(xs_opt, e_opt))
        # plt.show()
        # plt.clf()
        # print()
        matrix.append([str(t) for t in [planet_index, median, y0, xs_opt[1] - xs_opt[0], xs_opt[2] - xs_opt[1]]])
    if i0 == 0:
        print(header, file=new_csv)
    for line in matrix:
        print(",".join(line), file=new_csv)
    new_csv.close()




# import sys
# part = int(sys.argv[1])

# create("../database/csv/basic_mean_median.csv", part)
# join_csv()