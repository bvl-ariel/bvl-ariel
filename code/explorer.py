import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from utils import parse_input, get_planets
from tqdm import tqdm
import pandas as pnd
import seaborn as sns


def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-b*(x-c))) + d


def ramp(x, peak, a, b):
    return np.piecewise(x,
                        [x < a, (a < x) & (x < b), b < x],
                        [0, lambda x: peak*(x-a)/(b-a), peak])


def create_meta_dataset(subset):
    planets = get_planets(subset)
    fcnn_errors = pnd.read_csv('per_planet_wmae.csv', index_col=1)
    bagging_errors = pnd.read_csv('bagging_errors.csv', index_col=0)
    eps = [0.01, 0.005, 0.001, 0.0005]
    models = ",".join("model{}".format(e) for e in eps)
    rows = ['planet,noise,std,star_temp,star_logg,star_logg,star_mass,star_k_magg,period,{}'.format(models)]
    for planet in tqdm(planets):
        with open('completely_aggregated/{}.pickle'.format(planet), 'rb') as f:
            data = pickle.load(f)

        window_size = 1
        denoised = np.zeros((55, 300))
        for j in range(300):
            denoised[:, j] = np.mean(data['matrix'][:, max(0, j - window_size):j + window_size + 1], axis=1)
        maxes = np.mean(np.partition(denoised, -10, axis=1)[:, -10:], axis=1)
        std = np.std(maxes)

        offset = 1
        noises = []
        for c in range(55):
            r = data['matrix'][c]
            matrix = np.corrcoef(r[offset:], r[:-offset])
            assert abs(matrix[0, 1] - matrix[1, 0]) < 10 ** -10
            corr = max(matrix[0, 1], 0)  # 0 or negative --> 0
            noises.append(1 - corr)

        if subset == "train":
            targets = []
            diff = bagging_errors.loc[int(planet), 'baggingError'] - fcnn_errors.loc[int(planet), 'mean_wmae']
            for e in eps:
                if abs(diff) < e:
                    targets.append("any")
                elif diff > 0:
                    targets.append("fcnn")
                else:
                    targets.append("bagging")
        else:
            targets = ["?"] * len(eps)

        row = [planet, str(np.mean(noises)), str(std)]
        row += [str(x) for x in data['misc_inputs']]
        row += targets
        rows.append(','.join(row))

    with open('meta_dataset_{}.csv'.format(subset), 'w') as f:
        for row in rows:
            print(row, file=f)


def stats_per_planet(subset):
    planets = get_planets(subset)
    stats = {}
    folder = 'completely_aggregated'
    window_size = 1
    for planet in tqdm(planets):
        with open('{}/{}.pickle'.format(folder, planet), 'rb') as f:
            data = pickle.load(f)

        denoised = np.zeros((55, 300))
        for j in range(300):
            denoised[:, j] = np.mean(data['matrix'][:, max(0, j - window_size):j + window_size + 1], axis=1)
        maxes = np.mean(np.partition(denoised, -10, axis=1)[:, -10:], axis=1)
        means = np.mean(denoised, axis=1)
        stats[planet] = (np.std(maxes), np.mean(maxes - means), data.get('radii', '?'))

    return stats


def error_per_planet():
    df = pnd.read_csv('CV-report.csv', index_col=0)
    errors = {}
    maxes = np.max(df.values, axis=1)
    means = np.mean(df.values, axis=1)
    for i in range(df.shape[0]):
        errors['{:0>4}'.format(df.index[i])] = df.values[i]

    return errors


def std_error_correlation():
    stats = stats_per_planet('train')
    errors = error_per_planet()
    planets = list(stats.keys())

    mae = []
    maxe = []
    std_max = []
    mean_diff = []
    for p in planets:
        maxe.append(np.max(errors[p]*stats[p][2]))
        mae.append(np.mean(errors[p]*stats[p][2]))
        std_max.append(stats[p][0])
        mean_diff.append(stats[p][1])

    # df = pnd.DataFrame({'planet': planets, 'mean_wmae': mae, 'max_wmae': maxe})
    # df.to_csv('per_planet_wmae.csv')

    xdata = [std_max, mean_diff]
    xlabels = ['std(max)', 'mean(max - mean)']
    ydata = [maxe, mae]
    ylabels = ['max(ae)', 'mean(ae)']
    xthresh = [0.006, 0.035]
    ythresh = [0.0075, 0.002]
    fig, ax = plt.subplots(nrows=2, ncols=2)
    for row in range(2):
        for col in range(2):
            ax[row, col].scatter(xdata[col], ydata[row])
            ax[row, col].set_xlabel(xlabels[col])
            ax[row, col].set_ylabel(ylabels[row])

            for i in range(len(planets)):
                if xdata[col][i] > xthresh[col] or ydata[row][i] > ythresh[row]:
                    ax[row, col].annotate(planets[i], xy=(xdata[col][i], ydata[row][i]))

    plt.show()


def explore_planet(planet, channel):
    planet_matrix = np.zeros((10, 10, 55, 300), dtype=np.float64)
    for spot in range(1, 11):
        for gaus in range(1, 11):
            data = parse_input('../database/noisy_train/{}_{:0>2}_{:0>2}.txt'.format(planet, spot, gaus))
            planet_matrix[gaus-1, spot-1] = data['matrix']

    planet_matrix = 1 - planet_matrix[:, :, channel]

    gaus_agg = np.median(planet_matrix, axis=0)
    spot_agg = np.median(gaus_agg, axis=0)

    window_size = 1
    denoised = np.zeros(300)
    for i in range(300):
        denoised[i] = np.mean(spot_agg[max(0, i - window_size):i + window_size])

    ydata = np.concatenate(np.concatenate(planet_matrix))
    xdata = [i/300 for i in range(150)] + [i/300 for i in range(149, -1, -1)]
    popt_sigmoid, _ = opt.curve_fit(sigmoid, xdata * 100, ydata, p0=(0.1, 1, 0.3, 0), method='trf')
    print(popt_sigmoid)

    popt_ramp, _ = opt.curve_fit(ramp, xdata * 100, ydata, p0=(0.1, 0.3, 0.4), method='trf', bounds=(0, (1, 0.5, 0.5)))
    print(popt_ramp)

    reflected_xdata = [i/300 for i in range(300)]

    my_max = np.max(denoised)
    petkomax = np.mean(np.partition(spot_agg, -10)[-10:])
    comb_max = np.mean(np.partition(denoised, -10)[-10:])

    fig, ax = plt.subplots(nrows=2, ncols=2)
    for spot in range(10):
        for gaus in range(10):
            ax[0, 0].plot(reflected_xdata, planet_matrix[gaus, spot], 'b', alpha=0.1)
    ax[0, 0].plot(reflected_xdata, spot_agg, 'r--')
    ax[0, 1].plot(reflected_xdata, spot_agg, 'r--')
    ax[0, 1].plot(reflected_xdata, denoised, 'b')
    ax[0, 1].plot(reflected_xdata, [sigmoid(x, *popt_sigmoid) for x in xdata], 'g')
    ax[0, 1].plot(reflected_xdata, [ramp(x, *popt_ramp) for x in xdata], 'm')
    ax[1, 0].plot(reflected_xdata, spot_agg, 'r--')
    ax[1, 0].plot(reflected_xdata, [my_max]*300, 'm')
    ax[1, 0].plot(reflected_xdata, [petkomax]*300, 'b')
    ax[1, 0].plot(reflected_xdata, [comb_max]*300, 'g')

    plt.show()
    # plt.savefig('images2/{}_{}.png'.format(planet, channel))


def visualize_cv_report():
    df = pnd.read_csv('CV-report.csv', index_col=0)
    sns.heatmap(df.values)
    plt.show()


if __name__ == '__main__':
    # explore_planet(sys.argv[1], int(sys.argv[2]))
    # visualize_cv_report()
    # std_error_correlation()
    create_meta_dataset('test')
