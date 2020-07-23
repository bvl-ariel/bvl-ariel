try:
    import seaborn as sns
except ImportError:
    print("Could not import seaborn")
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Could not import matplotlib.pyplot")
import numpy as np
import random
import pickle

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm ..")
from data_evaluation import weird_mean_absolute_error
import os
import pandas as pd

random.seed(1)
IMAGES = "../images"


def get_planets(subset):
    planets = set()
    with open('../database/noisy_{}.txt'.format(subset)) as f:
        for line in f:
            planets.add(line.strip().split('/')[1].split('_')[0])

    return planets


def parse_output(path, join_misc_params=False):
    with open(path) as f:
        sma = float(f.readline().strip().split()[-1])
        incl = float(f.readline().strip().split()[-1])
        radii = [float(x) for x in f.readline().strip().split()]

    if not join_misc_params:
        return {
            'sma': sma,
            'incl': incl,
            'radii': np.array(radii, dtype=np.float64)
        }
    else:
        return {
            'misc_outputs': np.array([sma, incl]),
            'misc_outputs_names': ["sma", "incl"],
            'radii': np.array(radii, dtype=np.float64)
        }


def parse_input(path, load_time_series=True):
    with open(path) as f:
        star_temp = float(f.readline().strip().split()[-1])
        star_logg = float(f.readline().strip().split()[-1])
        star_rad = float(f.readline().strip().split()[-1])
        star_mass = float(f.readline().strip().split()[-1])
        star_k_mag = float(f.readline().strip().split()[-1])
        period = float(f.readline().strip().split()[-1])

        matrix = []
        if load_time_series:
            for _ in range(55):
                try:
                    row = [float(x) for x in f.readline().strip().split()]
                    matrix.append(row)
                except ValueError:
                    raise Exception('Check file: {}'.format(path))

    return {
        'misc_inputs': np.array([star_temp, star_logg, star_rad, star_mass, star_k_mag, period], dtype=np.float64),
        'misc_inputs_names': ["star_temp", "star_logg", "star_rad", "star_mass", "star_k_mag", "period"],
        'matrix': np.array(matrix),
    }


def heatmap_pickled_planets():
    train_planets = list(get_planets('train'))
    for planet in tqdm(train_planets):
        with open('completely_aggregated/{}.pickle'.format(planet), 'rb') as f:
            data = pickle.load(f)
        planet_matrix = data['matrix']
        sns.heatmap(planet_matrix)
        plt.savefig(os.path.join(IMAGES, '{}_heat.png'.format(planet)), dpi=300)
        plt.close()


def heatmap_all_planets():
    weird = ["0855", "1097", "0611", "1448", "1382", "1692", "1303", "1345", "2071", "1352", "1900", "1749", "0985",
             "1333", "1349", "0955", "1041", "0734", "1673", "1797", "0972", "0938", "1502"]
    path = '../database/noisy_test/'
    # train_planets = list(get_planets('train'))
    # for planet in train_planets:
    for planet in weird:
        planet_matrix = np.zeros((10, 55, 300))
        for spot in range(1, 11):
            spot_matrix = np.zeros((55, 300))
            for gaus in range(1, 11):
                data = parse_input('{}/{}_{:0>2}_{:0>2}.txt'.format(path, planet, spot, gaus))
                spot_matrix += data['matrix']

            spot_matrix /= 10
            planet_matrix[spot - 1] = spot_matrix
        planet_matrix = np.median(planet_matrix, axis=0)
        sns.heatmap(planet_matrix)
        plt.savefig(os.path.join(IMAGES, '{}_heat.png'.format(planet)), dpi=300)
        plt.close()


def plot_hot_candidates():
    weird = ["0855", "1097", "0611", "1448", "1382", "1692", "1303", "1345", "2071", "1352", "1900", "1749", "0985",
             "1333", "1349", "0955", "1041", "0734", "1673", "1797", "0972", "0938", "1502", "1583"][-1:]
    path = '../database/noisy_test/'
    # train_planets = list(get_planets('train'))
    # for planet in train_planets:
    for planet in weird:
        planet_matrix = np.zeros((10, 55, 300))
        for spot in range(1, 11):
            spot_matrix = np.zeros((55, 300))
            for gaus in range(1, 11):
                data = parse_input('{}/{}_{:0>2}_{:0>2}.txt'.format(path, planet, spot, gaus))
                spot_matrix += data['matrix']

            spot_matrix /= 10
            planet_matrix[spot - 1] = spot_matrix
        planet_matrix = np.median(planet_matrix, axis=0)
        fig, axs = plt.subplots(11, 5)
        fig.set_size_inches(20, 50)
        for i in range(11):
            for j in range(5):
                axs[i, j].plot(planet_matrix[5 * i + j, :])
                axs[i, j].set_title("Planet {} Channel {}".format(planet, 5 * i + j))

        # plt.show()
        os.makedirs("../images/hot", exist_ok=True)
        plt.savefig("../images/hot/planet{}.pdf".format(planet))
        plt.clf()
        plt.close()


def clus_eval():
    predictions = np.zeros((2097, 55))
    reals = np.zeros((2097, 55))
    with open('clus/settings.ens.xval.preds') as f:
        in_data = False
        for line in f:
            if line.strip().lower() == '@data':
                in_data = True
            elif line.startswith('% Target'):
                continue
            elif in_data:
                words = line.strip().split(',')
                if len(words) > 3:
                    planet = int(words[0])
                    #     reals[planet-1] = np.array([float(x) for x in words[1:56]])
                    #     predictions[planet-1] = np.array([float(x) for x in words[56:111]])
                    channel = int(words[1])
                    reals[planet - 1][channel] = float(words[2])
                    predictions[planet - 1][channel] = float(words[3])

    mse = weird_mean_absolute_error(np.array(predictions), np.array(reals))
    print(mse)


def radius_profiles():
    path = '../database/params_train/'
    if not os.path.exists(IMAGES):
        os.makedirs(IMAGES)
    planets_all = sorted(get_planets("train"))
    chosen = random.sample(range(len(planets_all)), 0)
    planets = {planets_all[i] for i in chosen} | {'1139', '0222'}  # make sure that the ugly ones are present
    for planet in planets:
        radii = parse_output(os.path.join(path, "{}_01_01.txt".format(planet)))["radii"]
        # normalise to the range [0, 1]
        # radii -= min(radii)
        # radii /= max(radii)
        plt.plot(range(len(radii)), radii, '-o')
        plt.savefig(os.path.join(IMAGES, '{}_radii.png'.format(planet)), dpi=300)
        plt.close()


def star_parameters_histogram():
    path = '../database/noisy_train/'
    planets = sorted(get_planets("train"))
    parameters = []
    names = []
    for planet in planets:
        data = parse_input(os.path.join(path, "{}_01_01.txt".format(planet)), False)
        parameters.append(data["misc_inputs"])
        if not names:
            names = data["misc_inputs_names"]
    parameters = np.array(parameters)

    n_params = len(parameters[0])
    n_bins = 100
    fig, axs = plt.subplots(n_params, 1, tight_layout=True, figsize=(5, 10))
    for i in range(n_params):
        axs[i].hist(parameters[:, i], bins=n_bins)
        axs[i].title.set_text(names[i])
    plt.savefig(os.path.join(IMAGES, 'histograms.png'))
    plt.close()


def correlations_with_target():
    data, i_target, feature_names = load_data(True, False)
    out_dir = "../images/feature_vs_target"
    os.makedirs(out_dir, exist_ok=True)
    ys = data[:, i_target]
    jet = plt.get_cmap('jet')
    for i, feature in enumerate(feature_names):
        if i != i_target:
            xs = data[:, i]
            if feature in ["max_shadow", "star_temp"]:
                for j in range(55):
                    ending = ".png"
                    # _ = plt.subplots(1, 1, tight_layout=True, figsize=(15, 15))
                    # colors = [j % 55 for j in range(len(xs))]
                    plt.scatter(xs[j::55], ys[j::55])  # , c=colors)
                    plt.title(feature)
                    plt.savefig(os.path.join(out_dir, feature + "_{}".format(j) + ending))
                    plt.close()
            else:
                continue
                # plt.plot(xs, ys, '.')
                # ending = ".png"
                plt.title(feature)
                plt.savefig(os.path.join(out_dir, feature + ending))
                plt.close()


def show_last_lines(file_name, n):
    with open(file_name) as f:
        lines = 0
        for _ in f:
            lines += 1
    with open(file_name) as f:
        for i in range(lines - n):
            f.readline()
        for l in f:
            print(l.strip())


def target_correlations(save_targets=None):
    data = load_targets("../database/csv/basic_mean_median.csv")
    # if save_targets is not None:
    #     data.to_csv(os.path.join(save_targets), index=False)

    corr = data.corr()
    if save_targets is not None:
        corr.to_csv(os.path.join(save_targets), index=False)
    plt.show()
    d = np.array(corr)
    c = np.min(d)
    m, n = d.shape
    assert m == n
    for i in range(n):
        values = [(v, j) for j, v in enumerate(d[i, :]) if j != i]
        values.sort(reverse=True)
        print(i, values)
    sns.heatmap(
        corr,
        vmin=c, vmax=1, center=(1 + c) / 2,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    plt.show()


def compare(pred1, pred2):
    a = np.genfromtxt(pred1, delimiter='\t')
    b = np.genfromtxt(pred2, delimiter='\t')
    c = np.max(np.abs(a - b))
    d = np.mean(np.abs(a - b))
    print("max and mean abs diff:", c, d)


def load_data(is_test, is_cv, path=None, is_denoised=False, classic_fresh="fresh"):
    if classic_fresh == "classic":
        print("USing old features")
        return load_data_old(is_test, is_cv, path, is_denoised)
    elif classic_fresh == "fresh":
        print("Using extended")
        return load_data_fresh(is_test, is_cv, path, is_denoised)
    else:
        raise ValueError("Wrong!")


def save_time_series(is_test):
    d = pd.read_csv("../database/csv/{}.csv".format("test2" if is_test else "basic_mean_median"), sep=',',
                    index_col=0)
    time_series_cols = ["m{}".format(i) for i in range(1, 301)]
    d = d[time_series_cols]
    matrix = np.array(d)
    for c in range(55):
        np.savetxt("../database/csv/channel_csv/channel{}.csv".format(c), matrix[c::55, :], delimiter=',', fmt='%.18f')


def convert_stacked_to_mtr(is_test, is_cv, path=None, denoised=False):
    """
    Raw star features are the same, radiation features are just permuted, so we concatenate only the extended features.
    :param is_test:
    :param is_cv:
    :param path:
    :param denoised:
    :return:
    """
    data, target_index, instance_keys, column_names = load_data(is_test, is_cv, path, denoised)
    constant_per_channel_features = "star_temp,star_logg,star_rad,star_mass,star_k_mag,period".split(',')
    constant_per_channel_features += ["radiation{}".format(i) for i in range(1, 56)]
    constant_per_channel_features += ["max_shadow1_{}".format(channel) for channel in range(55)]

    constant_locations = [column_names.index(f) for f in constant_per_channel_features]
    other_locations = sorted(set(range(len(column_names))) - set(constant_locations + [target_index]))
    descriptive = []
    target = []
    assert len(instance_keys) % 55 == 0
    n = len(instance_keys) // 55
    for instance in range(n):
        for channel in range(55):
            i = instance * 55 + channel
            if channel == 0:
                descriptive.append(list(data[i, constant_locations]))
                target.append([])
            descriptive[-1] += list(data[i, other_locations])
            target[-1].append(data[i, target_index])
    c_features = len(constant_locations)
    o_features = len(other_locations) * 55
    first_target_index = c_features + o_features
    return np.array([d + t for d, t in zip(descriptive, target)]), c_features, o_features, first_target_index


def load_targets(file):
    channels = 55
    data = [[] for _ in range(channels)]
    with open(file) as f:
        a = f.readline().strip().split(',')
        r_index = a.index("r")
        for i, l in enumerate(f):
            r = l.strip().split(',')[r_index]
            if r:
                data[i % channels].append(float(r))
    return pd.DataFrame({"r{}".format(i): data[i] for i in range(channels)})


def join_horizontally(files, out_file, separator="\t"):
    matrix = []
    for file in files:
        with open(file) as f:
            lines = [l.strip() for l in f.readlines()]
        if not matrix:
            matrix = lines
        else:
            assert len(matrix) == len(lines)
            for i in range(len(matrix)):
                matrix[i] += separator + lines[i]
    with open(out_file, "w") as f:
        for l in matrix:
            print(l, file=f)


def load_data_old(is_test, is_cv, path=None, is_denoised=False):
    if is_test:
        print("WARNING: avg & max not supported for testing")
    if path is None:
        in_dir = "../database/csv"
    else:
        in_dir = path
    sources = ["test2.csv", "rad_test.csv"] if is_test else ["basic_mean_median{}.csv",
                                                             "radiations.csv"]
    if not is_test:
        sources[0] = sources[0].format("_cleaned_120") if is_denoised else sources[0].format("")
    csv_files = [os.path.join(in_dir, csv) for csv in sources]
    d1 = pd.read_csv(csv_files[0], sep=',', index_col=0)
    # compute some features on the fly
    max_shadow_columns1 = ["max_shadow1_{}".format(channel) for channel in range(55)]
    max_values1 = {c: [] for c in max_shadow_columns1}

    # max_shadow_columns2 = ["max_shadow2_{}".format(channel) for channel in range(55)]
    max_values2 = {}  # c: [] for c in max_shadow_columns2}
    #
    # max_shadow_columns3 = ["max_shadow3_{}".format(channel) for channel in range(55)]
    max_values3 = {}  # c: [] for c in max_shadow_columns3}

    time_series_cols = ["m{}".format(i) for i in range(1, 301)]
    for i, (_1, row) in enumerate(d1.iterrows()):
        column1 = max_shadow_columns1[i % 55]
        # column2 = max_shadow_columns2[i % 55]
        # column3 = max_shadow_columns3[i % 55]
        ts = sorted(list(row[time_series_cols]))
        median1 = np.median(ts[-15:])
        # median2 = np.median(ts[-20:])
        # median3 = np.median(ts[-5:])

        for _2 in range(55):
            max_values1[column1].append(median1)
            # max_values2[column2].append(median2)
            # max_values3[column3].append(median3)
    max_values = {**max_values1, **max_values2, **max_values3}
    max_shadow_columns = max_shadow_columns1  # + max_shadow_columns2 + max_shadow_columns3
    d3 = pd.DataFrame(max_values, columns=max_shadow_columns, index=d1.index.tolist())

    # radiations_columns = ["radiation{}".format(i) for i in range(1, 56)]
    d1.drop(time_series_cols + ['sma', 'incl'], axis=1, inplace=True)
    d2 = pd.read_csv(csv_files[1], sep=',', index_col=0, float_precision="high")  # * 10 ** 8
    if False and not is_test:
        d4 = pd.read_csv(csv_files[3], sep=',', index_col=0, float_precision="high")
        d5 = pd.read_csv(csv_files[2], sep=',', index_col=0, float_precision="high")
        d5.drop(["middleMedian", "middleParabola"], axis=1, inplace=True)
        d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    else:
        d = pd.concat([d1, d2, d3], axis=1)
    if is_cv:
        d = d.dropna()  # remove test set planets
    target_index = d.columns.get_loc("r")
    data = np.array(d)
    # normalize
    for i in range(data.shape[1]):
        if i != target_index:
            data[:, i] /= np.max(data[:, i])
    return data, target_index, d.index, list(d.columns)


def load_data_fresh(is_test, is_cv, path=None, is_denoised=False):
    if is_test:
        print("WARNING: avg & max not supported for testing")
    if path is None:
        in_dir = "../database/csv"
    else:
        in_dir = path
    sources = ["test2.csv", "rad_test.csv"] if is_test else ["basic_mean_median{}.csv",
                                                             "radiations_extended.csv",  #
                                                             "new_maxima_and_stuff.csv",
                                                             "avg_max.csv"]
    if not is_test:
        sources[0] = sources[0].format("_cleaned_120") if is_denoised else sources[0].format("")
    csv_files = [os.path.join(in_dir, csv) for csv in sources]
    d1 = pd.read_csv(csv_files[0], sep=',', index_col=0)
    # compute some features on the fly
    max_shadow_columns1 = ["max_shadow1_{}".format(channel) for channel in range(55)]
    max_values1 = {c: [] for c in max_shadow_columns1}

    # max_shadow_columns2 = ["max_shadow2_{}".format(channel) for channel in range(55)]
    max_values2 = {}  # c: [] for c in max_shadow_columns2}
    #
    # max_shadow_columns3 = ["max_shadow3_{}".format(channel) for channel in range(55)]
    max_values3 = {}  # c: [] for c in max_shadow_columns3}

    time_series_cols = ["m{}".format(i) for i in range(1, 301)]
    for i, (_1, row) in enumerate(d1.iterrows()):
        column1 = max_shadow_columns1[i % 55]
        # column2 = max_shadow_columns2[i % 55]
        # column3 = max_shadow_columns3[i % 55]
        ts = sorted(list(row[time_series_cols]))
        median1 = np.median(ts[-15:])
        # median2 = np.median(ts[-20:])
        # median3 = np.median(ts[-5:])

        for _2 in range(55):
            max_values1[column1].append(median1)
            # max_values2[column2].append(median2)
            # max_values3[column3].append(median3)
    max_values = {**max_values1, **max_values2, **max_values3}
    max_shadow_columns = max_shadow_columns1  # + max_shadow_columns2 + max_shadow_columns3
    d3 = pd.DataFrame(max_values, columns=max_shadow_columns, index=d1.index.tolist())

    # radiations_columns = ["radiation{}".format(i) for i in range(1, 56)]
    d1.drop(time_series_cols + ['sma', 'incl'], axis=1, inplace=True)
    d2 = pd.read_csv(csv_files[1], sep=',', index_col=0, float_precision="high")  # * 10 ** 8
    if not is_test:
        d4 = pd.read_csv(csv_files[3], sep=',', index_col=0, float_precision="high")
        d5 = pd.read_csv(csv_files[2], sep=',', index_col=0, float_precision="high")
        d5.drop(["middleMedian", "middleParabola"], axis=1, inplace=True)
        d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    else:
        d = pd.concat([d1, d2, d3], axis=1)
    if is_cv:
        d = d.dropna()  # remove test set planets
    target_index = d.columns.get_loc("r")
    data = np.array(d)
    # normalize
    for i in range(data.shape[1]):
        if i != target_index:
            data[:, i] /= np.max(data[:, i])
    return data, target_index, d.index, list(d.columns)


if __name__ == '__main__':
    # heatmap_all_planets()
    # clus_eval()

    # maxes = []
    # for planet in tqdm(get_planets('train')):
    #     with open('completely_aggregated/{}.pickle'.format(planet), 'rb') as f:
    #         data = pickle.load(f)
    #     maxes.append(np.max(data['matrix']))
    #
    # plt.hist(maxes)
    # plt.show()

    # for planet in tqdm(get_planets('train')):
    #     last_output = None
    #     for spot in range(1, 11):
    #         for gaus in range(1, 11):
    #             output = parse_output('../database/params_train/{}_{:0>2}_{:0>2}.txt'.format(planet, spot, gaus))
    #             if last_output is None:
    #                 last_output = output['radii']
    #             if np.linalg.norm(last_output - output['radii']) > 0:
    #                 raise Exception('Fear! Fire! Foes! On planet {}'.format(planet))

    # compare("../experiments/bagging_str_250/predictions_test.csv.ccssvv",
    #         "../experiments/bagging_stacked/predictions_test.csv.ccssvv")
    # target_correlations(None)
    join_horizontally(
        ["../experiments/bagging_of_200boosting/predictions_test{}.csv".format(t) for t in [0, 11, 22, 33, 44]],
        "../experiments/bagging_of_200boosting/predictions_test.csv")
    # save_time_series(False)
    heatmap_all_planets()
    # plot_hot_candidates()
