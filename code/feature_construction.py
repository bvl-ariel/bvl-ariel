import numpy as np
from utils import get_planets, parse_input, parse_output
import pickle
from tqdm import tqdm
import os
import pandas as pd
import re
import scipy.optimize as opt
# from numba import njit
try:
    import tsfresh
except ImportError:
    print("tsfresh")

C = 299792458  # speed of light
H = 6.62607015 * 10 ** -34  # Planck constant
K = 1.380649 * 10 ** -23  # Boltzmann constant

# [blue filter, red filter, NIR1 filter, NIR2 spectrograph, IR spectrograph]
WAVELENGTHS_RANGES = 10 ** -6 * np.array([[0.5, 0.55], [0.8, 1.0], [1.05, 1.2], [1.25, 1.95], [1.95, 7.8]])
FREQUENCY_RANGES = C / WAVELENGTHS_RANGES

NOISE_ORDER = [2, 1, 6, 5, 4, 7, 3, 8, 0, 44, 12, 11, 14, 15, 13, 45, 16, 9, 18, 17, 19, 10, 46, 47, 20, 23, 22, 24, 21,
           28, 25, 26, 27, 29, 48, 30, 49, 32, 50, 31, 33, 34, 35, 51, 36, 38, 37, 39, 40, 52, 41, 53, 42, 54, 43]


def calculate_tsfresh_features():
    planets = get_planets('train') | get_planets('test')
    folder = 'completely_aggregated'

    values = {'planet_id': [], 'timestep': []}
    for c in range(55):
        values['channel_{}'.format(c+1)] = []

    for planet in tqdm(planets):
        with open('{}/{}.pickle'.format(folder, planet), 'rb') as f:
            data = pickle.load(f)
        for t in range(300):
            values['planet_id'].append(planet)
            values['timestep'].append(t)
            for c in range(55):
                values['channel_{}'.format(c+1)].append(data['matrix'][c, t])

    df = pd.DataFrame(values)
    features = tsfresh.extract_features(df, column_id='planet_id', column_sort='timestep', n_jobs=4)
    features.to_pickle('tsfresh_all.pki')


def aggregated_examples_with_tsfresh():
    tsfresh_df = pd.read_pickle('tsfresh_all.pki')
    train_planets = get_planets('train')
    test_planets = get_planets('test')

    train_df = tsfresh_df[tsfresh_df.index.isin(train_planets)]
    singleton_columns = [c for c in train_df.columns if len(set(train_df[c])) == 1]

    tsfresh_df.dropna(axis=1, inplace=True)
    tsfresh_df.drop(singleton_columns, axis=1, inplace=True)

    os.makedirs('aggregated_tsfresh', exist_ok=True)
    for planet in tqdm(train_planets | test_planets):
        data = pickle.load(open('completely_aggregated/{}.pickle'.format(planet), 'rb'))
        data['tsfresh'] = tsfresh_df.loc[[planet]].values[0]
        pickle.dump(data, open('aggregated_tsfresh/{}.pickle'.format(planet), 'wb'))


def examples_with_aggregated_matrices(subset):
    planets = get_planets(subset)
    os.makedirs('../database/completely_aggregated', exist_ok=True)
    for planet in tqdm(planets):
        data = None
        planet_matrix = np.zeros((10, 55, 300), dtype=np.float64)
        for spot in range(1, 11):
            spot_matrix = np.zeros((55, 300), dtype=np.float64)
            for gaus in range(1, 11):
                data = parse_input('../database/noisy_{}/{}_{:0>2}_{:0>2}.txt'.format(subset, planet, spot, gaus))
                spot_matrix += data['matrix']

            spot_matrix /= 10
            planet_matrix[spot - 1] = spot_matrix

        planet_matrix = 1 - np.median(planet_matrix, axis=0)
        data['matrix'] = planet_matrix

        window_size = 3
        temp = np.zeros((55, 300))
        for i in range(300):
            temp[:, i] = np.mean(planet_matrix[:, max(0, i-window_size):i+window_size], axis=1)
        data['maxes'] = np.max(temp, axis=1)
        data['relative_means'] = np.mean(temp, axis=1) / data['maxes']

        frequencies = create_ariel_frequencies(equidistant_frequency=True)
        radiations = [black_body_radiation(frequencies[i], data['misc_inputs'][0]) for i in NOISE_ORDER]
        data['radiation'] = np.array(radiations)

        out_path = '../database/params_train/{}_01_01.txt'.format(planet)
        if os.path.exists(out_path):
            data.update(parse_output(out_path))
        data['planet'] = planet

        pickle.dump(data, open('../database/completely_aggregated/{}.pickle'.format(planet), 'wb'))


def examples_with_custom_features(subset):

    # @njit
    def sigmoid(x, a, b, c, d):
        return a / (1 + np.exp(-b * (x - c))) + d

    def ramp(x, peak, a, b):
        return np.piecewise(x, [x < a, (a < x) & (x < b), b < x], [0, lambda x: peak * (x - a) / (b - a), peak])

    planets = get_planets(subset)
    os.makedirs('custom', exist_ok=True)
    for planet in tqdm(planets):
        planet_matrix = np.zeros((10, 10, 55, 300), dtype=np.float64)
        data = {'planet': planet}
        for spot in range(1, 11):
            for gaus in range(1, 11):
                input = parse_input('../database/noisy_{}/{}_{:0>2}_{:0>2}.txt'.format(subset, planet, spot, gaus))
                data.update(input)
                planet_matrix[gaus - 1, spot - 1] = 1 - input['matrix']

        del data['matrix']

        frequencies = create_ariel_frequencies(equidistant_frequency=True)
        radiations = [black_body_radiation(frequencies[i], data['misc_inputs'][0]) for i in NOISE_ORDER]
        data['radiation'] = np.array(radiations)

        gaus_agg = np.median(planet_matrix, axis=0)
        spot_agg = np.median(gaus_agg, axis=0)
        raw_curves = np.zeros((55, 3))
        agg_curves = np.zeros((55, 3))

        for channel in range(55):
            xdata = [i / 300 for i in range(150)] + [i/300 for i in range(149, -1, -1)]

            ydata_agg = spot_agg[channel]
            try:
                # popt_agg, _ = opt.curve_fit(sigmoid, xdata, ydata_agg, p0=(0, 1, 0.3, 0), maxfev=2*10**3)
                popt_agg, _ = opt.curve_fit(ramp, xdata, ydata_agg, p0=(0, 0.2, 0.4), maxfev=2*10**3)
            except RuntimeError:
                print('Failed agg for planet {} channel {}'.format(planet, channel))
                popt_agg = (0, 0, 0, 0)
            agg_curves[channel] = popt_agg

            ydata_raw = np.concatenate(np.concatenate(planet_matrix[:, :, channel, :]))
            try:
                # popt_raw, _ = opt.curve_fit(sigmoid, xdata*100, ydata_raw, p0=(0, 1, 0.3, 0), maxfev=2*10**3)
                popt_raw, _ = opt.curve_fit(ramp, xdata*100, ydata_raw, p0=(0, 0.2, 0.4), maxfev=2*10**3)
            except RuntimeError:
                print('Failed raw for planet {} channel {}'.format(planet, channel))
                popt_raw = (0, 0, 0, 0)
            raw_curves[channel] = popt_raw

        data['raw_curves'] = raw_curves
        data['agg_curves'] = agg_curves

        out_path = '../database/params_train/{}_01_01.txt'.format(planet)
        if os.path.exists(out_path):
            data.update(parse_output(out_path))

        pickle.dump(data, open('custom/{}.pickle'.format(planet), 'wb'))


def examples_with_aggregated_gauss(subset):
    planets = get_planets(subset)
    os.makedirs('gauss_aggregated', exist_ok=True)
    for planet in tqdm(planets):
        pickle_path = 'gauss_aggregated/{}.pickle'.format(planet)
        if not os.path.exists(pickle_path):
            data = None
            planet_matrix = np.zeros((10, 55, 300), dtype=np.float32)
            for spot in range(1, 11):
                spot_matrix = np.zeros((55, 300), dtype=np.float32)
                for gaus in range(1, 11):
                    data = parse_input('../database/noisy_{}/{}_{:0>2}_{:0>2}.txt'.format(subset, planet, spot, gaus))
                    spot_matrix += data['matrix']

                spot_matrix /= 10
                planet_matrix[spot-1] = 1 - spot_matrix

            data['matrix'] = planet_matrix
            out_path = '../database/params_train/{}_01_01.txt'.format(planet)
            if os.path.exists(out_path):
                data.update(parse_output(out_path))
            data['planet'] = planet

            pickle.dump(data, open(pickle_path, 'wb'))


def aggregate_matrices(matrices, aggregator):
    try:
        # tries to apply one of the numpy aggregation functions
        return aggregator(matrices, axis=0)
    except TypeError:
        pass
    n_rows, n_columns = matrices[0].shape
    matrix = np.zeros((n_rows, n_columns))
    for i in range(n_rows):
        for j in range(n_columns):
            matrix[i, j] = aggregator(matrices[:, i, j])
    return matrix


def create_fully_aggregated_csv(gauss_aggregation, spot_aggregation, file_name):
    """
    Creates a csv file that consists of aggregated values of time series for all planets (train and test).
    The csv format is as follows:
    ID,star_temp,star_logg,star_rad,star_mass,star_k_mag,period,sma,incl,m1,...,m300
    0001_1,...
    ...
    0001_55,...
    ...

    Missing values (values of sma, incl and target values for test set) are represented as empty strings.
    :param gauss_aggregation: function for aggregating the matrices <planet>_<spot>_<gauss
    to a single <planet>_<spot> matrix. Its signature is f(Iterable[float]) -> float.
    :param spot_aggregation: function for aggregating the matrices <planet> matrix,
    Its signature is f(Iterable[float]) -> float.
    :param file_name: the name of the output file (no path, just name, e.g., 'mean_mean.csv').
    :return:
    """

    out_dir = "../database/csv"
    os.makedirs(out_dir, exist_ok=True)
    columns = ["ID,star_temp,star_logg,star_rad,star_mass,star_k_mag,period,sma,incl".split(','),
               ["m{}".format(i) for i in range(1, 301)],
               ["r"]]
    data = {c: [] for cs in columns for c in cs}
    planets = [(planet_type, planet) for planet_type in ["train", "test"]
               for planet in sorted(get_planets(planet_type))]
    # planets = planets[:5] + planets[-5:]
    for planet_type, planet in tqdm(planets):
        planet_matrices = []
        for spot in range(1, 11):
            spot_matrices = []
            for gauss in range(1, 11):
                d = parse_input('../database/noisy_{}/{}_{:0>2}_{:0>2}.txt'.format(planet_type, planet, spot, gauss))
                spot_matrices.append(d['matrix'])
            planet_matrices.append(aggregate_matrices(np.array(spot_matrices), gauss_aggregation))
        # matrix
        planet_matrix = 1 - aggregate_matrices(planet_matrices, spot_aggregation)  # type: np.ndarray
        # additional parameters
        additional_parameters = dict(zip(d['misc_inputs_names'], d['misc_inputs']))
        if planet_type == "train":
            d = parse_output('../database/params_{}/{}_01_01.txt'.format(planet_type, planet), join_misc_params=True)
            additional_parameters.update(dict(zip(d['misc_outputs_names'], d['misc_outputs'])))
            additional_parameters.update({'r': d['radii']})
        for row in range(55):
            planet_id = "{}_{}".format(planet, row + 1)
            data[columns[0][0]].append(planet_id)
            # additional
            for c in columns[0][1:]:
                value = additional_parameters[c] if c in additional_parameters else None
                data[c].append(value)
            # time series
            for c, value in zip(columns[1], planet_matrix[row]):
                data[c].append(value)
            # radius
            r = columns[2][0]
            value = additional_parameters[r][row] if r in additional_parameters else None
            data[columns[2][0]].append(value)
    df = pd.DataFrame(data, columns=[c for cs in columns for c in cs])
    df.to_csv(os.path.join(out_dir, file_name), index=False)


def examples_with_maximums(subset, window_size=3):
    planets = get_planets(subset)
    os.makedirs('engineered', exist_ok=True)
    for planet in tqdm(planets):
        with open('completely_aggregated/{}.pickle'.format(planet), 'rb') as f:
            data = pickle.load(f)

        matrix = data['matrix']
        for i in range(300):
            matrix[:, i] = np.mean(matrix[:, max(0, i - window_size):i + window_size], axis=1)

        data['maxes'] = np.max(matrix, axis=1)
        del data['matrix']

        with open('engineered/{}.pickle'.format(planet), 'wb') as f:
            pickle.dump(data, f)


def histogram_arffs():
    bins = 50
    window_size = 5
    data_type = "test" if False else "train"
    with open('custom_{}.arff'.format(data_type), 'w') as f:
        print('@relation planets', file=f)
        print('@attribute planet string', file=f)
        print('@attribute channel string', file=f)
        print('@attribute sma numeric', file=f)
        print('@attribute incl numeric', file=f)

        print('@attribute radius numeric', file=f)
        print('@attribute radiation numeric', file=f)
        print('@attribute max numeric', file=f)
        print('@attribute avg numeric', file=f)
        for j in range(bins):
            print('@attribute histo_{} numeric'.format(j), file=f)
        # for i in range(55):
        #     print('@attribute radius{} numeric'.format(i+1), file=f)

        # for i in range(55):
        #     print('@attribute radiation{} numeric'.format(i+1), file=f)

        # for i in range(55):
        #     print('@attribute max{} numeric'.format(i+1), file=f)
        #     print('@attribute avg{} numeric'.format(i+1), file=f)

        # for i in range(55):
        #     for j in range(bins):
        #         print('@attribute histo_{}_{} numeric'.format(i+1, j), file=f)

        print('\n@data', file=f)
        for planet in tqdm(get_planets(data_type)):
            with open('../database/completely_aggregated/{}.pickle'.format(planet), 'rb') as g:
                data = pickle.load(g)
            matrix = data['matrix']
            # features = [data['planet'], data['sma'], data['incl']] + list(data['radii']) + list(data['radiation'])

            # for c in range(55):
            #     temp = np.zeros(300)
            #     for i in range(300):
            #         temp[i] = np.mean(matrix[c, max(0, i-window_size):i+window_size])
            #     m = np.max(temp)
            #     avg = np.mean(temp) / m
            #     features.append(m)
            #     features.append(avg)
            #
            # for c in range(55):
            #     h, _ = np.histogram(matrix[c], bins=bins, range=(0,0.1))
            #     features += list(h)

            # print(','.join([str(x) for x in features]), file=f)

            for c in range(55):
                # features = [data['planet'], c, data['sma'], data['incl'], data['radii'][c], data['radiation'][c]]
                features = [data['planet'], c, '?', '?', '?', data['radiation'][c]]
                temp = np.zeros(300)
                for i in range(300):
                    temp[i] = np.mean(matrix[c, max(0, i-window_size):i+window_size])
                m = np.max(temp)
                avg = np.mean(temp) / m
                features.append(m)
                features.append(avg)
                h, _ = np.histogram(matrix[c], bins=bins, range=(0, 0.1))
                features += list(h)

                print(','.join([str(x) for x in features]), file=f)


def create_ariel_frequencies(equidistant_frequency=False, equidistant_wavelengths=False):
    """
    Creates an array of 55 frequencies at which Ariel may be observing the planets.
    :param equidistant_frequency:
    :param equidistant_wavelengths:
    :return:
    """
    assert 1 == (equidistant_frequency + equidistant_wavelengths)
    if equidistant_frequency:
        ranges = FREQUENCY_RANGES
    else:
        ranges = WAVELENGTHS_RANGES
    # there are 55 channels
    extended = np.array([x for r in ranges for x in np.linspace(r[0], r[1], 33)])
    if equidistant_wavelengths:
        # convert to frequencies
        extended = C / extended
    return extended


def sort_rows_by_noise(csv_file, is_auto_corr=False):
    def compute_noise(r):
        r = list(r)
        if not is_auto_corr:
            # first idea of counting spikes etc.
            local_extrema = [i for i in range(1, n_measurements - 1)
                             if r[i] == max(r[i - 1:i + 1]) or r[i] == min(r[i - 1:i + 1])]
            noise = 0.0
            for i in local_extrema:
                noise += (abs(r[i] - r[i + 1]) + abs(r[i] - r[i - 1])) / 2
        else:
            # auto-correlation with offset 1
            offset = 1
            matrix = np.corrcoef(r[offset:], r[:-offset])
            assert abs(matrix[0, 1] - matrix[1, 0]) < 10 ** -10
            corr = max(matrix[0, 1], 0)  # 0 or negative --> 0
            noise = 1 - corr
        return noise

    if is_auto_corr:
        print("Auto corr")
    else:
        print("local extrema")
    n_measurements = 300
    with open(csv_file) as f:
        header = f.readline().strip().split(',')
    columns = [i for i in range(len(header)) if re.search('^m[0-9]+$', header[i]) is not None]

    assert len(columns) == n_measurements
    data = np.genfromtxt(csv_file, delimiter=',', usecols=columns, names=True)
    channels = 55
    planets = len(data) // channels
    assert len(data) == channels * planets
    orders = []
    for planet in tqdm(range(planets)):
        noise_levels = []
        for channel in range(channels):
            row = data[channels * planet + channel]
            noise_levels.append(max(0, compute_noise(row)))
        order = sorted(range(channels), key=lambda i: noise_levels[i])  # infra red at the end
        values = [0] * channels
        for position, channel in enumerate(order):
            values[channel] = position
        orders.append(values)
    orders = np.array(orders)
    mean_ranks = orders.mean(axis=0)
    std_deviations = orders.std(axis=0)
    print("Mean ranks and std deviations by channels")
    print(list(mean_ranks))
    print(list(std_deviations))
    print("Channels order")
    print(sorted(range(channels), key=lambda i: mean_ranks[i]))
    print(sorted(mean_ranks))


def black_body_radiation(nu, t):
    return 2 * H * nu ** 3 / C ** 2 / (np.exp(H * nu / K / t) - 1)


def create_radiation_features(basic_csv):
    """
    We read the ID and star temperature columns in the basic csv,
    and features that give the amount of the radiation around the frequency that we assume this channel has.
    They are sorted in a way that the closest frequency comes first, then its two neighbours etc.

    :param basic_csv:
    :return:
    """
    # NOt necssary: ordered channels from least to most noisy: see ../explore/noise.txt
    channels = 55
    # neighbourhoods = []
    # for position in range(channels):
    #     neighbourhoods.append(sorted(range(channels), key=lambda p: (abs(p - position), p > position)))

    frequencies = create_ariel_frequencies(equidistant_frequency=True)
    # positions = [-1] * channels
    # for i, channel in enumerate(NOISE_ORDER):
    #     positions[channel] = i
    columns = ["ID"] + ["radiation{}".format(f) for f in range(1, 1 + len(frequencies))]
    data = {c: [] for c in columns}
    with open(basic_csv) as f:
        header = f.readline().strip().split(',')
        i_id = header.index("ID")
        i_temp = header.index("star_temp")
        for l in tqdm(f):
            line = l.strip().split(',')
            # planet_channel = line[i_id].split('_')
            # channel = int(planet_channel[1]) - 1
            # i0 = positions[channel]
            nus = frequencies  # [frequencies[i] for i in neighbourhoods[i0]]
            temperature = float(line[i_temp])
            for c, n in zip(columns[1:], nus):
                data[c].append(black_body_radiation(n, temperature))
            data[columns[0]].append(line[i_id])
    df = pd.DataFrame(data, columns=columns)
    out_dir = "../database/csv"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "radiations_extended.csv"), index=False)


if __name__ == '__main__':
    # examples_with_aggregated_matrices('train')
    # examples_with_aggregated_matrices('test')

    # examples_with_custom_features('train')
    examples_with_custom_features('test')

    # histogram_arffs()
    # examples_with_maximums(get_planets('train'))

    # create_fully_aggregated_csv(np.mean, np.median, "basic_mean_median.csv")
    # create_fully_aggregated_csv(np.median, np.median, "basic_median_median.csv")

    # csv = "../database/csv/mean_median.csv"
    # print(csv)
    # sort_rows_by_noise(csv, True)

    # create_radiation_features("../database/csv/basic_mean_median.csv")  # does not matter which aggregation ...

    # calculate_tsfresh_features()
    # aggregated_examples_with_tsfresh()
