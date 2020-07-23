from tqdm import tqdm
from torch.utils.data import Dataset
from utils import get_planets
import pickle
import numpy as np


class PlanetDataset(Dataset):

    def __init__(self, all_folds, included_folds, train_or_test='train'):

        planets = list(get_planets(train_or_test))
        planets.sort()
        window_size = 5
        folder = 'completely_aggregated'
        # folder = 'aggregated_tsfresh'
        # folder = 'gauss_aggregated'
        # self.channel = 6
        self.rows = []
        for i, planet in tqdm(enumerate(planets)):
            if i % all_folds in included_folds:
                with open('{}/{}.pickle'.format(folder, planet), 'rb') as f:
                    data = pickle.load(f)

                temp = np.zeros((55, 300))
                for i in range(300):
                    temp[:, i] = np.mean(data['matrix'][:, max(0, i-window_size):i+window_size], axis=1)
                data['maxes'] = np.max(temp, axis=1)
                data['relative_means'] = np.mean(temp, axis=1) / data['maxes']

                # for STR
                # if 'radii' in data:
                #     data['radii'] = data['radii'][self.channel]

                self.rows.append(data)

                # for per channel
                # for j in range(55):
                #     d = {x: data[x] for x in data}
                #     d['matrix'] = d['matrix'][:, j, :]
                #     d['channel'] = j
                #     if 'radii' in d:
                #         d['radii'] = d['radii'][j]
                #     self.rows.append(d)

        self.size = len(self.rows)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.rows[item]
