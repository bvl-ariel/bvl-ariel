import numpy as np
from sklearn.decomposition import PCA
from tqdm import trange
import pandas as pd


pattern = "../database/csv/channel_csv/channel{}{}.csv"


def denoise(matrix, explained_variance, out_file=None):
    """
    :param explained_variance: 0 < ... < 1
    :return:
    """
    pca = PCA(n_components=explained_variance)
    pca.fit(matrix)
    # print("We keep", pca.n_components_, "components")
    denoised = pca.inverse_transform(pca.transform(matrix))
    if out_file is not None:
        np.savetxt(out_file, denoised, delimiter=',',
                   fmt='%.18f')
    return denoised


def denoise_channels(explained_variance):
    for c in trange(55):
        matrix = np.genfromtxt(pattern.format(c, ""), delimiter=',')
        denoise(matrix, explained_variance, pattern.format(c, "_cleaned_{}".format(explained_variance)))


def join_channels(explained_variance):
    matrices = []
    for c in trange(55):
        m = np.genfromtxt(pattern.format(c, "_cleaned_{}".format(explained_variance)), delimiter=',')
        matrices.append(m)
    matrix = np.zeros((55 * matrices[0].shape[0], 300))
    for c, m in enumerate(matrices):
        matrix[c::55, :m.shape[1]] = m
    index_values = pd.read_csv("../database/csv/avg_max.csv", sep=',', index_col=0).index.tolist()
    df = pd.DataFrame(data=matrix, columns=["m{}".format(i) for i in range(1, 301)], index=index_values)
    df.to_csv("../database/csv/xbasic_mean_median_cleaned_{}.csv".format(explained_variance))


def prepare_clean_basic_mean_median(explained_variance):
    file = "../database/csv/{}basic_mean_median{}.csv"
    appendix = "_cleaned_{}".format(explained_variance)
    df_ugly = pd.read_csv(file.format("", ""), sep=',', index_col=0, float_precision="high")
    df_clean = pd.read_csv(file.format("x", appendix), sep=',', index_col=0, float_precision="high")
    cs = ["m{}".format(i) for i in range(1, 301)]
    df_ugly[cs] = df_clean[cs]
    df_ugly.to_csv(file.format("", appendix))


n = 50
denoise_channels(n)
join_channels(n)
prepare_clean_basic_mean_median(n)
