from scipy.cluster.hierarchy import *
from scipy.spatial.distance import squareform as sf
import random


def normalize(clusters):
    return sorted([sorted(cluster) for cluster in clusters])


def is_ok(clusters):
    return list(range(55)) == sorted([i for cluster in clusters for i in cluster])


def create_clusters(clustering_matrix):
    thresholds = clustering_matrix[:, 2]
    clusterings = []
    for i in range(len(thresholds) - 1):
        t = (thresholds[i] + thresholds[i + 1]) / 2
        cluster_indices = fcluster(clustering_matrix, t, criterion='distance')  # 1-based for some reason
        clusters = [[] for _ in range(max(cluster_indices))]
        for channel, cluster_index in enumerate(cluster_indices):
            clusters[cluster_index - 1].append(channel)
        clusterings.append(normalize(clusters))
        if not is_ok(clusterings[-1]):
            raise ValueError("1. Weird cluster:", clusterings[-1])
    return clusterings


def create_random_clusters(n_clusterings, n_clusters):
    random.seed(123)
    clusterings = []
    for _ in range(n_clusterings):
        clusters = [[] for _ in range(n_clusters)]
        available = list(range(55))
        random.shuffle(available)
        for i in range(n_clusters):
            clusters[i].append(available[i])
        for i in range(n_clusters, len(available)):
            i_cluster = int(random.random() * len(clusters))
            clusters[i_cluster].append(available[i])
        clusterings.append(normalize(clusters))
        if not is_ok(clusterings[-1]):
            raise ValueError("2. Weird cluster:", clusterings[-1])
    return clusterings


def create_all():
    distances = []
    with open("../explore/cluster_targets/targets.csv") as f:
        for l in f:
            distances.append([1 - float(t) for t in l.strip().split(",")])
    d = sf(distances)
    clusterings = [[list(range(55))],
            [[i] for i in range(55)]]
    for f in [single, complete, average, weighted, centroid, median, ward]:
        z = f(d)
        clusterings += create_clusters(z)
        # fig = plt.figure(figsize=(25, 20))
        # dn = dendrogram(z)
        # plt.savefig("{}.pdf".format(f.__name__))
    for n_clusters in range(2, 40):
        clusterings += create_random_clusters(10, n_clusters)
    # filter redundant
    clusterings.sort()
    unique = []
    prev = []
    for clusters in clusterings:
        if clusters != prev:
            unique.append(clusters)
            prev = clusters

    print("Out of", len(clusterings), "we kept", len(unique))
    return unique


def create_all_clusters():
    distances = []
    with open("../explore/cluster_targets/targets.csv") as f:
        for l in f:
            distances.append([1 - float(t) for t in l.strip().split(",")])
    d = sf(distances)
    clusters = [list(range(55))] + [[i] for i in range(55)]  # sike targeti zaedno i sekoj target sam
    for f in [single, complete, average, weighted, centroid, median, ward]:
        z = f(d)
        for clustering in create_clusters(z):
            clusters += clustering
        # fig = plt.figure(figsize=(25, 20))
        # dn = dendrogram(z)
        # plt.savefig("{}.pdf".format(f.__name__))
    for n_clusters in range(2, 40):
        for clustering in create_random_clusters(10, n_clusters):
            clusters += clustering
    # filter redundant
    for c in clusters:
        c.sort()
    clusters.sort()
    unique = []
    prev = []
    for c in clusters:
        if c != prev:
            unique.append(c)
            prev = c

    print("Out of", len(clusters), "we kept", len(unique))
    return unique
