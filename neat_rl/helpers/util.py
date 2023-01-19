import numpy as np
import json
from sklearn.neighbors import KDTree


def load_cvt(k, dim):
    fname = f"CVT/centroids_{k}_{dim}.dat"    
    if dim == 1:
        if k == 1:
            return np.expand_dims(np.expand_dims(np.loadtxt(fname), axis=0), axis=1)
        return np.expand_dims(np.loadtxt(fname), axis=1)
    else:
        if k == 1:
            return np.expand_dims(np.loadtxt(fname), axis=0)
        return np.loadtxt(fname)

def load_kdtree(env):
    if "QDHalfCheetah" in env:
        n_niches = 1024
        behavior_dim = 2
    elif "QDAnt" in env:
        n_niches = 1296
        behavior_dim = 4
    elif "QDHopper" in env:
        n_niches = 1000
        behavior_dim = 1
    elif "QDWalker" in env:
        n_niches = 1024
        behavior_dim = 2

    kdt = KDTree(load_cvt(n_niches, behavior_dim), leaf_size=30, metric='euclidean')
    
    return kdt

def add_to_archive(org, archive, kdt):
    niche_index = kdt.query([org.behavior], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = tuple(niche)
    assert org.best_fitness != -100000
    
    if n in archive:
        if org.best_fitness > archive[n]:
            archive[n] = org.best_fitness
    else:
        archive[n] = org.best_fitness

def save_archive(archive, save_file):
    archive_items = list(archive.items())
    
    archive_dict = {
        "items": archive_items,
    }

    with open(save_file, "w") as f:
        json.dump(archive_dict, f)


def load_archive(save_file):
    archive = {}
    with open(save_file) as f:
        archive_dict = json.load(f)
    
    for item in archive_dict["items"]:
        archive[tuple(item[0])] = item[1]
    
    return archive


     