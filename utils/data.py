import os
from glob import glob

import numpy as np
from tqdm import tqdm


def load_from_directory(path):
    fpaths = sorted(glob(os.path.join(path,'*.npy')))
    imgs = []

    for f in tqdm(fpaths):
        imgs.append(np.load(f))
    
    return np.array(imgs)