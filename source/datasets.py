import os

# Set loglevel to suppress tensorflow GPU messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
from itertools import count
import numpy as np
import tensorflow as tf
from scipy.io import loadmat, savemat
from change_priors import eval_prior, remove_borders, image_in_patches

def load_prior(name, expected_hw=None):
    """ Load prior from disk, validate dimensions and force shape to (h, w, 1) """
    mat = loadmat("data/" + name + "/change-prior.mat")
    varname = (
        re.sub(r"\W+", "", "aff" + str(expected_hw))
        if expected_hw is not None
        else "aff"
    )
    prior = tf.convert_to_tensor(mat[varname], dtype=tf.float32)
    if expected_hw is not None and prior.shape[:2] != expected_hw:
        raise FileNotFoundError
    if prior.ndim == 2:
        prior = prior[..., np.newaxis]
    return prior

def evaluate_prior(name, x, y, **kwargs):
    alpha = eval_prior(name, x, y, **kwargs)
    varname = re.sub(r"\W+", "", "aff" + str(x.shape[:2]))
    prior_path = "data/" + name + "/change-prior.mat"
    try:
        mat = loadmat(prior_path)
        mat.update({varname: alpha})
    except FileNotFoundError as e:
        mat = {varname: alpha}
    savemat(prior_path, mat)
    return alpha

def _EmiliaRomagna(nc2=3):
    """ Load EmiliaRomagna dataset from .npy """

    t1 = np.load("./data/E_R/CSG_dualpol_20230521_mlk3_clip_resam.npy")
    t1 = np.log(t1 + 0.1)
    t2 = np.load(f"./data/E_R/PRISMA_{nc2}ch.npy")
    change_mask = np.load(f"./data/E_R/ground_truth_flood.npy")

    if t1.shape[-1] == 3:
        t1 = t1[..., 0]
    t1, t2, change_mask = (
        remove_borders(t1, 2),
        remove_borders(t2, 2),
        remove_borders(change_mask, 2),
    )
    t1, t2 = _clip(t1, log=True), _clip(t2)

    change_mask = tf.convert_to_tensor(change_mask, dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    change_mask = change_mask[..., :1]
    return t1, t2, change_mask

def _EmiliaRomagna2(nc2=3, reduction_method="UMAP"):
    """ Load EmiliaRomagna 2 dataset from .npy """

    t1 = np.load("./data/E_R2/CSG_dualpol_20230521_mlk3_clip_resam.npy")
    t2 = np.load(f"./data/E_R2/PRISMA_{reduction_method}_{nc2}ch.npy")
    # change_mask = np.array(np.load(f"./data/E_R2/ground_truth_flood.npy"), dtype=np.int8)

    print("Warning! Using corrected GT!")
    change_mask = np.array(np.load(f"./data/E_R2/GT_corrected_311023.npy"), dtype=np.int8)
    
    exclude_mask = np.load(f"./data/E_R2/exclude.npy")
    change_mask[exclude_mask==1] = -1

    if len(t1.shape)==2:
        t1 = t1[..., np.newaxis]

    if len(t2.shape)==2:
        t2 = t2[..., np.newaxis]

    if len(change_mask.shape)==2:
        change_mask = change_mask[..., np.newaxis]

    t1, t2, change_mask = (
        remove_borders(t1, 2),
        remove_borders(t2, 2),
        remove_borders(change_mask, 2),
    )
    t1, t2 = _clip(t1, log=True), _clip(t2)

    change_mask = tf.convert_to_tensor(change_mask, dtype=tf.int8)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    change_mask = change_mask[..., :1]
    return t1, t2, change_mask

def _Lucca(nc2=3, reduction_method="UMAP"):
    """ Load Lucca dataset from .npy """    

    t1 = np.load("./data/LUCCA/S1_VV_clipped_clip2_bands.npy")
    t2 = np.load(f"./data/LUCCA/PRISMA_{reduction_method}_{nc2}ch.npy")
    # change_mask = np.array(np.load(f"./data/LUCCA/GT.npy"), dtype=np.int8)
    change_mask = np.zeros(t1.shape[:2], dtype=np.int8)

    if len(t1.shape)==2:
        t1 = t1[..., np.newaxis]

    if len(t2.shape)==2:
        t2 = t2[..., np.newaxis]

    if len(change_mask.shape)==2:
        change_mask = change_mask[..., np.newaxis]

    t1, t2, change_mask = (
        remove_borders(t1, 2),
        remove_borders(t2, 2),
        remove_borders(change_mask, 2),
    )
    t1, t2 = _clip(t1, log=True), _clip(t2)

    change_mask = tf.convert_to_tensor(change_mask, dtype=tf.int8)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    change_mask = change_mask[..., :1]
    return t1, t2, change_mask

def _Bolsena_30m(nc2=3, reduction_method="UMAP"):
    """ Load Bolsena lake at 30m spatial resolution dataset from .npy """
    t1 = np.load("./data/Bolsena_30m/CSG_SSAR2_GTC_B_DualPol_RD_F_20231004_Clipped_mod_resamp.npy")
    t2 = np.load(f"./data/Bolsena_30m/PRISMA_{reduction_method}_{nc2}ch.npy")
    change_mask = np.array(np.load(f"./data/Bolsena_30m/GT_difference.npy"), dtype=np.int8)

    if len(t1.shape)==2:
        t1 = t1[..., np.newaxis]

    if len(t2.shape)==2:
        t2 = t2[..., np.newaxis]

    if len(change_mask.shape)==2:
        change_mask = change_mask[..., np.newaxis]

    t1, t2, change_mask = (
        remove_borders(t1, 2),
        remove_borders(t2, 2),
        remove_borders(change_mask, 2),
    )
    t1, t2 = _clip(t1, log=True), _clip(t2)

    change_mask = tf.convert_to_tensor(change_mask, dtype=tf.int8)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    change_mask = change_mask[..., :1]
    return t1, t2, change_mask

def _clip(image, log=False):
    """
        Normalize image from R_+ to [-1, 1].

        For each channel, clip any value larger than mu + 3sigma,
        where mu and sigma are the channel mean and standard deviation.
        Scale to [-1, 1] by (2*pixel value)/(max(channel)) - 1

        Input:
            image - (h, w, c) image array in R_+
        Output:
            image - (h, w, c) image array normalized within [-1, 1]
    """

    temp = np.reshape(image, (-1, image.shape[-1]))

    if log:
        temp = np.log(temp + 0.1)

    upper_limits = tf.reduce_mean(temp, 0) + 3.0 * tf.math.reduce_std(temp, 0)
    lower_limits = tf.reduce_mean(temp, 0) - 3.0 * tf.math.reduce_std(temp, 0)
    for i, (upper, lower) in enumerate(zip(upper_limits, lower_limits)):
        channel = temp[:, i]
        channel = tf.clip_by_value(channel, lower, upper)
        max, min = tf.reduce_max(channel), tf.reduce_min(channel)
        channel = 2.0 * ((channel - min) / (max - min)) - 1
        temp[:, i] = channel

    return tf.reshape(tf.convert_to_tensor(temp, dtype=tf.float32), image.shape)

def _training_data_generator(x, y, p, patch_size):
    """
        Factory for generator used to produce training dataset.
        The generator will choose a random patch and flip/rotate the images

        Input:
            x - tensor (h, w, c_x)
            y - tensor (h, w, c_y)
            p - tensor (h, w, 1)
            patch_size - int in [1, min(h,w)], the size of the square patches
                         that are extracted for each training sample.
        Output:
            to be used with tf.data.Dataset.from_generator():
                gen - generator callable yielding
                    x - tensor (ps, ps, c_x)
                    y - tensor (ps, ps, c_y)
                    p - tensor (ps, ps, 1)
                dtypes - tuple of tf.dtypes
                shapes - tuple of tf.TensorShape
    """
    c_x, c_y = x.shape[2], y.shape[2]
    chs = c_x + c_y + 1
    x_chs = slice(0, c_x, 1)
    y_chs = slice(c_x, c_x + c_y, 1)
    p_chs = slice(c_x + c_y, chs, 1)

    data = tf.concat([x, y, p], axis=-1)

    def gen():
        for _ in count():
            tmp = tf.image.random_crop(data, [patch_size, patch_size, chs])
            tmp = tf.image.rot90(tmp, np.random.randint(4))
            tmp = tf.image.random_flip_up_down(tmp)

            yield tmp[:, :, x_chs], tmp[:, :, y_chs], tmp[:, :, p_chs]

    dtypes = (tf.float32, tf.float32, tf.float32)
    shapes = (
        tf.TensorShape([patch_size, patch_size, c_x]),
        tf.TensorShape([patch_size, patch_size, c_y]),
        tf.TensorShape([patch_size, patch_size, 1]),
    )

    return gen, dtypes, shapes

DATASETS = {
    "Bolsena_30m": _Bolsena_30m,
    "E_R": _EmiliaRomagna,
    "E_R2": _EmiliaRomagna2,
    "LUCCA": _Lucca,
}

def fetch(name, **kwargs):
    """
        Input:
            name - dataset name, should be in DATASETS
            kwargs - config {key: value} pairs.
                     Key should be in DATASET_DEFAULT_CONFIG
        Output:
            training_data - tf.data.Dataset with (x, y, prior)
                            shapes like (inf, patch_size, patch_size, ?)
            evaluation_data - tf.data.Dataset with (x, y, change_map)
                              shapes (1, h, w, ?)
            channels - tuple (c_x, c_y), number of channels for domains x and y
    """

    n_ch_y = kwargs['n_channels_y']
    red_method = kwargs['reduction_method']
    x_im, y_im, target_cm = DATASETS[name](n_ch_y, red_method)

    if not tf.config.list_physical_devices("GPU"):
        dataset = [
            tf.image.central_crop(tensor, 0.1) for tensor in [x_im, y_im, target_cm]
        ]
    else:
        dataset = [x_im, y_im, target_cm]

    dataset = [tf.expand_dims(tensor, 0) for tensor in dataset]
    x, y = dataset[0], dataset[1]
    evaluation_data = tf.data.Dataset.from_tensor_slices(tuple(dataset))

    c_x, c_y = x_im.shape[-1], y_im.shape[-1]

    return x, y, evaluation_data, (c_x, c_y)


if __name__ == "__main__":
    for DATASET in DATASETS:
        print(f"Loading {DATASET}")
        fetch(DATASET)
