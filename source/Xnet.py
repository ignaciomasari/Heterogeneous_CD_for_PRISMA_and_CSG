import os

# Set loglevel to suppress tensorflow GPU messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import datasets
from change_detector import ChangeDetector
from image_translation import WeightedTranslationNetwork
from change_priors import eval_prior
from config import get_config_Xnet
from decorators import image_to_tensorboard
import numpy as np

class XNet(ChangeDetector):
    def __init__(self, translation_spec, **kwargs):
        """
                Input:
                    translation_spec - dict with keys 'X_to_Y', 'Y_to_X'.

                    Values passed as kwargs:
                    W_CYCLE=2 - float, loss weight
                    W_TRAN=3 - float, loss weight
                    W_REG=1e-3 - float, loss weight
                    learning_rate=1e-4 - float, initial learning rate for
                                         ExponentialDecay
                    clipnorm=1 - gradient norm clip value, passed to
                                    tf.clip_by_global_norm if not None
                    logdir=./logs/{dataset_name}/ - path to log directory. If provided, tensorboard
                                                    logging of training and evaluation is set up at
                                                    'logdir/timestamp/' + 'train' and 'evaluation'
        """

        super().__init__(**kwargs)

        self.W_TRAN = kwargs.get("W_TRAN", 1)
        self.W_CYCLE = kwargs.get("W_CYCLE", 1)
        self.l2_lambda = kwargs.get("W_REG", 0.01)
        self.min_impr = kwargs.get("minimum improvement", 1e-3)
        self.patience = kwargs.get("patience", 10)
        self.crop_factor = kwargs.get("crop_factor", 0.2)
        
        # encoder from X to Y
        self._X_to_Y = WeightedTranslationNetwork(
            **translation_spec["X_to_Y"], name="X_to_Y", l2_lambda=self.l2_lambda
        )

        # encoder from Y to X
        self._Y_to_X = WeightedTranslationNetwork(
            **translation_spec["Y_to_X"], name="Y_to_X", l2_lambda=self.l2_lambda
        )

        self.loss_object = tf.keras.losses.MeanSquaredError()

        self.train_metrics["cycle_x"] = tf.keras.metrics.Sum(name="cycle_x MSE sum")
        self.train_metrics["alpha_x"] = tf.keras.metrics.Sum(name="alpha_x MSE sum")
        self.train_metrics["cycle_y"] = tf.keras.metrics.Sum(name="cycle_y MSE sum")
        self.train_metrics["alpha_y"] = tf.keras.metrics.Sum(name="alpha_y MSE sum")
        self.train_metrics["l2"] = tf.keras.metrics.Sum(name="l2 MSE sum")
        self.train_metrics["total"] = tf.keras.metrics.Sum(name="total MSE sum")

        # Track total loss history for use in early stopping
        self.metrics_history["total"] = []

    def save_all_weights(self):
        self._X_to_Y.save_weights(self.log_path + "/weights/_X_to_Y/")
        self._Y_to_X.save_weights(self.log_path + "/weights/_Y_to_X/")

    def load_all_weights(self, folder):
        self._X_to_Y.load_weights(folder + "/weights/_X_to_Y/")
        self._Y_to_X.load_weights(folder + "/weights/_Y_to_X/")

    @image_to_tensorboard()
    def X_to_Y(self, inputs, training=False):
        """ Wraps encoder call for TensorBoard printing and image save """
        return self._X_to_Y(inputs, training)

    @image_to_tensorboard()
    def Y_to_X(self, inputs, training=False):
        return self._Y_to_X(inputs, training)

    def early_stopping_criterion(self):
        temp = tf.math.reduce_min([self.stopping, self.patience]) + 1
        self.stopping.assign_add(1)
        last_losses = np.array(self.metrics_history["total"][-(temp):])
        idx_min = np.argmin(last_losses)
        if idx_min == (temp - 1):
            self.save_all_weights()
        while idx_min > 0:
            idx_2nd_min = np.argmin(last_losses[:idx_min])
            improvement = last_losses[idx_2nd_min] - last_losses[idx_min]
            if improvement > self.min_impr:
                break
            else:
                idx_min = idx_2nd_min
        stop = idx_min == 0 and self.stopping > self.patience
        tf.print(
            "total_loss",
            last_losses[-1],
            "Target",
            last_losses[idx_min],
            "Left",
            self.patience - (temp - 1) + idx_min,
        )
        return stop

    @tf.function
    def __call__(self, inputs, training=False):
        x, y = inputs
        tf.debugging.Assert(tf.rank(x) == 4, [x.shape])
        tf.debugging.Assert(tf.rank(y) == 4, [y.shape])

        if training:
            x_hat, y_hat = self._Y_to_X(y, training), self._X_to_Y(x, training)
            x_dot, y_dot = self._Y_to_X(y_hat, training), self._X_to_Y(x_hat, training)
            retval = [x_hat, y_hat, x_dot, y_dot]

        else:
            x_hat, y_hat = self.Y_to_X(y, name="x_hat"), self.X_to_Y(x, name="y_hat")
            difference_img = self._difference_img(x, y, x_hat, y_hat)
            retval = difference_img

        return retval

    @tf.function
    def _train_step(self, x, y, non_change_prob):
        """
        Input:
        x - tensor of shape (bs, ps_h, ps_w, c_x)
        y - tensor of shape (bs, ps_h, ps_w, c_y)
        aff - affinity, tensor of shape (bs, ps_h, ps_w, 1)
        """
        with tf.GradientTape() as tape:
            x_hat, y_hat, x_dot, y_dot = self(
                [x, y], training=True
            )            
            l2_loss = (
                sum(self._X_to_Y.losses)
                + sum(self._Y_to_X.losses)
            )
            cycle_x_loss = self.W_CYCLE * self.loss_object(x, x_dot)
            cycle_y_loss = self.W_CYCLE * self.loss_object(y, y_dot)
            alpha_x_loss = self.W_TRAN * self.loss_object(x, x_hat, non_change_prob)
            alpha_y_loss = self.W_TRAN * self.loss_object(y, y_hat, non_change_prob)                        

            total_loss = (
                cycle_x_loss
                + cycle_y_loss
                + alpha_x_loss
                + alpha_y_loss
                + l2_loss
            )

            targets_all = (
                self._X_to_Y.trainable_variables
                + self._Y_to_X.trainable_variables
            )

            gradients_all = tape.gradient(total_loss, targets_all)

            if self.clipnorm is not None:
                gradients_all, _ = tf.clip_by_global_norm(gradients_all, self.clipnorm)
            self._optimizer_all.apply_gradients(zip(gradients_all, targets_all))

        self.train_metrics["cycle_x"].update_state(cycle_x_loss)
        self.train_metrics["alpha_x"].update_state(alpha_x_loss)
        self.train_metrics["cycle_y"].update_state(cycle_y_loss)
        self.train_metrics["alpha_y"].update_state(alpha_y_loss)
        self.train_metrics["l2"].update_state(l2_loss)
        self.train_metrics["total"].update_state(total_loss)

def test(DATASET="E_R2", CONFIG=None, n_ch_y=None, reduction_method=None):
    """
    XNet from L. T. Luppino et al., "Deep Image Translation With an 
    Affinity-Based Change Prior for Unsupervised Multimodal Change Detection,"
    in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-22,
    2022, Art no. 4700422, doi: 10.1109/TGRS.2021.3056196

    Steps:
    1. Fetch data (x, y, change_map)
    2. Compute/estimate A_x and A_y (for patches)
    3. Compute change_prior
    4. Define dataset with (x, A_x, y, A_y, alpha). Choose patch size compatible
       with affinity computations.
    5. Train XNet unsupervised
    6. Evaluate the change detection scheme
        a. change_map = threshold (CRF[(x - f_y(y))/2 + (y - f_x(x))/2])
    """
    if CONFIG is None:
        CONFIG = get_config_Xnet(DATASET)

    if n_ch_y is not None:
        CONFIG.update({"n_channels_y":n_ch_y})

    if reduction_method is not None:
        CONFIG.update({"reduction_method":reduction_method})

    print(f"Loading {DATASET} data")
    x_im, y_im, EVALUATE, (C_X, C_Y) = datasets.fetch(DATASET, **CONFIG)
    if tf.config.list_physical_devices("GPU") and not CONFIG["debug"]:
        
        NF1 = CONFIG["NF1"]
        NF2 = CONFIG["NF2"]
        NF3 = CONFIG["NF3"]

        FS1 = CONFIG["FS1"]
        FS2 = CONFIG["FS2"]
        FS3 = CONFIG["FS3"]
        FS4 = CONFIG["FS4"]

        print("working with GPU")
        NETWORK_SPEC = {
            "X_to_Y": {"input_chs": C_X, "filter_spec": [
                                                            [C_X, NF1, FS1, 1],
                                                            [NF1, NF2, FS2, 1],
                                                            [NF2, NF3, FS3, 1],
                                                            [NF3, C_Y, FS4, 1],
                                                        ]},
            "Y_to_X": {"input_chs": C_Y, "filter_spec": [
                                                            [C_Y, NF1, FS1, 1],
                                                            [NF1, NF2, FS2, 1],
                                                            [NF2, NF3, FS3, 1],
                                                            [NF3, C_X, FS4, 1],
                                                        ]},
        }
    else:
        print("working with CPU")
        FS1 = 3
        NETWORK_SPEC = {
            "X_to_Y": {"input_chs": C_X, "filter_spec": [C_X, C_Y, FS1, 1]},
            "Y_to_X": {"input_chs": C_Y, "filter_spec": [C_Y, C_X, FS1, 1]},
        }

    print("Change Detector Init")
    cd = XNet(NETWORK_SPEC, **CONFIG)

    n_channels_y = CONFIG["n_channels_y"]
    reduction_method = CONFIG["reduction_method"]

    print("Pre-training")
    alpha_path = f"./data/{DATASET}/change-prior_{reduction_method}_{n_channels_y}.npy"
    if CONFIG["PRE_TRAIN"] is False:
        try:
            alpha = np.array(np.squeeze(np.load(alpha_path)), dtype=np.float32)
            alpha = tf.expand_dims(tf.expand_dims(alpha, 0), -1)
            print("Prior loaded correctly")
        except Exception as exc:
            print(exc)
            print("Prior under evaluation")
            alpha = eval_prior(DATASET, np.array(x_im[0], dtype=np.float32), np.array(y_im[0], dtype=np.float32), **CONFIG)
            np.save(alpha_path, alpha)
            alpha = tf.expand_dims(alpha, 0)
    else:
        print("Prior under evaluation")
        alpha = eval_prior(DATASET, np.array(x_im[0], dtype=np.float32), np.array(y_im[0], dtype=np.float32), **CONFIG)
        np.save(alpha_path, alpha)
        alpha = tf.expand_dims(alpha, 0)
 
    non_change_prob = 1.0 - alpha

    print("Training")
    training_time = 0
    for epochs in CONFIG["list_epochs"]:
        CONFIG.update(epochs=epochs)
        tr_gen, dtypes, shapes = datasets._training_data_generator(
            x_im[0], y_im[0], non_change_prob[0], CONFIG["patch_size"]
        )
        TRAIN = tf.data.Dataset.from_generator(tr_gen, dtypes, shapes)
        TRAIN = TRAIN.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        tr_time, _ = cd.train(TRAIN, evaluation_dataset=EVALUATE, **CONFIG)

        if CONFIG['UPDATE_ALPHA']:
            for x, y, _ in EVALUATE.batch(1):
                alpha = eval_prior(DATASET, np.array(x[0], dtype=np.float32), np.array(y[0], dtype=np.float32), **CONFIG)    
            non_change_prob = 1.0 - alpha

        training_time += tr_time

    cd.load_all_weights(cd.log_path)
    cd.final_evaluate(EVALUATE, **CONFIG)
    metrics = {}
    for key in list(cd.difference_img_metrics.keys()) + list(
        cd.change_map_metrics.keys()
    ):
        metrics[key] = cd.metrics_history[key][-1]

    metrics["P_change"] = metrics["TP"] / (metrics["TP"] + metrics["FP"])
    metrics["P_no_change"] = metrics["TN"] / (metrics["TN"] + metrics["FN"])
    metrics["R_change"] = metrics["TP"] / (metrics["TP"] + metrics["FN"])
    metrics["R_no_change"] = metrics["TN"] / (metrics["TN"] + metrics["FP"])                
    metrics["FAR"] = metrics["FP"] / (metrics["TN"] + metrics["FP"])
    timestamp = cd.timestamp
    epoch = cd.epoch.numpy()
    speed = (epoch, training_time, timestamp)
    del cd

    return metrics, speed

if __name__ == "__main__":

    DATASET = "Bolsena_30m"

    print(test(DATASET))
