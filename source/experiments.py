import os
import gc


# Set loglevel to suppress tensorflow GPU messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
from datetime import datetime
import pandas as pd
import os.path
import errno
import tensorflow as tf

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

from config import *
import shelve
import numpy as np

from Code_Aligned_Autoencoders import test as test_kACE
from SCCN import test as test_SCCN
from Xnet import test as test_Xnet
from I2Inet import test as test_I2Inet
from I2Inet2 import test as test_I2Inet2

MODELS = {"Kern_AceNet": test_kACE,
          "SCCN": test_SCCN,
          "Xnet": test_Xnet,
          "I2INet": test_I2Inet,
          "I2INet2": test_I2Inet2}

CONFIG_FUNS = {"Kern_AceNet": get_config_kACE,
               "SCCN": get_config_SCCN,
               "Xnet": get_config_Xnet,
               "I2INet": get_config_Xnet,
               "I2INet2": get_config_Xnet}
NAMES = {
    "No_Cycle": {"cycle_lambda": 0},
    "No_Rec": {"recon_lambda": 0},
    "No_align": {"kernels_lambda": 0},
    "No_tr": {"cross_lambda": 0},
    "lambda_k=1": {"kernels_lambda": 1},
    "lambda_k=10": {"kernels_lambda": 2},
    "lambda_k=.01": {"kernels_lambda": 0.01},
    "lambda_k=0.5": {"kernels_lambda": 0.5},
}

def models_run(models, runs, dataset, logdir, shelve_path, debug):
    for name, test in models.items():
        m_logdir = os.path.join(logdir, name)
        config = CONFIG_FUNS[name](dataset, debug=debug)
        print(
            f"Performing experiment with {runs} runs of {config['epochs']} epochs:",
            f"The logdir is {m_logdir}",
            sep="\n\t",
            end="\n\n",
        )
        config.update(evaluation_frequency=0)
        for run in range(runs):
            print(f"Run {run+1} of {name}:")
            tf.keras.backend.clear_session()
            config.update(logdir=os.path.join(m_logdir, str(run + 1)))
            performance, speed = test(dataset, config)
            final_kappa, final_acc = performance["Kappa"], performance["ACC"]
            epochs, training_time, timestamp = speed
            print(
                f"Trained {epochs} epochs in {training_time}",
                f"Final kappa = {final_kappa}",
                sep="\n\t",
                end="\n\n### ### ###\n",
            )
            performance.update({"training time": training_time})
            with shelve.open(shelve_path, writeback=True) as shelf:
                shelf[name][timestamp] = performance


def kACE_ablation(runs, dataset, logdir, shelve_path, debug):
    for name, param in NAMES.items():

        m_logdir = os.path.join(logdir, "Ablation", name)
        config = CONFIG_FUNS["Kern_AceNet"](dataset, debug=debug)
        print(
            f"Performing experiment with {runs} runs of {config['epochs']} epochs:",
            f"The logdir is {m_logdir}",
            sep="\n\t",
            end="\n\n",
        )
        config.update(evaluation_frequency=0)
        config.update(save_images=False)
        config.update(param)
        for run in range(runs):
            print(f"Run {run+1} of {name}:")
            tf.keras.backend.clear_session()
            config.update(logdir=os.path.join(m_logdir, str(run + 1)))

            performance, speed = test_kACE(dataset, config)
            final_kappa, final_acc = performance["Kappa"], performance["ACC"]
            epochs, training_time, timestamp = speed
            print(
                f"Trained {epochs} epochs in {training_time}",
                f"Final kappa = {final_kappa}",
                sep="\n\t",
                end="\n\n### ### ###\n",
            )
            performance.update({"training time": training_time})
            with shelve.open(shelve_path, writeback=True) as shelf:
                shelf[name][timestamp] = performance


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return path


def experiment(DATASETS=["Texas"]):
    DEBUG = False
    ABLATION = True
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    _LOGDIR = make_dir(f"logs/Experiments/{TIMESTAMP}/")
    for DATASET in DATASETS:
        LOGDIR = os.path.join(_LOGDIR, DATASET)
        SHELVE_PATH = make_dir(LOGDIR + "/shelf/") + "experiment"
        RUNS = 10 if not DEBUG and tf.config.list_physical_devices("GPU") else 2

        with shelve.open(SHELVE_PATH) as shelf:
            shelf["config"] = {"timestamp": TIMESTAMP, "dataset": DATASET, "runs": RUNS}
            if ABLATION:
                for key in NAMES:
                    shelf[key] = {}
            else:
                for key in MODELS:
                    shelf[key] = {}

        if ABLATION:
            kACE_ablation(RUNS, DATASET, LOGDIR, SHELVE_PATH, DEBUG)
        else:
            models_run(MODELS, RUNS, DATASET, LOGDIR, SHELVE_PATH, DEBUG)

        with shelve.open(SHELVE_PATH) as shelf:
            with open(LOGDIR + "config.json", "x") as FILE:
                json.dump(shelf["config"], FILE, indent=4)

        with shelve.open(SHELVE_PATH) as shelf:
            if ABLATION:
                METRICS = {key: shelf[key] for key in NAMES}
            else:
                METRICS = {key: shelf[key] for key in MODELS}

        DF = pd.DataFrame.from_dict(
            {(i, j): METRICS[i][j] for i in METRICS.keys() for j in METRICS[i].keys()},
            orient="index",
        )
        print(DF)

        with open(LOGDIR + "metrics.csv", "x") as FILE:
            DF.to_csv(FILE, index_label=["model", "timestamp"])

def models_run_save_file(DATASET, N, print_metrics, channels_y, model, reduction_method):
    
    test = MODELS[model]

    print_string = ''
    
    with open(f'./logs/{DATASET}/{model}_{reduction_method}_results.txt', 'a') as f:
        f.write(f"Channels_y: {channels_y} --------------------------\n")
    
    metrics_list = []

    config = CONFIG_FUNS[model](DATASET)
    config.update(n_channels_y=channels_y)
    config.update(reduction_method=reduction_method)
    logdir = config["logdir"] + f"_{model}_{reduction_method}_{channels_y}"

    for run in range(N):
        config.update(logdir=os.path.join(logdir,str(run + 1)))
        metrics_list.append([])

        tf.keras.backend.clear_session()
        gc.collect()

        metrics, _ = test(DATASET, CONFIG=config)
        metrics_list[-1].append(np.fromiter(metrics.values(), dtype=np.float32))

    metrics_array = np.array(metrics_list)
    mean = np.mean(metrics_array, axis=0)
    std = np.std(metrics_array, axis=0)

    for idx, metric_name in enumerate(metrics.keys()):
        if metric_name not in print_metrics:
            continue
        print_string += f"{mean[0,idx]} {std[0,idx]} "

    print_string += '\n'

    # write print_string to the end of results.txt file
    with open(f'./logs/{DATASET}/{model}_{reduction_method}_results.txt', 'a') as f:
        f.write(print_string)



if __name__ == "__main__":
    DATASET = "E_R2" #"Bolsena30m", "E_R2", "LUCCA"
    N = 5
    print_metrics = ['AUC', 'ACC', 'Kappa', 'P_change', 'P_no_change', 'R_change', 'R_no_change', 'FAR']
    
    channels_list = [1, 2, 3]
    reduction_methods = ['kPCA_linear', 'UMAP']#, 'kPCA_poly', 'kPCA_rbf']
    
    for reduction_method in reduction_methods:
        for channels_y in channels_list:
            
            models_run_save_file(DATASET, N, print_metrics, channels_y, 'I2INet2', reduction_method)
            models_run_save_file(DATASET, N, print_metrics, channels_y, 'Xnet', reduction_method)
            models_run_save_file(DATASET, N, print_metrics, channels_y, 'I2INet', reduction_method)
            models_run_save_file(DATASET, N, print_metrics, channels_y, 'Kern_AceNet', reduction_method)
            models_run_save_file(DATASET, N, print_metrics, channels_y, 'SCCN', reduction_method)
