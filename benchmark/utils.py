from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import style
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

style.use('ggplot')


def read_tf_event(file_path):

    event_acc = EventAccumulator(str(file_path))
    event_acc.Reload()
    data = {}

    for tag in event_acc.Tags()['scalars']:
        _, data['steps'], data[tag] = zip(*event_acc.Scalars(tag))

    steps = data.pop('steps')

    min_len = min([len(v) for v in data.values()])
    data = {k: v[:min_len] for k, v in data.items()}
    df = pd.DataFrame(data)
    df.insert(0, 'Steps', steps[:min_len])
    return df


def get_logs(log_dir, tag):
    log_dir = Path(log_dir)
    tag += '*'
    files_path = [list(p.glob('events.out.tfevents.*'))[0] for p in log_dir.glob(tag)]

    runs = {}
    for i, f in enumerate(files_path):
        runs[i] = read_tf_event(f)

    return pd.Panel(runs)


def plot_panel(panel, column, ax=None, label=None, window=15):
    if ax is None:
        fig, ax = plt.subplots()

    panel_mean = panel.mean(0).rolling(window).mean()
    panel_min = panel.min(0).rolling(window).mean()
    panel_max = panel.max(0).rolling(window).mean()

    ax.plot(panel_mean['Steps'], panel_mean[column], label=label)
    ax.fill_between(panel_mean['Steps'], panel_min[column], panel_max[column], alpha=0.3)
    ax.set_title(column)
    ax.legend()


def plot_logs(log_dir, tags):
    fig = None
    for tag in tags:
        panel = get_logs(log_dir=log_dir, tag=tag)
        columns = list(panel[0].columns)

        if fig is None:
            fig, axs = plt.subplots(len(columns), 1, figsize=(8, len(columns) * 8))

        for i, column in enumerate(columns):
            plot_panel(panel, column=column, ax=axs.flat[i], label=tag)
