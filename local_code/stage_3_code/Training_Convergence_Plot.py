import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


def save_training_loss_curve(fold_loss_histories, plot_file_path, title, line_color='tab:blue'):
    # save an averaged training-loss curve across folds.


    if not fold_loss_histories:
        return None

    min_length = min(len(history) for history in fold_loss_histories)
    if min_length == 0:
        return None

    trimmed_histories = np.array([history[:min_length] for history in fold_loss_histories], dtype=float)
    mean_loss = np.mean(trimmed_histories, axis=0)

    epochs = np.arange(1, min_length + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mean_loss, label='Average Training Loss', color=line_color, linewidth=2)
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss Value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_parent = os.path.dirname(os.path.abspath(plot_file_path))
    os.makedirs(plot_parent, exist_ok=True)
    plt.savefig(plot_file_path)
    plt.close()
    print('Saved training convergence plot to:', plot_file_path)
    return plot_file_path