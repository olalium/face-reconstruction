import matplotlib.pyplot as plt
import numpy as np


def plot_loss():
    training_log_path = '../training/training_logs/log_2020_06_01_21_09_55.txt'
    # training_log_path = '../training/training_logs/log_2020_06_02_22_10_11.txt'
    plot_type = 'synthetic'  # '300w-lp' # 'synthetic' #

    with open(training_log_path) as f:
        lines = f.readlines()

    epochs = []
    val_losses = []
    for line in lines:
        val_losses.append(float(line[70:78]))
        epochs.append(int(line[28:31]))

    # plt.title('validation loss for transfer training')
    plt.title('validation loss for ' + plot_type + ' data training')
    plt.plot(epochs, val_losses)
    plt.ylabel('validation loss')
    plt.xlabel('epoch')
    plt.xticks(np.arange(0, 100, step=10))
    # plt.yticks(np.arange(0, 0.0025, step = 0.0005))
    # plt.ylim([0., 0.02])
    plt.text(60, 0.0004, 'lowest loss: 0.000082')
    plt.savefig('real_val_' + plot_type + '.png')


if __name__ == '__main__':
    plot_loss()
