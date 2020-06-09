import matplotlib.pyplot as plt
import numpy as np


def plot_ced():
    ced_single_path = '../evaluation/evaluation_results_single.txt'
    ced_synth_path = '../evaluation/evaluation_results_synth.txt'
    ced_two_path = '../evaluation/evaluation_results_two.txt'

    ced_single = np.loadtxt(ced_single_path)[:-1]
    ced_synth = np.loadtxt(ced_synth_path)[:-1]
    ced_two = np.loadtxt(ced_two_path)
    print(np.mean(ced_single), np.mean(ced_synth), np.mean(ced_two))
    return
    ced_data = np.zeros((20, 3))
    for i, _ in enumerate(ced_data):
        ced_data[i] = [ced_single[i], ced_synth[i], ced_two[i]]  # [ced_two[i], ced_single[i], ced_synth[i]]

    fig, ax = plt.subplots()
    plt.hist(ced_data, np.arange(0, 0.05, step=0.0001), histtype='step',
             density=True, cumulative=True, linewidth=1.5,
             label=['PRN', 'synthetically trained network', 'transfer trained network'],
             color=['red', 'lightblue',
                    'blue'])  # ['transfer trained network', 'PRN', 'synthetically trained network'])
    # plt.plot(ced_data, val_losses)
    # plt.yticks(np.arange(0, 0.0025, step = 0.0005))
    ax.set_ylabel('percentage of images')
    ax.set_xlabel('NME')
    plt.legend(loc='lower right')
    plt.ylim([0., 1.01])
    plt.xlim([0., 0.048])

    # plt.text(60, 0.0004, 'lowest loss: 0.000082')
    plt.savefig('ced.png')


if __name__ == '__main__':
    plot_ced()
