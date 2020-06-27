import pickle
import matplotlib.pyplot as plt


if __name__ == "__main__":
    def smooth(y, cutoff=2000, fs=50000 * 0.5):
        from scipy.signal import butter, lfilter

        def butter_lowpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        def butter_lowpass_filter(data, cutoff, fs, order=5):
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = lfilter(b, a, data)
            return y

        y_smooth = butter_lowpass_filter(y, cutoff, fs)
        return y_smooth

    folders  = [
        # ("experiments/1.A2C/", "A2C", 2000, 50000 * 0.5),
        # ("experiments/1.DQN/", "DQN", 2000, 50000 * 0.5),
        #
        # ("experiments/2.A2C/", "A2C", 2000, 50000 * 0.5),
        # ("experiments/2.DQN/", "DQN", 2000, 50000 * 0.5),
        #
        # ("experiments/3/", "A2C R- S1", 2000, 50000 * 0.5),
        #
        # ("experiments/4/", "A2C R- S2", 2000, 50000 * 0.5),
        #
        # ("experiments/5/", "A2C R+ S1", 2000, 50000 * 0.5),
        #
        # ("experiments/6/", "A2C R+ S2", 2000, 50000 * 0.5),
        #
        ("experiments/7/", "A2C/DQN R+ S1", 2000, 50000 * 0.5),

        ("experiments/8/", "A2C/DQN R+ S2", 2000, 50000 * 0.5),
    ]
    max_x_length = 300
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))

    for cur_folder in folders:
        with open(cur_folder[0] + "/episode_n_catchcount", 'rb') as f:
            episode_n_catchcount = pickle.load(f)
        with open(cur_folder[0] + "/episode_n_catch_time", 'rb') as f:
            episode_n_catch_time = pickle.load(f)
        with open(cur_folder[0] + "/episode_n_reward_victim", 'rb') as f:
            episode_n_reward_victim = pickle.load(f)
        with open(cur_folder[0] + "/episode_n_reward_predator", 'rb') as f:
            episode_n_reward_predator = pickle.load(f)

        axs[0, 0].plot(
            [i for (i, j) in episode_n_catchcount],
            smooth([(j * 100) for (i, j) in episode_n_catchcount], cur_folder[2], cur_folder[3]),
            label=cur_folder[1]
        )
        axs[0, 0].set_title('Кількість зустрічів')
        axs[0, 0].set_xlabel('Номер епізоду')
        axs[0, 0].set_ylabel('% епізодів')
        axs[0, 0].legend(loc="best")
        axs[0, 0].axis([0, max_x_length, 0, 120])

        axs[0, 1].plot(
            [i for (i, j) in episode_n_catch_time],
            smooth([j for (i, j) in episode_n_catch_time], cur_folder[2], cur_folder[3]),
            label=cur_folder[1])
        axs[0, 1].set_title('Тривалість епізоду')
        axs[0, 1].set_xlabel('Номер епізоду')
        axs[0, 1].set_ylabel('Кількість кадрів')
        axs[0, 1].legend(loc="best")
        axs[0, 1].axis([0, max_x_length, 0, 500])

        axs[1, 0].plot(
            [i for (i, j) in episode_n_reward_victim],
            smooth([j for (i, j) in episode_n_reward_victim], cur_folder[2], cur_folder[3]),
            label=cur_folder[1])
        axs[1, 0].set_title('Сумарна нагорода за епізод (жертва)')
        axs[1, 0].set_xlabel('Номер епізоду')
        axs[1, 0].set_ylabel('Сумарна нагорода')
        axs[1, 0].legend(loc="best")
        axs[1, 0].axis([0, max_x_length, -30, 30])

        axs[1, 1].plot(
            [i for (i, j) in episode_n_reward_predator],
            smooth([j for (i, j) in episode_n_reward_predator], cur_folder[2], cur_folder[3]),
            label=cur_folder[1])
        axs[1, 1].set_title('Сумарна нагорода за епізод (хижак)')
        axs[1, 1].set_xlabel('Номер епізоду')
        axs[1, 1].set_ylabel('Сумарна нагорода')
        axs[1, 1].legend(loc="best")
        axs[1, 1].axis([0, max_x_length, -30, 30])

    plt.show()
    plt.close()
