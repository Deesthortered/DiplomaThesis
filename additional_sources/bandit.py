import numpy as np
import matplotlib.pyplot as plt


def game(bandit_probability_array, memory_array, strategy_function):
    general_reward = 0.0
    range_size = 5

    for i in range(range_size):
        action_number = strategy_function(memory_array)
        reward = 1 if np.random.rand() <= bandit_probability_array[action_number] else 0
        memory_array[action_number] += reward
        general_reward += reward

    return memory_array, general_reward


def greedy_strategy(memory_array):
    return np.argmax(memory_array)


def epsilon_greedy_strategy(memory_array):
    epsilon = 0.2
    if np.random.rand() <= epsilon:
        return np.random.randint(0, len(memory_array))
    return np.argmax(memory_array)


def epsilon_greedy_strategy1(memory_array):
    epsilon = 0.1
    if np.random.rand() <= epsilon:
        return np.random.randint(0, len(memory_array))
    return np.argmax(memory_array)


def draw(x, y, label_text):
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

    cutoff = 2000 * 1.1
    fs = 50000 * 10
    y_smooth = butter_lowpass_filter(y, cutoff, fs)
    plot_P_vs_t = plt.subplot(111)
    plot_P_vs_t.set_ylabel('Сумарна нагорода', labelpad=6)
    plot_P_vs_t.set_xlabel('Кількість ігор', labelpad=6)
    plot_P_vs_t.plot(x, y_smooth, linewidth=1.0, label=label_text)
    plot_P_vs_t.legend(loc='best', frameon=False)


if __name__ == "__main__":
    global_bandit_probability_array = [0.1, 0.8, 0.2, 0.15, 0.1, 0.05, 0.2, 0.15, 0.01, 0.5]
    size = len(global_bandit_probability_array)
    global_memory_array = np.zeros(size)

    X  = []
    Y  = []
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    game_count = 2000

    current_memory_array = global_memory_array.copy()
    for i in range(2000):
        current_memory_array, current_general_reward = game(global_bandit_probability_array,
                                                            current_memory_array,
                                                            epsilon_greedy_strategy)
        X.append(i)
        Y.append(current_general_reward)

    current_memory_array = global_memory_array.copy()
    for i in range(2000):
        current_memory_array, current_general_reward = game(global_bandit_probability_array,
                                                            current_memory_array,
                                                            epsilon_greedy_strategy1)
        X2.append(i)
        Y2.append(current_general_reward)


    for i in range(2000):
        _, current_general_reward = game(global_bandit_probability_array,
                                                            global_memory_array.copy(),
                                                            greedy_strategy)
        X1.append(i)
        Y1.append(current_general_reward)


    draw(X1, Y1, "ε = 0")
    draw(X, Y, "ε = 0.2")
    draw(X2, Y2, "ε = 0.1")
    plt.show()
    plt.close()
