import gym
import numpy as np
import matplotlib.pyplot as plt


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 100000
PRINT_CYCLE = 1000


def game_Q_learning():
    env = gym.make("MountainCar-v0")

    epsilone = 0.5
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // 2
    epsilone_decay_value = epsilone/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
    print("epsilone_decay_value = ", epsilone_decay_value)

    print("Количество переменных = ", env.observation_space)
    print("Верхние границы = ", env.observation_space.high)
    print("Нижние границы = ", env.observation_space.low)

    DISCRETE_SIZE = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_SIZE
    q_table = np.random.uniform(low = -2, high = 0, size=(DISCRETE_SIZE + [env.action_space.n]))
    print("Шаг дискретности: ", discrete_os_win_size)
    print("Размерность Q-таблицы: ", q_table.shape)

    def get_discrete_state(state):
        current_discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(current_discrete_state.astype(np.int))

    X = []
    Y = []
    success_count = 0

    for episode in range(EPISODES):
        done = False
        discrete_state = get_discrete_state(env.reset())

        render = False
        if episode % PRINT_CYCLE == 0:
            print(episode, epsilone)
            render = True

        while not done:
            if np.random.random() > 0.1:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)
            new_state, reward, done, info = env.step(action)
            new_discrete_state = get_discrete_state(new_state)

            if False and render:
                env.render()

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action, )]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                q_table[discrete_state + (action, )] = new_q
            elif new_state[0] >= env.goal_position:
                q_table[discrete_state + (action, )] = 0
                success_count += 1

            discrete_state = new_discrete_state

        X.append(episode)
        Y.append(success_count / max(1, episode))

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilone -= epsilone_decay_value

    env.close()
    return X, Y


def game_sarsa():
    env = gym.make("MountainCar-v0")

    epsilone = 0.5
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // 2
    epsilone_decay_value = epsilone/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
    print("epsilone_decay_value = ", epsilone_decay_value)

    print("Количество переменных = ", env.observation_space)
    print("Верхние границы = ", env.observation_space.high)
    print("Нижние границы = ", env.observation_space.low)

    DISCRETE_SIZE = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_SIZE
    q_table = np.random.uniform(low = -2, high = 0, size=(DISCRETE_SIZE + [env.action_space.n]))
    print("Шаг дискретности: ", discrete_os_win_size)
    print("Размерность Q-таблицы: ", q_table.shape)

    def get_discrete_state(state):
        current_discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(current_discrete_state.astype(np.int))

    action = None
    new_action = np.random.randint(0, env.action_space.n)

    X = []
    Y = []
    success_count = 0

    for episode in range(EPISODES):
        done = False
        discrete_state = get_discrete_state(env.reset())

        render = False
        if episode % PRINT_CYCLE == 0:
            print(episode, epsilone)
            render = True

        while not done:
            action = new_action
            new_state, reward, done, info = env.step(action)
            new_discrete_state = get_discrete_state(new_state)

            if False and render:
                env.render()

            if not done:
                if np.random.random() > 0.1:
                    new_action = np.argmax(q_table[discrete_state])
                else:
                    new_action = np.random.randint(0, env.action_space.n)
                future_q = np.max(q_table[new_discrete_state + (new_action, )])

                current_q = q_table[discrete_state + (action, )]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * future_q)
                q_table[discrete_state + (action, )] = new_q
            elif new_state[0] >= env.goal_position:
                q_table[discrete_state + (action, )] = 0
                success_count += 1

            discrete_state = new_discrete_state

        X.append(episode)
        Y.append(success_count / max(1, episode))

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilone -= epsilone_decay_value
            epsilone = max(0, epsilone)

    env.close()
    return X, Y


def monte_carlo():
    env = gym.make("MountainCar-v0")

    epsilone = 0.5
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // 2
    epsilone_decay_value = epsilone/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
    print("epsilone_decay_value = ", epsilone_decay_value)

    print("Количество переменных = ", env.observation_space)
    print("Верхние границы = ", env.observation_space.high)
    print("Нижние границы = ", env.observation_space.low)

    DISCRETE_SIZE = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_SIZE
    q_table = np.random.uniform(low = -2, high = 0, size=(DISCRETE_SIZE + [env.action_space.n]))
    print("Шаг дискретности: ", discrete_os_win_size)
    print("Размерность Q-таблицы: ", q_table.shape)

    def get_discrete_state(state):
        current_discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(current_discrete_state.astype(np.int))

    X = []
    Y = []
    success_count = 0

    episode_memory = []

    for episode in range(EPISODES):
        episode_memory.clear()
        done = False
        discrete_state = get_discrete_state(env.reset())

        render = False
        if episode % PRINT_CYCLE == 0:
            print(episode, epsilone)
            render = True

        while not done:
            if np.random.random() > 0.1:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, info = env.step(action)
            new_discrete_state = get_discrete_state(new_state)

            if False and render:
                env.render()

            if not done:
                episode_memory.append((episode, discrete_state, action, reward, new_discrete_state))
            else:
                win = new_state[0] >= env.goal_position
                if win:
                    print("Success on", episode)
                for i_episode, i_discrete_state, i_action, i_reward, i_new_discrete_state in episode_memory:
                    q_table[i_discrete_state + (i_action, )] += (1 if win else -1)
                success_count += 1 if win else 0

            discrete_state = new_discrete_state

        X.append(episode)
        Y.append(success_count / max(1, episode))

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilone -= epsilone_decay_value
            epsilone = max(0, epsilone)

    env.close()
    return X, Y


def actor_critic():
    env = gym.make("MountainCar-v0")

    epsilone = 0.5
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // 2
    epsilone_decay_value = epsilone/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
    print("epsilone_decay_value = ", epsilone_decay_value)

    print("Количество переменных = ", env.observation_space)
    print("Верхние границы = ", env.observation_space.high)
    print("Нижние границы = ", env.observation_space.low)

    DISCRETE_SIZE = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_SIZE
    q_table = np.random.uniform(low = -2, high = 0, size=(DISCRETE_SIZE + [env.action_space.n]))
    v_table = np.random.uniform(low = -2, high = 0, size=DISCRETE_SIZE)
    print("Шаг дискретности: ", discrete_os_win_size)
    print("Размерность Q-таблицы: ", q_table.shape)
    print("Размерность V-таблицы: ", v_table.shape)

    def get_discrete_state(state):
        current_discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(current_discrete_state.astype(np.int))

    X = []
    Y = []
    success_count = 0

    for episode in range(EPISODES):
        done = False
        discrete_state = get_discrete_state(env.reset())

        render = False
        if episode % PRINT_CYCLE == 0:
            print(episode, epsilone)
            render = True

        while not done:
            if np.random.random() > 0.1:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)
            new_state, reward, done, info = env.step(action)
            new_discrete_state = get_discrete_state(new_state)

            if False and render:
                env.render()

            if not done:
                current_v = v_table[discrete_state]
                next_v = v_table[new_discrete_state]
                new_v = (1 - LEARNING_RATE) * current_v + LEARNING_RATE * (reward + DISCOUNT * next_v)
                v_table[discrete_state] = new_v

                current_q = q_table[discrete_state + (action, )]
                advantage = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (new_v - current_v)
                q_table[discrete_state + (action, )] = advantage
            elif new_state[0] >= env.goal_position:
                q_table[discrete_state + (action, )] = 0
                success_count += 1

            discrete_state = new_discrete_state

        X.append(episode)
        Y.append(success_count / max(1, episode))

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilone -= epsilone_decay_value

    env.close()
    return X, Y


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

    cutoff = 2000
    fs = 50000
    y_smooth = butter_lowpass_filter(y, cutoff, fs)
    plot_P_vs_t = plt.subplot(111)
    plot_P_vs_t.set_ylabel('Відсоток успішних ігор', labelpad=6)
    plot_P_vs_t.set_xlabel('Кількість ігор', labelpad=6)
    plot_P_vs_t.plot(x, y_smooth, linewidth=1.0, label=label_text)
    plot_P_vs_t.legend(loc='best', frameon=False)


if __name__ == "__main__":
    X1, Y1 = game_Q_learning()
    X2, Y2 = game_sarsa()
    X3, Y3 = monte_carlo()
    X4, Y4 = actor_critic()

    draw(X1, Y1, "Q-learning")
    draw(X2, Y2, "SARSA")
    draw(X3, Y3, "Monte-Carlo")
    draw(X4, Y4, "Actor-Critic")
    plt.show()
    plt.close()
