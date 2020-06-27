import os
import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilone = 1.0
        self.epsilone_decay = 0.995
        self.epsilone_min = 0.01
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(Dense(24, input_dim = self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilone:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        X = []
        Y = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            X.append(state[0])
            Y.append(target_f[0])

        print(X)
        print("\n\n\n\n\n")
        print(Y)
        self.model.fit(np.array(X), np.array(Y), epochs=batch_size, verbose=0)

        if self.epsilone > self.epsilone_min:
            self.epsilone *= self.epsilone_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def game():
    env = gym.make('CartPole-v0')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    batch_size = 32
    n_episodes = 1001
    output_dir = 'cartpole/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    agent = DQNAgent(state_size, action_size)

    done = False
    for e in range(n_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(5000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else 10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode {}/{}, score:{}, e: {:.2}".format(e, n_episodes, time, agent.epsilone))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if e % 100 == 0:
            agent.save(output_dir + "weights_" + '{:04d}'.format(e) + '.hdf5')
    env.close()


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    game()
