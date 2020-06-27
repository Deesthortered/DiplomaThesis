import numpy as np
import tensorflow.keras as kr
import random
from collections import deque


class DQNAgent:
    def __init__(self, environment, log_file):
        self.log_file = log_file
        self.is_logging = False

        self.state_size = len(environment.observation_space.high)
        self.action_size = environment.action_space.n

    def init(self):
        self.memory = deque(maxlen=5000)

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995

        self.gamma = 0.9
        self.learning_rate = 0.01

        self.model = kr.models.Sequential()
        self.model.add(kr.layers.Dense(self.state_size * 3, input_dim=self.state_size, activation=kr.activations.linear))
        self.model.add(kr.layers.Dense(self.state_size * 3, activation=kr.activations.linear))
        self.model.add(kr.layers.Dense(self.state_size * 4, activation=kr.activations.linear))
        self.model.add(kr.layers.Dense(self.action_size, activation=kr.activations.linear))
        self.model.compile(loss=kr.losses.mse,
                           optimizer=kr.optimizers.Adam(lr=self.learning_rate),
                           metrics=['accuracy'])

    def close(self):
        self.memory.clear()
        self.model = None
        kr.backend.clear_session()

    def act(self, state):
        self.log("*** agent acting: start")
        rand_val = np.random.rand()
        self.log("  random_value = " + str(rand_val) + "; epsilon = " + str(self.epsilon))
        if rand_val <= self.epsilon:
            self.log("   random action! :D ")
            return random.randrange(self.action_size)

        state = np.reshape(state, [1, len(state)])
        self.log("state = " + str(state))

        act_values = self.model.predict(state)
        self.log("Q values = " + str(act_values))
        self.log("action = " + str(np.argmax(act_values[0])))
        self.log("*** agent acting: end")
        return np.argmax(act_values[0])

    def train(self, batch_size):
        self.log("*** agent training: start")
        self.log("   available frames = " + str(len(self.memory)) + "; parameter_batch_size = " + str(batch_size))
        if len(self.memory) > batch_size:
            batch = random.sample(self.memory, batch_size)

            self.log("   got batch of frames before iteration")
            for state, next_state, action, reward, done in batch:
                self.log("   *** training iteration: start")
                self.log("       * data:")
                self.log("       state      = " + str(state))
                self.log("       next_state = " + str(next_state))
                self.log("                  ~ " + str([i - j for i, j in zip(next_state, state)]))
                self.log("       action     = " + str(action))
                self.log("       reward     = " + str(reward))
                self.log("       done       = " + str(done))
                self.log("")

                Q_values = self.model.predict(state)
                Q_new_values = self.model.predict(next_state)
                self.log("       Q_values     = " + str(Q_values) + " - (" + str(np.argmax(Q_values)) + ", " + str(np.amax(Q_values)) + ")")
                self.log("       Q_new_values = " + str(Q_new_values) + " - (" + str(np.argmax(Q_new_values)) + ", " + str(np.amax(Q_new_values)) + ")")

                target = reward
                if done:
                    self.log("       target = reward")
                    self.log("       target = " + str(reward))
                if not done:
                    target = (reward + self.gamma * np.amax(Q_new_values))
                    self.log("       target = (reward + self.gamma * np.amax(Q_new_values))")
                    self.log("       target = " + str(reward) + " + " + str(self.gamma) + " * " + str(np.amax(Q_new_values)) + " = " + str(target))

                self.log("       Q_values before = " + str(Q_values))
                Q_values[0][action] = target
                self.log("       Q_values after  = " + str(Q_values))

                self.log("       got (X,Y):")
                self.log("         X = " + str(state[0]))
                self.log("         Y = " + str(Q_values[0]))
                self.log("         Start fitting...")
                self.model.fit([[state[0]]], [[Q_values[0]]], epochs=1, verbose=0, shuffle=False)

                self.log("   *** training iteration: end")

            self.log("   epsilon decaying:")
            self.log("      old = :" + str(self.epsilon))
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.log("      new = :" + str(self.epsilon))
        else:
            self.log("    not enough")
        self.log("*** agent training: end")

    def remember(self, state, next_state, action, reward, done):
        self.log("*** agent training: start")
        state = np.reshape(state, [1, len(state)])
        next_state = np.reshape(next_state, [1, len(next_state)])
        self.memory.append((state, next_state, action, reward, done))
        self.log("    state      = " + str(state))
        self.log("    next_state = " + str(next_state))
        self.log("    action     = " + str(action))
        self.log("    reward     = " + str(reward))
        self.log("    done       = " + str(done))
        self.log("*** agent training: end")

    def load_weights(self, path):
        self.log("*** loading weights from " + path)
        print("*** loading weights from " + path)

        self.model.load_weights(path)
        self.log(str(self.model.get_weights()))

    def save_weights(self, path):
        self.log("*** saving weights to " + path + '.h5')
        print("*** saving weights to " + path + '.h5')

        self.model.save_weights(path + '.h5')
        self.log(str(self.model.get_weights()))

    def log(self, text):
        if self.is_logging:
            self.log_file.write(text + "\n")
