import numpy as np
import tensorflow.keras as kr
import random
import os


class A2CAgent:
    def __init__(self, environment, log_file):
        self.state_size = len(environment.observation_space.high)
        self.action_size = environment.action_space.n

        self.log_file = log_file
        self.is_logging = False

    def init(self):
        self.epsilon       = 1.0
        self.epsilon_min   = 0.1
        self.epsilon_decay = 0.9995

        self.gamma = 0.75
        self.gamma_max = 0.95
        self.gamma_decay = 1.0001
        self.learning_rate = 0.001

        self.actor_model = kr.models.Sequential()
        self.actor_model.add(kr.layers.Dense(self.state_size * 20, input_dim=self.state_size, activation=kr.activations.relu))
        self.actor_model.add(kr.layers.Dense(self.state_size * 16, activation=kr.activations.relu))
        self.actor_model.add(kr.layers.Dense(self.state_size * 8, activation=kr.activations.relu))
        self.actor_model.add(kr.layers.Dense(self.state_size * 8, activation=kr.activations.relu))
        self.actor_model.add(kr.layers.Dense(self.action_size, activation=kr.activations.relu))
        self.actor_model.compile(loss=kr.losses.mse,
                                 optimizer=kr.optimizers.Adam(lr=self.learning_rate),
                                 metrics=['accuracy'])

        self.critic_model = kr.models.Sequential()
        self.critic_model.add(kr.layers.Dense(self.state_size * 20, input_dim=self.state_size, activation=kr.activations.relu))
        self.critic_model.add(kr.layers.Dense(self.state_size * 16, activation=kr.activations.relu))
        self.critic_model.add(kr.layers.Dense(self.state_size * 8, activation=kr.activations.relu))
        self.critic_model.add(kr.layers.Dense(self.state_size * 8, activation=kr.activations.relu))
        self.critic_model.add(kr.layers.Dense(1, activation=kr.activations.linear))
        self.critic_model.compile(loss=kr.losses.mse,
                                  optimizer=kr.optimizers.Adam(lr=self.learning_rate),
                                  metrics=['accuracy'])

    def close(self):
        self.actor_model  = None
        self.critic_model = None
        kr.backend.clear_session()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = np.reshape(state, [1, len(state)])
        act_values = self.actor_model.predict(state)
        return np.argmax(act_values[0])

    def train(self, batch_size):
        pass

    def remember(self, state, new_state, action, reward, done):
        self.log("Start remembering!!!!!!")

        state = np.reshape(state, [1, len(state)])
        new_state = np.reshape(new_state, [1, len(new_state)])
        V_value = self.critic_model.predict(state)[0][0]
        V_new_value = self.critic_model.predict(new_state)[0][0]
        Q_values = self.actor_model.predict(state)
        Q_new_values = self.actor_model.predict(new_state)

        self.log("state        = " + str(state))
        self.log("new_state    = " + str(new_state))
        self.log("             ~ " + str([i - j for i, j in zip(new_state, state)]))

        self.log("V_value      = " + str(V_value))
        self.log("V_new_value  = " + str(V_new_value))
        self.log("             ~ " + str(V_new_value - V_value))

        self.log("Q_values     = " + str(Q_values))
        self.log("Q_new_values = " + str(Q_new_values))
        self.log("             ~ " + str([i - j for i, j in zip(Q_new_values, Q_values)]))

        self.log("v_target = reward + self.gamma * V_new_value")
        self.log("v_target = " + str(reward) + " + " + str(self.gamma) + " * " + str(V_new_value))
        v_target = reward + self.gamma * V_new_value
        self.log("v_target = " + str(v_target))
        if done:
            self.log("but it's done, so v_target = reward = " + str(reward))
            v_target = reward
        self.log("")

        self.log("fitting critic: state = " + str(state) + "; [[v_target]] = " + str([[v_target]]))
        self.critic_model.fit(state, [[v_target]], epochs=1, verbose=0)
        self.log("")

        self.log("advantage = v_target - V_value")
        self.log("advantage = " + str(v_target) + " - " + str(V_value))
        advantage = v_target - V_value
        self.log("advantage = " + str(advantage))

        self.log("q_target = advantage + self.gamma * np.amax(Q_new_values[0])")
        self.log("q_target = " + str(advantage) + " + " + str(self.gamma) + " * max(" + str(Q_new_values[0]) + ")")
        q_target = advantage + self.gamma * np.amax(Q_new_values[0])
        self.log("q_target = " + str(q_target))
        if done:
            self.log("but it's done, so q_target = advantage = " + str(advantage))
            q_target = advantage

        Q_values[0][action] = q_target

        self.log("fitting actor: state = " + str(state) + "; Q_values = " + str(Q_values))
        self.actor_model.fit(state, Q_values, epochs=1, verbose=0)

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # gamma decay
        if self.gamma < self.gamma_max:
            self.gamma *= self.gamma_decay

        self.log("self.epsilon = " + str(self.epsilon))
        self.log("self.gamma = " + str(self.gamma))

        self.log("End remembering !!!!!!!!!!!!!!!")
        self.log("")
        self.log("")
        self.log("")
        self.log("")
        self.log("")
        self.log("")

    def load_weights(self, path):
        path = path[:path.rfind('/')]
        self.log("*** loading weights to " + path + "/(actor/critic).h5")
        print("*** loading weights to " + path + "/(actor/critic).h5")

        self.actor_model.load_weights(path + '/actor.h5')
        self.critic_model.load_weights(path + '/critic.h5')
        self.log(str(self.actor_model.get_weights()))
        self.log(str(self.critic_model.get_weights()))

    def save_weights(self, path):
        self.log("*** saving weights to " + path + "/(actor/critic).h5")
        print("*** saving weights to " + path + "/(actor/critic).h5")

        if not os.path.exists(path):
            os.mkdir(path)
        self.actor_model.save_weights(path + '/actor.h5')
        self.critic_model.save_weights(path + '/critic.h5')
        self.log(str(self.actor_model.get_weights()))
        self.log(str(self.critic_model.get_weights()))

    def log(self, text):
        if self.is_logging:
            self.log_file.write(text + "\n")
