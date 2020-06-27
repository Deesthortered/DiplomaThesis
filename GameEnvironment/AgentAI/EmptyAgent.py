# If just you want agent that will do nothing


class EmptyAgent:
    def __init__(self, environment, log_file):
        self.is_logging = False
        pass

    def init(self):
        pass

    def close(self):
        pass

    def act(self, state):
        return 0

    def train(self, batch_size):
        pass

    def remember(self, state, new_state, action, reward, done):
        pass

    def load_weights(self, path):
        pass

    def save_weights(self, path):
        pass
