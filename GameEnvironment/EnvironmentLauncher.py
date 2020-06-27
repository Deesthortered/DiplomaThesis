import GameEnvironment.EnvironmentConfigurer
import GameEnvironment.EnvironmentCore
import GameEnvironment.EnvironmentRenderer
import os
from datetime import datetime

import tensorflow as tf
import numpy as np


class EnvironmentLauncher:
    def __init__(self):
        self.log_path = "./logs/"
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        self.log_file = open(self.log_path + str(datetime.now()).replace(" ", "_").replace(":", "-") + ".log", "w+")

        self.env_renderer = GameEnvironment.EnvironmentRenderer.EnvironmentRenderer(self.log_file)
        self.env_core = GameEnvironment.EnvironmentCore.EnvironmentCore(self.env_renderer, self.log_file)
        self.env_configurer = GameEnvironment.EnvironmentConfigurer.EnvironmentConfigurer(self.env_core,
                                                                                          self.env_renderer,
                                                                                          self.log_file)

        # Order is important, some things should be initialized earlier although dependencies usually appear earlier.
        # Therefore init() function should to initialize/define object but not his depended objects
        self.env_configurer.init()
        self.env_renderer.init()
        self.env_core.reset(False)

    def start(self):
        self.env_configurer.start_threads()
        self.log_file.flush()
        self.log_file.close()


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    launcher = EnvironmentLauncher()
    launcher.start()
