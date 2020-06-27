import random
import numpy as np
import gym


class EnvironmentCore:
    def __init__(self, env_renderer, log_file):
        self.env_renderer = env_renderer
        self.log_file = log_file
        self.is_logging = False

        self.action_space      = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(low=np.array([1.0,
                                                              -1.0, -1.0,
                                                              -1.0, -1.0,
                                                              -1.0, -1.0,
                                                              -1.0, -1.0,
                                                              #-1.0, -1.0,
                                                              #-1.0, -1.0
                                                              ]),
                                                high=np.array([1.0,
                                                               1.0, 1.0,
                                                               1.0, 1.0,
                                                               1.0, 1.0,
                                                               1.0, 1.0,
                                                               #1.0, 1.0,
                                                               #1.0, 1.0
                                                               ]),
                                                dtype=np.float32)
        self.reset(False)

    def reset(self, random_position=False, predator_pos=[40.0, 0.0], victim_pos=[-40.0, 0.0]):
        self.FIELD_SIZE = 700  # base = 700
        self.GAME_STEPS = 500
        self.turn_timer = False
        self.game_current_step = 0

        self.MIN_CATCH_DISTANCE = 50
        self.NATURE_DECELERATION = 0.005
        self.VELOCITY_EPS = 0.015

        self.PREDATOR_REWARD_KOEFF = 0.01
        self.FINISH_PREDATOR_REWARD = 10
        self.FINISH_PREDATOR_PUNISHMENT = -5
        self.PREDATOR_MAX_ACCELERATION_VALUE = 5
        self.given_predator_direction_vector = [0.0, 0.0]
        self.PREDATOR_MAX_VELOCITY_VALUE = 30
        self.predator_velocity_vector = np.array([0.0, 0.0])
        if random_position:
            self.predator_pos = [
                random.randint(-self.FIELD_SIZE // 2, self.FIELD_SIZE // 2),
                random.randint(-self.FIELD_SIZE // 2, self.FIELD_SIZE // 2)
            ]
        else:
            self.predator_pos = predator_pos
        self.predator_pos_prev = self.predator_pos

        self.VICTIM_REWARD_KOEFF = 0.01
        self.FINISH_VICTIM_REWARD = 10
        self.FINISH_VICTIM_PUNISHMENT = -10
        self.VICTIM_MAX_ACCELERATION_VALUE = 20
        self.given_victim_direction_vector = [0.0, 0.0]
        self.VICTIM_MAX_VELOCITY_VALUE = 15
        self.victim_velocity_vector = np.array([0.0, 0.0])
        if random_position:
            self.victim_pos = [
                random.randint(-self.FIELD_SIZE // 2, self.FIELD_SIZE // 2),
                random.randint(-self.FIELD_SIZE // 2, self.FIELD_SIZE // 2)
            ]
            while np.linalg.norm(np.array(self.predator_pos) - np.array(self.victim_pos)) < self.MIN_CATCH_DISTANCE * 2:
                self.victim_pos = [
                    random.randint(-self.FIELD_SIZE // 2, self.FIELD_SIZE // 2),
                    random.randint(-self.FIELD_SIZE // 2, self.FIELD_SIZE // 2)
                ]
        else:
            self.victim_pos = victim_pos
        self.victim_pos_prev = self.victim_pos

        self.predator_is_out = False
        self.victim_is_out = False

        return self._get_observation()

    def step(self, action_dictionary):
        self.given_predator_direction_vector = self._get_mapped_action(action_dictionary["agent_0"])
        self.given_victim_direction_vector = self._get_mapped_action(action_dictionary["agent_1"])

        if self.__is_episode_done():
            return self._get_observation(), self._get_rewards(), self._get_dones(), self._get_infos()

        if self.turn_timer:
            self.game_current_step += 1

        predator_acceleration_vector = np.array([i * self.PREDATOR_MAX_ACCELERATION_VALUE for i in self.given_predator_direction_vector])
        victim_acceleration_vector = np.array([i * self.VICTIM_MAX_ACCELERATION_VALUE for i in self.given_victim_direction_vector])

        self.predator_velocity_vector += predator_acceleration_vector
        self.victim_velocity_vector += victim_acceleration_vector

        if self.NATURE_DECELERATION != 0:
            predator_velocity_val = np.linalg.norm(self.predator_velocity_vector)
            victim_velocity_val = np.linalg.norm(self.victim_velocity_vector)

            if predator_velocity_val > self.VELOCITY_EPS:
                predator_velocity_direction = self.predator_velocity_vector / predator_velocity_val
                self.predator_velocity_vector -= predator_velocity_direction * self.NATURE_DECELERATION
            else:
                self.predator_velocity_vector = np.array([0.0, 0.0])

            if victim_velocity_val > self.VELOCITY_EPS:
                victim_velocity_direction = self.victim_velocity_vector / victim_velocity_val
                self.victim_velocity_vector -= victim_velocity_direction * self.NATURE_DECELERATION
            else:
                self.victim_velocity_vector = np.array([0.0, 0.0])

        if np.linalg.norm(self.predator_velocity_vector) > self.PREDATOR_MAX_VELOCITY_VALUE:
            self.predator_velocity_vector = (self.predator_velocity_vector / np.linalg.norm(self.predator_velocity_vector)) * self.PREDATOR_MAX_VELOCITY_VALUE
        if np.linalg.norm(self.victim_velocity_vector) > self.VICTIM_MAX_VELOCITY_VALUE:
            self.victim_velocity_vector = (self.victim_velocity_vector / np.linalg.norm(self.victim_velocity_vector)) * self.VICTIM_MAX_VELOCITY_VALUE

        self.predator_pos_prev = self.predator_pos
        self.victim_pos_prev = self.victim_pos
        self.predator_pos = list(np.array(self.predator_pos) + self.predator_velocity_vector)
        self.victim_pos = list(np.array(self.victim_pos) + self.victim_velocity_vector)

        # Border handling
        # self.predator_is_out = self.__agent_is_out_of_the_field(self.predator_pos)
        # self.victim_is_out = self.__agent_is_out_of_the_field(self.victim_pos)

        return self._get_observation(), self._get_rewards(), self._get_dones(), self._get_infos()

    def render(self):
        self.env_renderer.draw([round(i) for i in self.predator_pos],
                               [round(i) for i in self.victim_pos],
                               self.given_predator_direction_vector, self.given_victim_direction_vector,
                               np.linalg.norm(self.given_predator_direction_vector),
                               np.linalg.norm(self.given_victim_direction_vector))

    def _get_mapped_action(self, action):
        if action == 0:
            return [0.0, 0.0]
        elif action == 1:
            return [-1.0, -1.0]
        elif action == 2:
            return [-1.0, 0.0]
        elif action == 3:
            return [-1.0, 1.0]
        elif action == 4:
            return [0.0, 1.0]
        elif action == 5:
            return [1.0, 1.0]
        elif action == 6:
            return [1.0, 0.0]
        elif action == 7:
            return [1.0, -1.0]
        elif action == 8:
            return [0.0, -1.0]
        else:
            print("Action mapping error!")
            raise ValueError

    def _get_observation(self):
        direction = np.array(self.predator_pos) - np.array(self.victim_pos)
        direction_velocity = self.predator_velocity_vector - self.victim_velocity_vector

        max_velocity = max(self.PREDATOR_MAX_VELOCITY_VALUE, self.VICTIM_MAX_VELOCITY_VALUE)

        normalized_direction = (direction / np.linalg.norm(direction))
        normalized_direction_velocity = (direction_velocity / max_velocity)
        normalized_position_predator = np.array(self.predator_pos) / float(self.FIELD_SIZE // 2)
        normalized_position_victim = np.array(self.victim_pos) / float(self.FIELD_SIZE // 2)
        normalized_velocity_predator = (self.predator_velocity_vector / max_velocity)
        normalized_velocity_victim = (self.victim_velocity_vector / max_velocity)

        # normalized_direction.tolist() + \
        result = [1] + \
                normalized_position_predator.tolist() + \
                normalized_position_victim.tolist() + \
                normalized_velocity_predator.tolist() + \
                normalized_velocity_victim.tolist()

        return {
            "agent_0": result,
            "agent_1": result,
        }

    def _get_rewards(self):
        if self.__is_agent_done():
            return {
                "agent_0": 5,
                "agent_1": -5,  # self.__reward_victim(),
            }
        return {
            "agent_0": self.__reward_predator(),
            "agent_1": self.__reward_victim(),
        }

    def _get_dones(self):
        return {
            "agent_0": self.__is_agent_done(),
            "agent_1": self.__is_agent_done(),
            "__all__": self.__is_episode_done(),
        }

    def _get_infos(self):
        return {
            "agent_0": {},
            "agent_1": {},
        }

    def __is_agent_done(self):
        return self.__distance_between_agents_cur_pos() <= self.MIN_CATCH_DISTANCE

    def __is_episode_done(self):
        return self.game_current_step >= self.GAME_STEPS or self.__is_agent_done()

    def __reward_predator(self):
        dist = (self.__distance_between_agents_prev_pos() - self.__distance_between_agents_cur_pos())
        return dist * self.PREDATOR_REWARD_KOEFF

    def __reward_victim(self):
        dist = (self.__distance_between_agents_prev_pos() - self.__distance_between_agents_cur_pos())
        return -dist * self.VICTIM_REWARD_KOEFF

    def __distance_between_agents_cur_pos(self):
        return np.linalg.norm(np.array(self.predator_pos) - np.array(self.victim_pos))

    def __distance_between_agents_prev_pos(self):
        return np.linalg.norm(np.array(self.predator_pos_prev) - np.array(self.victim_pos_prev))

    def __agent_is_out_of_the_field(self, position):
        return position[0] < -self.FIELD_SIZE/2 or position[0] > self.FIELD_SIZE/2 or \
               position[1] < -self.FIELD_SIZE/2 or position[1] > self.FIELD_SIZE/2

    def log(self, text):
        if self.is_logging:
            self.log_file.write(text + "\n")
