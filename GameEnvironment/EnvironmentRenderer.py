import pygame
import numpy as np
from collections import deque


class EnvironmentRenderer:
    def __init__(self, log_file):
        self.log_file = log_file
        self.is_logging = False

        self.draw_direction = True
        self.draw_trajectory = True

        self.FRAME_WIDTH = 700
        self.FRAME_HEIGHT = 700
        self.screen = None
        self.center_point = [0, 0]
        self.scale_koeff = 1

    def init(self):
        self.BACKGROUND_COLOR = pygame.Color("white")
        self.DIRECTION_VECTOR_MAX_LENGTH = 50
        self.DIRECTION_VECTOR_THICKNESS = 5
        self.TRAJECTORY_THICKNESS = 3
        self.BORDER_WIDTH = 80

        self.AGENT_CIRCLE_RADIUS = 25
        self.PREDATOR_COLOR = pygame.Color("red")
        self.VICTIM_COLOR = pygame.Color("green")

        self.screen = pygame.display.set_mode((self.FRAME_WIDTH, self.FRAME_HEIGHT))
        self.screen.fill(self.BACKGROUND_COLOR)

        pygame.display.init()
        pygame.display.update()
        pygame.mixer.init()

        self.draw_direction = True
        self.center_point = [0, 0]
        self.scale_koeff = 1

        self.trajectory_frequency = 1
        self.trajectory_frequency_counter = 0
        self.trajectory_predator_memory = deque(maxlen=15)
        self.trajectory_victim_memory = deque(maxlen=15)

    def reset(self):
        self.trajectory_predator_memory.clear()
        self.trajectory_victim_memory.clear()

    def draw(self, predator_pos, victim_pos, predator_acc, victim_acc, max_predator_acc, max_victim_acc):
        self.center_point = self.__get_center_point(predator_pos, victim_pos)
        relative_predator_pos = [i - j for i, j in zip(predator_pos, self.center_point)]
        relative_victim_pos = [i - j for i, j in zip(victim_pos, self.center_point)]
        self.scale_koeff = self.__get_scale_koeff(relative_predator_pos, relative_victim_pos)
        scaled_relative_predator_pos = [int(i * self.scale_koeff) for i in relative_predator_pos]
        scaled_relative_victim_pos = [int(i * self.scale_koeff) for i in relative_victim_pos]

        self.screen.fill(self.BACKGROUND_COLOR)
        if self.draw_direction:
            self.__draw_direction(scaled_relative_predator_pos, scaled_relative_victim_pos, predator_acc, victim_acc,
                                  max_predator_acc, max_victim_acc)

        self.trajectory_frequency_counter += 1
        if self.trajectory_frequency_counter >= self.trajectory_frequency:
            self.trajectory_frequency_counter = 0
            self.trajectory_predator_memory.append(predator_pos)
            self.trajectory_victim_memory.append(victim_pos)

        if self.draw_trajectory:
            for point in self.trajectory_victim_memory:
                relative_point = [i - j for i, j in zip(point, self.center_point)]
                scaled_relative_point = [int(i * self.scale_koeff) for i in relative_point]
                pygame.draw.circle(self.screen, self.VICTIM_COLOR,
                                   self.__map_coordinates(scaled_relative_point),
                                   int(self.TRAJECTORY_THICKNESS * self.scale_koeff))
            for point in self.trajectory_predator_memory:
                relative_point = [i - j for i, j in zip(point, self.center_point)]
                scaled_relative_point = [int(i * self.scale_koeff) for i in relative_point]
                pygame.draw.circle(self.screen, self.PREDATOR_COLOR,
                                   self.__map_coordinates(scaled_relative_point),
                                   int(self.TRAJECTORY_THICKNESS * self.scale_koeff))

        pygame.draw.circle(self.screen, self.VICTIM_COLOR,
                           self.__map_coordinates(scaled_relative_victim_pos),
                           int(self.AGENT_CIRCLE_RADIUS * self.scale_koeff))
        pygame.draw.circle(self.screen, self.PREDATOR_COLOR,
                           self.__map_coordinates(scaled_relative_predator_pos),
                           int(self.AGENT_CIRCLE_RADIUS * self.scale_koeff))

        pygame.display.update()

    def clear_screen(self):
        self.screen.fill(self.BACKGROUND_COLOR)
        pygame.display.update()

    def play_sound(self, path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

    def __get_center_point(self, predator_pos, victim_pos):
        average_point = [(i + j) // 2 for i, j in zip(predator_pos, victim_pos)]
        difference = np.array(average_point) - np.array(self.center_point)
        if np.linalg.norm(difference) >= min(self.FRAME_WIDTH // 2, self.FRAME_HEIGHT // 2):
            return average_point
        return self.center_point

    def __get_scale_koeff(self, predator_pos, victim_pos):
        max_val = abs(max(predator_pos + victim_pos, key=abs))
        if max_val == 0:
            return 1
        koeff = min(self.FRAME_HEIGHT // 2 - self.BORDER_WIDTH, self.FRAME_WIDTH // 2 - self.BORDER_WIDTH) / max_val
        return 1 if koeff >= 1 else koeff

    def __map_coordinates(self, coordinates):
        return [coordinates[0] + self.FRAME_WIDTH // 2, -coordinates[1] + self.FRAME_HEIGHT // 2]

    def __draw_direction(self, predator_pos, victim_pos, predator_acc, victim_acc, max_predator_acc, max_victim_acc):
        victim_vec = [
            0 if max_victim_acc == 0 else i * self.DIRECTION_VECTOR_MAX_LENGTH * self.scale_koeff / max_victim_acc for i
            in victim_acc]
        predator_vec = [
            0 if max_predator_acc == 0 else i * self.DIRECTION_VECTOR_MAX_LENGTH * self.scale_koeff / max_predator_acc
            for i in predator_acc]
        scaled_thickness = int(self.DIRECTION_VECTOR_THICKNESS * self.scale_koeff)
        pygame.draw.line(self.screen, self.VICTIM_COLOR, self.__map_coordinates(victim_pos),
                         self.__map_coordinates([sum(x) for x in zip(victim_pos, victim_vec)]),
                         scaled_thickness if scaled_thickness > 0 else 1)
        pygame.draw.line(self.screen, self.PREDATOR_COLOR, self.__map_coordinates(predator_pos),
                         self.__map_coordinates([sum(x) for x in zip(predator_pos, predator_vec)]),
                         scaled_thickness if scaled_thickness > 0 else 1)

    def log(self, text):
        if self.is_logging:
            self.log_file.write(text)
