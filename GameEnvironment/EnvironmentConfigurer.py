import os
import platform
import tkinter
import tkinter.filedialog
import threading
import pygame
import time
import numpy as np


WINDOW_TITLE = "* RL Predator-Victim experiment *"
WINDOW_WIDTH = 1250
WINDOW_HEIGHT = 700
FRAME_WIDTH = 700
FRAME_HEIGHT = 700


class EnvironmentConfigurer:
    def __init__(self, env_core, env_renderer, log_file):
        self.env_core = env_core
        self.env_renderer = env_renderer
        self.log_file = log_file

        # GUI elements
        self.gui_root = None
        self.gui_callback_functions = None
        self.gui_renderer_frame = None
        self.gui_right_frame = None
        self.gui_main_scrollbar = None
        self.gui_main_scrollbar_canvas = None
        self.gui_prepared_frame = None

        self.gui_render_options_frame = None
        self.gui_is_render_checkbox = None
        self.gui_is_render_booleanVar = None
        self.gui_delay_frame = None
        self.gui_is_delay_checkbox = None
        self.gui_is_delay_booleanVar = None
        self.gui_delay_scale = None
        self.gui_draw_direction_booleanVar = None
        self.gui_draw_direction_checkbox = None
        self.gui_draw_trajectory_booleanVar = None
        self.gui_draw_trajectory_checkbox = None

        self.gui_agent_start_position_frame = None
        self.gui_random_start_position_booleanVar = None
        self.gui_random_start_position_radiobutton = None
        self.gui_determined_start_position_radiobutton = None
        self.gui_determined_start_position_frame = None
        self.gui_determined_predator_start_position_frame = None
        self.gui_determined_victim_start_position_frame = None
        self.gui_determined_predator_start_position_x_frame = None
        self.gui_determined_predator_start_position_x_label = None
        self.gui_determined_predator_start_position_x_entry = None
        self.gui_determined_predator_start_position_y_frame = None
        self.gui_determined_predator_start_position_y_label = None
        self.gui_determined_predator_start_position_y_entry = None
        self.gui_determined_victim_start_position_x_frame = None
        self.gui_determined_victim_start_position_x_label = None
        self.gui_determined_victim_start_position_x_entry = None
        self.gui_determined_victim_start_position_y_frame = None
        self.gui_determined_victim_start_position_y_label = None
        self.gui_determined_victim_start_position_y_entry = None
        self.gui_set_start_position_button = None

        self.gui_observation_info_frame = None
        self.gui_show_observation_info_booleanVar = None
        self.gui_show_observation_info_checkbox = None
        self.gui_show_observation_info_internal_frame = None
        self.gui_observation_info_predator_frame = None
        self.gui_observation_info_predator_position_frame = None
        self.gui_observation_info_predator_position_meta_label = None
        self.gui_observation_info_predator_position_value_label = None
        self.gui_observation_info_predator_velocity_frame = None
        self.gui_observation_info_predator_velocity_meta_label = None
        self.gui_observation_info_predator_velocity_value_label = None
        self.gui_observation_info_victim_frame = None
        self.gui_observation_info_victim_position_frame = None
        self.gui_observation_info_victim_position_meta_label = None
        self.gui_observation_info_victim_position_value_label = None
        self.gui_observation_info_victim_velocity_frame = None
        self.gui_observation_info_victim_velocity_meta_label = None
        self.gui_observation_info_victim_velocity_value_label = None
        self.gui_observation_info_reward_frame = None
        self.gui_observation_info_predator_reward_frame = None
        self.gui_observation_info_predator_reward_meta_label = None
        self.gui_observation_info_predator_reward_value_label = None
        self.gui_observation_info_victim_reward_frame = None
        self.gui_observation_info_victim_reward_meta_label = None
        self.gui_observation_info_victim_reward_value_label = None

        self.gui_timer_frame = None
        self.gui_show_timer_booleanVar = None
        self.gui_show_timer_checkbox = None
        self.gui_timer_count_frame = None
        self.gui_timer_count_meta_label = None
        self.gui_timer_count_value_label = None
        self.gui_timer_scale = None
        self.gui_timer_buttons_frame = None
        self.gui_timer_play_pause_button = None
        self.gui_timer_stop_button = None

        self.gui_ai_control_frame = None
        self.gui_ai_control_manual_control_booleanVar = None
        self.gui_ai_control_turn_manual_control_checkbox = None
        self.gui_ai_control_make_collision_sound_booleanVar = None
        self.gui_ai_control_play_collision_sound_checkbox = None
        self.gui_ai_control_turn_ai_control_booleanVar = None
        self.gui_ai_control_turn_ai_control_checkbox = None
        self.gui_ai_control_turn_ai_learning_booleanVar = None
        self.gui_ai_control_turn_ai_learning_checkbox = None
        self.gui_ai_control_is_logging_booleanVar = None
        self.gui_ai_control_logging_checkbox = None
        self.gui_remember_frequency_frame = None
        self.gui_remember_frequency_scale = None
        self.gui_agents_batch_size_frame = None
        self.gui_set_batch_size_label = None
        self.gui_batch_size_entry = None
        self.gui_set_batch_size_button = None

        self.gui_ai_control_engines_frame = None
        self.gui_predator_engine_frame = None
        self.gui_predator_engine_listbox = None
        self.gui_victim_engine_frame = None
        self.gui_victim_engine_listbox = None
        self.gui_save_weights_frequency_frame = None
        self.gui_save_weights_label = None
        self.gui_save_weights_count_entry = None
        self.gui_save_weights_set_frequency_button = None
        self.gui_predator_engine_save_weights_frame = None
        self.gui_predator_engine_save_weights_booleanVar = None
        self.gui_predator_engine_save_weights_checkbox = None
        self.gui_predator_engine_save_weights_folder_entry = None
        self.gui_predator_engine_save_weights_set_button = None
        self.gui_victim_engine_save_weights_frame = None
        self.gui_victim_engine_save_weights_booleanVar = None
        self.gui_victim_engine_save_weights_checkbox = None
        self.gui_victim_engine_save_weights_folder_entry = None
        self.gui_victim_engine_save_weights_set_button = None
        self.gui_load_weights_frame = None
        self.gui_load_weights_label = None
        self.gui_load_weights_predator_button = None
        self.gui_load_weights_victim_button = None
        self.gui_predator_engine_save_weights_directory_name = None
        self.gui_victim_engine_save_weights_directory_name = None
        self.gui_load_weights_directory_name_predator_filedialog = None
        self.gui_load_weights_directory_name_victim_filedialog = None
        self.gui_game_summary_frame = None
        self.gui_current_episode_frame = None
        self.gui_current_episode_meta_label = None
        self.gui_current_episode_value_label = None
        self.gui_success_episodes_count_frame = None
        self.gui_success_episodes_count_meta_label = None
        self.gui_success_episodes_count_value_label = None
        #

        # GUI managed parameters (initial values can be set here, handling is automated)
        self.parameter_is_render = True
        self.parameter_is_delay = True
        self.parameter_render_delay = 0.01
        self.parameter_draw_direction = True
        self.parameter_draw_trajectory = True

        self.parameter_start_position_random = True
        self.parameter_predator_start_position = [40.0, 0.0]
        self.parameter_victim_start_position   = [-40.0, 0.0]

        self.parameter_show_observation_info = False

        self.parameter_show_timer = False

        self.parameter_turn_manual_control = False
        self.parameter_play_collision_sound = False
        self.parameter_turn_ai_control = False
        self.parameter_turn_ai_learning = False
        self.parameter_is_logging = False
        self.parameter_remember_frequency = 3
        self.parameter_batch_size = 256

        self.parameter_predator_engine_save_weights = True
        self.parameter_victim_engine_save_weights = True
        self.parameter_save_weights_episode_count = 1000
        #

        # Internal variables (don't change values!)
        self.predator_ai_engines_list = []
        self.selected_predator_engine_index = 0
        self.selected_predator_ai_engine_changed = False

        self.victim_ai_engines_list = []
        self.selected_victim_engine_index = 0
        self.selected_victim_ai_engine_changed = False

        self.ready_to_load_weights_predator = False
        self.ready_to_load_weights_victim = False
        self.load_predator_engine_weights_filepath = ""
        self.load_victim_engine_weights_filepath = ""

        self.game_ai_thread = None
        self.started = False
        self.current_episode = 1
        self.success_episodes_count = 0

        self.current_ai_agent_predator = None
        self.current_ai_agent_victim = None

        self.current_observation = {
            "agent_0": np.zeros(len(self.env_core.observation_space.high)),
            "agent_1": np.zeros(len(self.env_core.observation_space.high)),
        }
        self.current_predator_reward = 0
        self.current_victim_reward = 0
        #

        self.episode_n_gametime = []
        self.episode_n_catch_time = []
        self.episode_n_reward_predator = []
        self.episode_n_reward_victim = []
        self.episode_n_catchcount = []

    def init(self):
        self.__init_agent_ai_list()
        self._init_gui()

        self.started = False
        self.game_ai_thread = threading.Thread(target=self.__game_ai_thread, args=())
        self.game_ai_thread.setDaemon(True)

    def __init_agent_ai_list(self):
        from GameEnvironment.AgentAI.EmptyAgent import EmptyAgent
        from GameEnvironment.AgentAI.AgentDQN import DQNAgent
        from GameEnvironment.AgentAI.A2CAgent import A2CAgent

        self.predator_ai_engines_list.append(("Empty", EmptyAgent(self.env_core, self.log_file)))
        self.predator_ai_engines_list.append(("DQN", DQNAgent(self.env_core, self.log_file)))
        self.predator_ai_engines_list.append(("A2C", A2CAgent(self.env_core, self.log_file)))

        self.victim_ai_engines_list.append(("Empty", EmptyAgent(self.env_core, self.log_file)))
        self.victim_ai_engines_list.append(("DQN", DQNAgent(self.env_core, self.log_file)))
        self.victim_ai_engines_list.append(("A2C", A2CAgent(self.env_core, self.log_file)))

    def _init_gui(self):
        self.gui_callback_functions = []

        def __make_root_frame():
            self.gui_root = tkinter.Tk(className=WINDOW_TITLE)
            self.gui_root.geometry(str(WINDOW_WIDTH) + "x" + str(WINDOW_HEIGHT))
            self.gui_root.resizable(0, 0)
            self.gui_root.protocol("WM_DELETE_WINDOW", self.__event_on_closing)
        __make_root_frame()

        def __make_renderer_frame():
            self.gui_renderer_frame = tkinter.Frame(self.gui_root, width=FRAME_WIDTH, height=FRAME_HEIGHT)
            self.gui_renderer_frame.pack(side=tkinter.LEFT)
            os.environ['SDL_WINDOWID'] = str(self.gui_renderer_frame.winfo_id())
            if platform.system == "Windows":
                os.environ['SDL_VIDEODRIVER'] = 'windib'
        __make_renderer_frame()

        def __make_gui_frame():
            self.gui_right_frame = tkinter.LabelFrame(self.gui_root, width=WINDOW_WIDTH - FRAME_WIDTH, height=WINDOW_HEIGHT,
                                           text="Options")
            self.gui_right_frame.pack_propagate(0)
            self.gui_right_frame.pack(side=tkinter.RIGHT)

            self.gui_main_scrollbar_canvas = tkinter.Canvas(self.gui_right_frame)
            self.gui_main_scrollbar = tkinter.Scrollbar(self.gui_right_frame, orient="vertical", command=self.gui_main_scrollbar_canvas.yview)

            self.gui_prepared_frame = tkinter.Frame(self.gui_main_scrollbar_canvas)
            self.gui_prepared_frame.bind(
                "<Configure>",
                lambda e: self.gui_main_scrollbar_canvas.configure(
                    scrollregion=self.gui_main_scrollbar_canvas.bbox("all")
                )
            )
            self.gui_main_scrollbar_canvas.create_window((0, 0), window=self.gui_prepared_frame, anchor="nw")
            self.gui_main_scrollbar_canvas.configure(yscrollcommand=self.gui_main_scrollbar.set)

            self.gui_right_frame.pack(side=tkinter.RIGHT)
            self.gui_main_scrollbar_canvas.pack(side="left", fill="both", expand=True)
            self.gui_main_scrollbar.pack(side="right", fill="y")
        __make_gui_frame()

        def __make_render_options_frame():
            def __make_option_is_render():
                self.gui_is_render_booleanVar = tkinter.BooleanVar()
                self.gui_is_render_booleanVar.set(self.parameter_is_render)

                def _callback_is_render():
                    self.parameter_is_render = self.gui_is_render_booleanVar.get()
                    if self.parameter_is_render:
                        self.gui_is_render_checkbox.pack_forget()
                        self.gui_delay_frame.pack_forget()
                        self.gui_draw_direction_checkbox.pack_forget()
                        self.gui_draw_trajectory_checkbox.forget()
                        self.gui_is_render_checkbox.pack(anchor=tkinter.NW)
                        self.gui_delay_frame.pack(anchor=tkinter.NW)
                        self.gui_draw_direction_checkbox.pack(anchor=tkinter.NW)
                        self.gui_draw_trajectory_checkbox.pack(anchor=tkinter.NW)
                    else:
                        self.gui_is_render_checkbox.pack_forget()
                        self.gui_delay_frame.pack_forget()
                        self.gui_draw_direction_checkbox.pack_forget()
                        self.gui_draw_trajectory_checkbox.forget()
                        self.gui_is_render_checkbox.pack(anchor=tkinter.NW)
                    print("Is render = ", self.parameter_is_render)
                self.gui_callback_functions.append(_callback_is_render)

                self.gui_is_render_checkbox = tkinter.Checkbutton(self.gui_render_options_frame,
                                                                  text="Render image (should be turned off for longtime training)",
                                                                  variable=self.gui_is_render_booleanVar,
                                                                  onvalue=True,
                                                                  offvalue=False,
                                                                  command=_callback_is_render)
                self.gui_is_render_checkbox.pack(anchor=tkinter.NW)

            def __make_option_delay():
                self.gui_delay_frame = tkinter.Frame(self.gui_render_options_frame)

                self.gui_is_delay_booleanVar = tkinter.BooleanVar()
                self.gui_is_delay_booleanVar.set(self.parameter_is_delay)

                def _callback_is_delay():
                    self.parameter_is_delay = self.gui_is_delay_booleanVar.get()
                    print("Is delay = ", self.parameter_is_delay)

                self.gui_is_delay_checkbox = tkinter.Checkbutton(self.gui_delay_frame,
                                                                 text="Is delay: ",
                                                                 variable=self.gui_is_delay_booleanVar,
                                                                 onvalue=True,
                                                                 offvalue=False,
                                                                 command=_callback_is_delay)
                self.gui_delay_scale = tkinter.Scale(self.gui_delay_frame,
                                                     from_=0.001,
                                                     to=0.2,
                                                     resolution=0.001,
                                                     orient=tkinter.HORIZONTAL,
                                                     sliderlength=20,
                                                     length=350)
                self.gui_delay_scale.set(self.parameter_render_delay)
                self.gui_delay_frame.pack(anchor=tkinter.NW)
                self.gui_is_delay_checkbox.pack(side=tkinter.LEFT)
                self.gui_delay_scale.pack(side=tkinter.LEFT)

            def __make_option_draw_acceleration_vectors():
                self.gui_draw_direction_booleanVar = tkinter.BooleanVar()
                self.gui_draw_direction_booleanVar.set(self.parameter_draw_direction)

                def _callback_draw_direction():
                    self.parameter_draw_direction = self.gui_draw_direction_booleanVar.get()
                    print("Draw acceleration direction = ", self.parameter_draw_direction)

                self.gui_draw_direction_checkbox = tkinter.Checkbutton(self.gui_render_options_frame,
                                                                       text="Draw acceleration vectors",
                                                                       variable=self.gui_draw_direction_booleanVar,
                                                                       onvalue=True,
                                                                       offvalue=False,
                                                                       command=_callback_draw_direction)
                self.gui_draw_direction_checkbox.pack(anchor=tkinter.NW)

            def __make_option_draw_trajectory():
                self.gui_draw_trajectory_booleanVar = tkinter.BooleanVar()
                self.gui_draw_trajectory_booleanVar.set(self.parameter_draw_trajectory)

                def _callback_draw_trajectory():
                    self.parameter_draw_trajectory = self.gui_draw_trajectory_booleanVar.get()
                    print("Draw trajectory = ", self.parameter_draw_trajectory)

                self.gui_draw_trajectory_checkbox = tkinter.Checkbutton(self.gui_render_options_frame,
                                                                        text="Draw agent's trajectory",
                                                                        variable=self.gui_draw_trajectory_booleanVar,
                                                                        onvalue=True,
                                                                        offvalue=False,
                                                                        command=_callback_draw_trajectory)
                self.gui_draw_trajectory_checkbox.pack(anchor=tkinter.NW)

            self.gui_render_options_frame = tkinter.LabelFrame(self.gui_prepared_frame, text="Render options")
            self.gui_render_options_frame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)
            __make_option_is_render()
            __make_option_delay()
            __make_option_draw_acceleration_vectors()
            __make_option_draw_trajectory()
        __make_render_options_frame()

        def __make_start_position_frame():
            def __make_start_position_radiobuttons():
                self.gui_random_start_position_booleanVar = tkinter.BooleanVar()
                self.gui_random_start_position_booleanVar.set(self.parameter_start_position_random)

                def _callback_start_position_radiobuttons():
                    if self.gui_random_start_position_booleanVar.get():
                        self.gui_random_start_position_radiobutton.pack_forget()
                        self.gui_determined_start_position_radiobutton.pack_forget()
                        self.gui_determined_start_position_frame.pack_forget()
                        self.gui_random_start_position_radiobutton.pack(anchor=tkinter.NW)
                        self.gui_determined_start_position_radiobutton.pack(anchor=tkinter.NW)
                        self.parameter_start_position_random = True
                    else:
                        self.gui_random_start_position_radiobutton.pack_forget()
                        self.gui_determined_start_position_radiobutton.pack_forget()
                        self.gui_determined_start_position_frame.pack_forget()
                        self.gui_random_start_position_radiobutton.pack(anchor=tkinter.NW)
                        self.gui_determined_start_position_radiobutton.pack(anchor=tkinter.NW)
                        self.gui_determined_start_position_frame.pack(anchor=tkinter.NW)
                        self.parameter_start_position_random = False
                    print("Start position random = ", self.parameter_start_position_random)
                self.gui_callback_functions.append(_callback_start_position_radiobuttons)

                self.gui_random_start_position_radiobutton = tkinter.Radiobutton(
                                                                self.gui_agent_start_position_frame,
                                                                text='Random position',
                                                                variable=self.gui_random_start_position_booleanVar,
                                                                value=True,
                                                                command=_callback_start_position_radiobuttons)
                self.gui_determined_start_position_radiobutton = tkinter.Radiobutton(
                                                                self.gui_agent_start_position_frame,
                                                                text='Determined position',
                                                                variable=self.gui_random_start_position_booleanVar,
                                                                value=False,
                                                                command=_callback_start_position_radiobuttons)

                self.gui_random_start_position_radiobutton.pack(anchor=tkinter.NW)
                self.gui_determined_start_position_radiobutton.pack(anchor=tkinter.NW)

            def __make_determined_start_position_frame():
                self.gui_determined_start_position_frame = tkinter.Frame(self.gui_agent_start_position_frame)

            def __make_determined_predator_start_position_frame():
                self.gui_determined_predator_start_position_frame = tkinter.LabelFrame(
                                                                            self.gui_determined_start_position_frame,
                                                                            text="Predator start position")
                self.gui_determined_predator_start_position_frame.pack(side=tkinter.LEFT)

                self.gui_determined_predator_start_position_x_frame = tkinter.Frame(self.gui_determined_predator_start_position_frame)
                self.gui_determined_predator_start_position_x_frame.pack(anchor=tkinter.NW)
                self.gui_determined_predator_start_position_x_label = tkinter.Label(self.gui_determined_predator_start_position_x_frame,
                                                                                    text='X = ')
                self.gui_determined_predator_start_position_x_entry = tkinter.Entry(self.gui_determined_predator_start_position_x_frame,
                                                                                    width=15)
                self.gui_determined_predator_start_position_x_entry.insert(0, str(self.parameter_predator_start_position[0]))
                self.gui_determined_predator_start_position_x_label.pack(side=tkinter.LEFT)
                self.gui_determined_predator_start_position_x_entry.pack(side=tkinter.LEFT)

                self.gui_determined_predator_start_position_y_frame = tkinter.Frame(self.gui_determined_predator_start_position_frame)
                self.gui_determined_predator_start_position_y_frame.pack(anchor=tkinter.NW)
                self.gui_determined_predator_start_position_y_label = tkinter.Label(self.gui_determined_predator_start_position_y_frame,
                                                                                    text='Y = ')
                self.gui_determined_predator_start_position_y_entry = tkinter.Entry(self.gui_determined_predator_start_position_y_frame,
                                                                                    width=15)
                self.gui_determined_predator_start_position_y_entry.insert(0, str(self.parameter_predator_start_position[1]))
                self.gui_determined_predator_start_position_y_label.pack(side=tkinter.LEFT)
                self.gui_determined_predator_start_position_y_entry.pack(side=tkinter.LEFT)

            def __make_determined_victim_start_position_frame():
                self.gui_determined_victim_start_position_frame = tkinter.LabelFrame(
                    self.gui_determined_start_position_frame,
                    text="Victim start position")
                self.gui_determined_victim_start_position_frame.pack(side=tkinter.LEFT)

                self.gui_determined_victim_start_position_x_frame = tkinter.Frame(self.gui_determined_victim_start_position_frame)
                self.gui_determined_victim_start_position_x_frame.pack(anchor=tkinter.NW)
                self.gui_determined_victim_start_position_x_label = tkinter.Label(self.gui_determined_victim_start_position_x_frame,
                                                                                  text='X = ')
                self.gui_determined_victim_start_position_x_entry = tkinter.Entry(self.gui_determined_victim_start_position_x_frame,
                                                                                  width=15)
                self.gui_determined_victim_start_position_x_entry.insert(0, str(self.parameter_victim_start_position[0]))
                self.gui_determined_victim_start_position_x_label.pack(side=tkinter.LEFT)
                self.gui_determined_victim_start_position_x_entry.pack(side=tkinter.LEFT)

                self.gui_determined_victim_start_position_y_frame = tkinter.Frame(self.gui_determined_victim_start_position_frame)
                self.gui_determined_victim_start_position_y_frame.pack(anchor=tkinter.NW)
                self.gui_determined_victim_start_position_y_label = tkinter.Label(self.gui_determined_victim_start_position_y_frame,
                                                                                  text='Y = ')
                self.gui_determined_victim_start_position_y_entry = tkinter.Entry(self.gui_determined_victim_start_position_y_frame,
                                                                                  width=15)
                self.gui_determined_victim_start_position_y_entry.insert(0, str(self.parameter_victim_start_position[1]))
                self.gui_determined_victim_start_position_y_label.pack(side=tkinter.LEFT)
                self.gui_determined_victim_start_position_y_entry.pack(side=tkinter.LEFT)

            def __make_set_determined_start_position_button():
                def _callback_set_start_position_button():
                    self.parameter_predator_start_position = [
                        float(self.gui_determined_predator_start_position_x_entry.get()),
                        float(self.gui_determined_predator_start_position_y_entry.get())
                    ]
                    self.parameter_victim_start_position = [
                        float(self.gui_determined_victim_start_position_x_entry.get()),
                        float(self.gui_determined_victim_start_position_y_entry.get())
                    ]
                    print("Start position is set = ", self.parameter_predator_start_position, " and ",
                          self.parameter_victim_start_position)

                self.gui_set_start_position_button = tkinter.Button(self.gui_determined_start_position_frame,
                                                                    text="Set positions",
                                                                    command=_callback_set_start_position_button)
                self.gui_set_start_position_button.pack(anchor=tkinter.NW, padx=10, pady=18)

            self.gui_agent_start_position_frame = tkinter.LabelFrame(self.gui_prepared_frame,
                                                                     text="Agent start positions")
            self.gui_agent_start_position_frame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

            __make_start_position_radiobuttons()
            __make_determined_start_position_frame()
            __make_determined_predator_start_position_frame()
            __make_determined_victim_start_position_frame()
            __make_set_determined_start_position_button()
        __make_start_position_frame()

        def __make_observation_frame():
            def __make_observation_info_checkbox_and_internal_frame():
                self.gui_observation_info_frame = tkinter.LabelFrame(self.gui_prepared_frame, text="Observation info")
                self.gui_observation_info_frame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

                self.gui_show_observation_info_booleanVar = tkinter.BooleanVar()
                self.gui_show_observation_info_booleanVar.set(self.parameter_show_observation_info)

                def _callback_show_observation_info():
                    self.parameter_show_observation_info = self.gui_show_observation_info_booleanVar.get()
                    if self.parameter_show_observation_info:
                        self.gui_show_observation_info_internal_frame.pack_forget()
                        self.gui_show_observation_info_internal_frame.pack(side=tkinter.LEFT)
                    else:
                        self.gui_show_observation_info_internal_frame.pack_forget()
                    print("Show observation info = ", self.parameter_show_observation_info)
                self.gui_callback_functions.append(_callback_show_observation_info)

                self.gui_show_observation_info_checkbox = tkinter.Checkbutton(self.gui_observation_info_frame,
                                                                              text="Show observation info",
                                                                              variable=self.gui_show_observation_info_booleanVar,
                                                                              onvalue=True,
                                                                              offvalue=False,
                                                                              command=_callback_show_observation_info)
                self.gui_show_observation_info_checkbox.pack(anchor=tkinter.NW)

                self.gui_show_observation_info_internal_frame = tkinter.Frame(self.gui_observation_info_frame)
                self.gui_show_observation_info_internal_frame.pack(side=tkinter.LEFT)

            def __make_observation_info_predator_frame():
                self.gui_observation_info_predator_frame = tkinter.LabelFrame(self.gui_show_observation_info_internal_frame,
                                                                              text="Predator",
                                                                              width=50)
                self.gui_observation_info_predator_frame.pack(side=tkinter.LEFT, fill=tkinter.Y, padx=5, pady=5)

                self.gui_observation_info_predator_position_frame = tkinter.Frame(self.gui_observation_info_predator_frame)
                self.gui_observation_info_predator_position_meta_label = tkinter.Label(self.gui_observation_info_predator_position_frame,
                                                                                       text="Position =")
                self.gui_observation_info_predator_position_value_label = tkinter.Label(self.gui_observation_info_predator_position_frame,
                                                                                        text=str(self.env_core.predator_pos))
                self.gui_observation_info_predator_position_frame.pack(anchor=tkinter.NW, padx=3)
                self.gui_observation_info_predator_position_meta_label.pack(side=tkinter.LEFT)
                self.gui_observation_info_predator_position_value_label.pack(side=tkinter.LEFT)

                self.gui_observation_info_predator_velocity_frame = tkinter.Frame(self.gui_observation_info_predator_frame)
                self.gui_observation_info_predator_velocity_meta_label = tkinter.Label(self.gui_observation_info_predator_velocity_frame,
                                                                                       text="Velocity =")
                self.gui_observation_info_predator_velocity_value_label = tkinter.Label(self.gui_observation_info_predator_velocity_frame,
                                                                                        text=str(list(self.env_core.predator_velocity_vector)))
                self.gui_observation_info_predator_velocity_frame.pack(anchor=tkinter.NW, padx=3)
                self.gui_observation_info_predator_velocity_meta_label.pack(side=tkinter.LEFT)
                self.gui_observation_info_predator_velocity_value_label.pack(side=tkinter.LEFT)

            def __make_observation_info_victim_frame():
                self.gui_observation_info_victim_frame = tkinter.LabelFrame(self.gui_show_observation_info_internal_frame,
                                                                            text="Victim",
                                                                            width=50)
                self.gui_observation_info_victim_frame.pack(side=tkinter.LEFT, fill=tkinter.Y, padx=5, pady=5)

                self.gui_observation_info_victim_position_frame = tkinter.Frame(self.gui_observation_info_victim_frame)
                self.gui_observation_info_victim_position_meta_label = tkinter.Label(self.gui_observation_info_victim_position_frame,
                                                                                     text="Position =")
                self.gui_observation_info_victim_position_value_label = tkinter.Label(self.gui_observation_info_victim_position_frame,
                                                                                      text=str(self.env_core.victim_pos))
                self.gui_observation_info_victim_position_frame.pack(anchor=tkinter.NW, padx=3)
                self.gui_observation_info_victim_position_meta_label.pack(side=tkinter.LEFT)
                self.gui_observation_info_victim_position_value_label.pack(side=tkinter.LEFT)

                self.gui_observation_info_victim_velocity_frame = tkinter.Frame(self.gui_observation_info_victim_frame)
                self.gui_observation_info_victim_velocity_meta_label = tkinter.Label(self.gui_observation_info_victim_velocity_frame,
                                                                                     text="Velocity =")
                self.gui_observation_info_victim_velocity_value_label = tkinter.Label(self.gui_observation_info_victim_velocity_frame,
                                                                                      text=str(list(self.env_core.victim_velocity_vector)))
                self.gui_observation_info_victim_velocity_frame.pack(anchor=tkinter.NW, padx=3)
                self.gui_observation_info_victim_velocity_meta_label.pack(side=tkinter.LEFT)
                self.gui_observation_info_victim_velocity_value_label.pack(side=tkinter.LEFT)

            def __make_observation_info_reward_frame():
                self.gui_observation_info_reward_frame = tkinter.LabelFrame(self.gui_show_observation_info_internal_frame,
                                                                            text="Agent rewards",
                                                                            width=50)
                self.gui_observation_info_reward_frame.pack(side=tkinter.LEFT, fill=tkinter.Y, padx=5, pady=5)

                self.gui_observation_info_predator_reward_frame = tkinter.Frame(self.gui_observation_info_reward_frame)
                self.gui_observation_info_predator_reward_meta_label = tkinter.Label(self.gui_observation_info_predator_reward_frame,
                                                                                     text="Predator =")
                self.gui_observation_info_predator_reward_value_label = tkinter.Label(self.gui_observation_info_predator_reward_frame,
                                                                                      text=str(0))
                self.gui_observation_info_predator_reward_frame.pack(anchor=tkinter.NW, padx=3)
                self.gui_observation_info_predator_reward_meta_label.pack(side=tkinter.LEFT)
                self.gui_observation_info_predator_reward_value_label.pack(side=tkinter.LEFT)

                self.gui_observation_info_victim_reward_frame = tkinter.Frame(self.gui_observation_info_reward_frame)
                self.gui_observation_info_victim_reward_meta_label = tkinter.Label(self.gui_observation_info_victim_reward_frame,
                                                                                   text="Victim =")
                self.gui_observation_info_victim_reward_value_label = tkinter.Label(self.gui_observation_info_victim_reward_frame,
                                                                                    text=str(0))
                self.gui_observation_info_victim_reward_frame.pack(anchor=tkinter.NW, padx=3)
                self.gui_observation_info_victim_reward_meta_label.pack(side=tkinter.LEFT)
                self.gui_observation_info_victim_reward_value_label.pack(side=tkinter.LEFT)

            __make_observation_info_checkbox_and_internal_frame()
            __make_observation_info_predator_frame()
            __make_observation_info_victim_frame()
            __make_observation_info_reward_frame()
        __make_observation_frame()

        def __make_timer_frame():
            self.gui_timer_frame = tkinter.LabelFrame(self.gui_prepared_frame, text="Timer control")
            self.gui_timer_frame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

            def __make_show_timer_checkbox():
                self.gui_show_timer_booleanVar = tkinter.BooleanVar()
                self.gui_show_timer_booleanVar.set(self.parameter_show_timer)

                def _callback_show_timer():
                    self.parameter_show_timer = self.gui_show_timer_booleanVar.get()
                    if self.parameter_show_timer:
                        self.gui_timer_count_frame.pack_forget()
                        self.gui_timer_scale.pack_forget()
                        self.gui_timer_buttons_frame.pack_forget()
                        self.gui_timer_count_frame.pack(anchor=tkinter.NW)
                        self.gui_timer_scale.pack(anchor=tkinter.NW)
                        self.gui_timer_buttons_frame.pack(anchor=tkinter.NW)
                    else:
                        self.gui_timer_count_frame.pack_forget()
                        self.gui_timer_scale.pack_forget()
                        self.gui_timer_buttons_frame.pack_forget()
                        self.gui_timer_stop_button.invoke()
                    print("Show timer = ", self.parameter_show_timer)
                self.gui_callback_functions.append(_callback_show_timer)

                self.gui_show_timer_checkbox = tkinter.Checkbutton(self.gui_timer_frame,
                                                                   text="Turn timer",
                                                                   variable=self.gui_show_timer_booleanVar,
                                                                   onvalue=True,
                                                                   offvalue=False,
                                                                   command=_callback_show_timer)
                self.gui_show_timer_checkbox.pack(anchor=tkinter.NW)

            def __make_timer_count_frame():
                self.gui_timer_count_frame = tkinter.Frame(self.gui_timer_frame)
                self.gui_timer_count_meta_label = tkinter.Label(self.gui_timer_count_frame, text="Timer count: ")
                self.gui_timer_count_value_label = tkinter.Label(self.gui_timer_count_frame, text="0")
                self.gui_timer_count_meta_label.pack(side=tkinter.LEFT)
                self.gui_timer_count_value_label.pack(side=tkinter.LEFT)
                self.gui_timer_scale = tkinter.Scale(self.gui_timer_frame, from_=10, to=2000, resolution=10,
                                                     orient=tkinter.HORIZONTAL, sliderlength=20, length=350,
                                                     label="Timer duration")
                self.gui_timer_scale.set(self.env_core.GAME_STEPS)

            def __make_timer_buttons_frame():
                self.gui_timer_buttons_frame = tkinter.Frame(self.gui_timer_frame)

                def _callback_timer_play_pause():
                    self.env_core.turn_timer = not self.env_core.turn_timer
                    print("Pause/Play button")

                def _callback_timer_stop():
                    self.env_core.turn_timer = False
                    self.env_core.game_current_step = 0
                    print("Stop button")

                self.gui_timer_play_pause_button = tkinter.Button(self.gui_timer_buttons_frame,
                                                                  text="Play/Pause",
                                                                  command=_callback_timer_play_pause)
                self.gui_timer_stop_button = tkinter.Button(self.gui_timer_buttons_frame,
                                                            text="Stop",
                                                            command=_callback_timer_stop)
                self.gui_timer_play_pause_button.pack(side=tkinter.LEFT)
                self.gui_timer_stop_button.pack(side=tkinter.LEFT)

            __make_show_timer_checkbox()
            __make_timer_count_frame()
            __make_timer_buttons_frame()
        __make_timer_frame()

        def __make_ai_control_frame():
            def __make_manual_control_checkbox():
                self.gui_ai_control_manual_control_booleanVar = tkinter.BooleanVar()
                self.gui_ai_control_manual_control_booleanVar.set(self.parameter_turn_manual_control)

                def _check_turn_manual_control():
                    self.parameter_turn_manual_control = self.gui_ai_control_manual_control_booleanVar.get()
                    print("Turn manual control = ", self.parameter_turn_manual_control)
                self.gui_callback_functions.append(_check_turn_manual_control)

                self.gui_ai_control_turn_manual_control_checkbox = tkinter.Checkbutton(
                                                                self.gui_ai_control_frame,
                                                                text="Manual control - ZAQWEDCX & NumPad Numbers (Use delay!!!)",
                                                                variable=self.gui_ai_control_manual_control_booleanVar,
                                                                onvalue=True,
                                                                offvalue=False,
                                                                command=_check_turn_manual_control)
                self.gui_ai_control_turn_manual_control_checkbox.pack(anchor=tkinter.NW)

            def __make_play_collision_sound_checkbox():
                self.gui_ai_control_make_collision_sound_booleanVar = tkinter.BooleanVar()
                self.gui_ai_control_make_collision_sound_booleanVar.set(self.parameter_play_collision_sound)

                def _callback_play_collision_sound():
                    self.parameter_play_collision_sound = self.gui_ai_control_make_collision_sound_booleanVar.get()
                    print("Play collision sound = ", self.parameter_play_collision_sound)
                self.gui_callback_functions.append(_callback_play_collision_sound)

                self.gui_ai_control_play_collision_sound_checkbox = tkinter.Checkbutton(
                                                                    self.gui_ai_control_frame,
                                                                    text="Play collision sound",
                                                                    variable=self.gui_ai_control_make_collision_sound_booleanVar,
                                                                    onvalue=True,
                                                                    offvalue=False,
                                                                    command=_callback_play_collision_sound)
                self.gui_ai_control_play_collision_sound_checkbox.pack(anchor=tkinter.NW)

            def __make_turn_ai_control_checkbox():
                self.gui_ai_control_turn_ai_control_booleanVar = tkinter.BooleanVar()
                self.gui_ai_control_turn_ai_control_booleanVar.set(self.parameter_turn_ai_control)

                def _callback_turn_ai_control():
                    self.parameter_turn_ai_control = self.gui_ai_control_turn_ai_control_booleanVar.get()
                    if not self.parameter_turn_ai_control:
                        self.gui_ai_control_turn_ai_learning_checkbox.deselect()
                        self.parameter_turn_ai_learning = False
                    print("Turn AI control = ", self.parameter_turn_ai_control)
                self.gui_callback_functions.append(_callback_turn_ai_control)

                self.gui_ai_control_turn_ai_control_checkbox = tkinter.Checkbutton(
                                                            self.gui_ai_control_frame,
                                                            text="Turn AI control",
                                                            variable=self.gui_ai_control_turn_ai_control_booleanVar,
                                                            onvalue=True,
                                                            offvalue=False,
                                                            command=_callback_turn_ai_control)
                self.gui_ai_control_turn_ai_control_checkbox.pack(anchor=tkinter.NW)

            def __make_turn_ai_learning_checkbox():
                self.gui_ai_control_turn_ai_learning_booleanVar = tkinter.BooleanVar()
                self.gui_ai_control_turn_ai_learning_booleanVar.set(self.parameter_turn_ai_learning)

                def _callback_turn_ai_learning():
                    self.parameter_turn_ai_learning = self.gui_ai_control_turn_ai_learning_booleanVar.get()
                    print("Turn AI learning = ", self.parameter_turn_ai_learning)
                self.gui_callback_functions.append(_callback_turn_ai_learning)

                self.gui_ai_control_turn_ai_learning_checkbox = tkinter.Checkbutton(self.gui_ai_control_frame,
                                                                                    text="Turn AI learning",
                                                                                    variable=self.gui_ai_control_turn_ai_learning_booleanVar,
                                                                                    onvalue=True,
                                                                                    offvalue=False,
                                                                                    command=_callback_turn_ai_learning)
                self.gui_ai_control_turn_ai_learning_checkbox.pack(anchor=tkinter.NW)

            def __make_logging_checkbox():
                self.gui_ai_control_is_logging_booleanVar = tkinter.BooleanVar()
                self.gui_ai_control_is_logging_booleanVar.set(self.parameter_is_logging)

                def _callback_is_logging():
                    self.parameter_is_logging = self.gui_ai_control_is_logging_booleanVar.get()
                    self.env_core.is_logging = self.gui_ai_control_is_logging_booleanVar.get()
                    self.env_renderer.is_logging = self.gui_ai_control_is_logging_booleanVar.get()
                    if self.current_ai_agent_predator:
                        self.current_ai_agent_predator.is_logging = self.gui_ai_control_is_logging_booleanVar.get()
                    if self.current_ai_agent_victim:
                        self.current_ai_agent_victim.is_logging = self.gui_ai_control_is_logging_booleanVar.get()
                    print("Logging = ", self.parameter_is_logging)
                self.gui_callback_functions.append(_callback_is_logging)

                self.gui_ai_control_logging_checkbox = tkinter.Checkbutton(self.gui_ai_control_frame,
                                                                           text="Logging",
                                                                           variable=self.gui_ai_control_is_logging_booleanVar,
                                                                           onvalue=True,
                                                                           offvalue=False,
                                                                           command=_callback_is_logging)
                self.gui_ai_control_logging_checkbox.pack(anchor=tkinter.NW)

            def __make_remember_frequency_frame():
                self.gui_remember_frequency_frame = tkinter.LabelFrame(self.gui_ai_control_frame,
                                                                       text="Remember frequency (1 time per N steps)")
                self.gui_remember_frequency_frame.pack(anchor=tkinter.NW, padx=10)

                self.gui_remember_frequency_scale = tkinter.Scale(self.gui_remember_frequency_frame,
                                                                  from_=1,
                                                                  to=50,
                                                                  resolution=1,
                                                                  orient=tkinter.HORIZONTAL,
                                                                  sliderlength=30,
                                                                  length=350)
                self.gui_remember_frequency_scale.pack(side=tkinter.LEFT)
                self.gui_remember_frequency_scale.set(self.parameter_remember_frequency)

            def __make_batch_size_frame():
                self.gui_agents_batch_size_frame = tkinter.Frame(self.gui_ai_control_frame)
                self.gui_agents_batch_size_frame.pack(anchor=tkinter.NW, padx=10)

                def _callback_set_batch_size():
                    self.parameter_batch_size = int(self.gui_batch_size_entry.get())
                    print("Batch size = ", self.parameter_batch_size)
                self.gui_callback_functions.append(_callback_set_batch_size)

                self.gui_set_batch_size_label = tkinter.Label(self.gui_agents_batch_size_frame, text="Batch size =")
                self.gui_batch_size_entry = tkinter.Entry(self.gui_agents_batch_size_frame, text=str(self.parameter_batch_size))
                self.gui_batch_size_entry.insert(0, str(self.parameter_batch_size))
                self.gui_set_batch_size_button = tkinter.Button(self.gui_agents_batch_size_frame,
                                                                text="Apply",
                                                                command=_callback_set_batch_size)
                self.gui_set_batch_size_label.pack(side=tkinter.LEFT, padx=2)
                self.gui_batch_size_entry.pack(side=tkinter.LEFT, padx=2)
                self.gui_set_batch_size_button.pack(side=tkinter.LEFT, padx=2)

            def __make_agent_engines_frame():
                def __make_predator_engine_listbox():
                    self.gui_predator_engine_frame = tkinter.LabelFrame(self.gui_ai_control_engines_frame,
                                                                        text="Predator engines:")
                    self.gui_predator_engine_frame.pack(side=tkinter.LEFT, fill=tkinter.Y, padx=10)

                    self.gui_predator_engine_listbox = tkinter.Listbox(self.gui_predator_engine_frame,
                                                                       exportselection=False,
                                                                       width=35,
                                                                       height=5)
                    self.gui_predator_engine_listbox.pack(anchor=tkinter.CENTER, padx=5)
                    for i, (name, engine) in zip(range(len(self.predator_ai_engines_list)), self.predator_ai_engines_list):
                        self.gui_predator_engine_listbox.insert(i, name)

                    def _callback_select_predator_engine_listbox(event):
                        selection = event.widget.curselection()
                        if selection and int(selection[0]) != self.selected_predator_engine_index:
                            self.gui_predator_engine_save_weights_folder_entry.delete(0, 'end')
                            self.gui_predator_engine_save_weights_folder_entry.insert(0, "weights_predator_" +
                                                                  self.predator_ai_engines_list[int(selection[0])][0])
                            self.gui_predator_engine_save_weights_set_button.invoke()
                            self.selected_predator_engine_index = int(selection[0])
                            print("Selected new predator engine = ",
                                  self.predator_ai_engines_list[self.selected_predator_engine_index])
                            self.selected_predator_ai_engine_changed = True

                    self.gui_predator_engine_listbox.bind('<<ListboxSelect>>', _callback_select_predator_engine_listbox)

                    if self.predator_ai_engines_list:
                        self.gui_predator_engine_listbox.select_set(0)

                def __make_victim_engine_listbox():
                    self.gui_victim_engine_frame = tkinter.LabelFrame(self.gui_ai_control_engines_frame,
                                                                      text="Victim engines:")
                    self.gui_victim_engine_frame.pack(side=tkinter.RIGHT, fill=tkinter.Y, padx=10)

                    self.gui_victim_engine_listbox = tkinter.Listbox(self.gui_victim_engine_frame,
                                                                     exportselection=False,
                                                                     width=35,
                                                                     height=5)
                    self.gui_victim_engine_listbox.pack(anchor=tkinter.CENTER, padx=5)
                    for i, (name, engine) in zip(range(len(self.victim_ai_engines_list)), self.victim_ai_engines_list):
                        self.gui_victim_engine_listbox.insert(i, name)

                    def _callback_select_victim_engine_listbox(event):
                        selection = event.widget.curselection()
                        if selection and int(selection[0]) != self.selected_victim_engine_index:
                            self.gui_victim_engine_save_weights_folder_entry.delete(0, 'end')
                            self.gui_victim_engine_save_weights_folder_entry.insert(0, "weights_victim_" +
                                                                  self.victim_ai_engines_list[int(selection[0])][0])
                            self.gui_victim_engine_save_weights_set_button.invoke()
                            self.selected_victim_engine_index = int(selection[0])
                            print("Selected new victim engine = ", self.victim_ai_engines_list[self.selected_victim_engine_index])
                            self.selected_victim_ai_engine_changed = True

                    self.gui_victim_engine_listbox.bind('<<ListboxSelect>>', _callback_select_victim_engine_listbox)

                    if self.victim_ai_engines_list:
                        self.gui_victim_engine_listbox.select_set(0)

                def __make_save_weights_frequency_frame():
                    self.gui_save_weights_frequency_frame = tkinter.Frame(self.gui_ai_control_frame)
                    self.gui_save_weights_frequency_frame.pack(anchor=tkinter.NW, padx=10)

                    self.gui_save_weights_label = tkinter.Label(self.gui_save_weights_frequency_frame,
                                                                text="Save engine's weights each N episodes, N = ")
                    self.gui_save_weights_label.pack(side=tkinter.LEFT)

                    self.gui_save_weights_count_entry = tkinter.Entry(self.gui_save_weights_frequency_frame, width=7)
                    self.gui_save_weights_count_entry.insert(0, self.parameter_save_weights_episode_count)
                    self.gui_save_weights_count_entry.pack(side=tkinter.LEFT)

                    def _callback_set_save_weight_frequency():
                        self.parameter_save_weights_episode_count = int(self.gui_save_weights_count_entry.get())
                        print("New save weights frequency = ", self.parameter_save_weights_episode_count)
                    self.gui_callback_functions.append(_callback_set_save_weight_frequency)

                    self.gui_save_weights_set_frequency_button = tkinter.Button(self.gui_save_weights_frequency_frame,
                                                                                text="OK",
                                                                                command=_callback_set_save_weight_frequency)
                    self.gui_save_weights_set_frequency_button.pack(side=tkinter.LEFT)

                def __make_predator_engine_save_weights_frame():
                    self.gui_predator_engine_save_weights_frame = tkinter.Frame(self.gui_ai_control_frame)
                    self.gui_predator_engine_save_weights_frame.pack(anchor=tkinter.NW, padx=10)

                    self.gui_predator_engine_save_weights_booleanVar = tkinter.BooleanVar()
                    self.gui_predator_engine_save_weights_booleanVar.set(self.parameter_predator_engine_save_weights)

                    def _callback_predator_engine_save_weights():
                        self.parameter_predator_engine_save_weights = self.gui_predator_engine_save_weights_booleanVar.get()
                        print("Predator engine save weights = ", self.parameter_predator_engine_save_weights)
                    self.gui_callback_functions.append(_callback_predator_engine_save_weights)

                    self.gui_predator_engine_save_weights_checkbox = tkinter.Checkbutton(
                                                            self.gui_predator_engine_save_weights_frame,
                                                            text="Save predator engine weights. Destination folder = ",
                                                            variable=self.gui_predator_engine_save_weights_booleanVar,
                                                            onvalue=True,
                                                            offvalue=False,
                                                            command=_callback_predator_engine_save_weights)
                    self.gui_predator_engine_save_weights_checkbox.pack(side=tkinter.LEFT)

                    self.gui_predator_engine_save_weights_folder_entry = tkinter.Entry(self.gui_predator_engine_save_weights_frame,
                                                                                       width=25)
                    if len(self.gui_predator_engine_listbox.curselection()):
                        self.gui_predator_engine_save_weights_directory_name = "weights_predator_" + self.predator_ai_engines_list[int(self.gui_predator_engine_listbox.curselection()[0])][0]
                    else:
                        self.gui_predator_engine_save_weights_directory_name = "Engine is not selected!"
                        self.gui_predator_engine_save_weights_checkbox.deselect()
                    self.gui_predator_engine_save_weights_folder_entry.insert(0, self.gui_predator_engine_save_weights_directory_name)
                    self.gui_predator_engine_save_weights_folder_entry.pack(side=tkinter.LEFT)

                    def _entry_save_weights_set_button_callback():
                        self.gui_predator_engine_save_weights_directory_name = self.gui_predator_engine_save_weights_folder_entry.get()
                        print("Predator engine save weights directory name = ", self.gui_predator_engine_save_weights_directory_name)

                    self.gui_predator_engine_save_weights_set_button = tkinter.Button(self.gui_predator_engine_save_weights_frame,
                                                                                      text="OK",
                                                                                      command=_entry_save_weights_set_button_callback)
                    self.gui_predator_engine_save_weights_set_button.pack(side=tkinter.LEFT)

                def __make_victim_engine_save_weights_frame():
                    self.gui_victim_engine_save_weights_frame = tkinter.Frame(self.gui_ai_control_frame)
                    self.gui_victim_engine_save_weights_frame.pack(anchor=tkinter.NW, padx=10)

                    self.gui_victim_engine_save_weights_booleanVar = tkinter.BooleanVar()
                    self.gui_victim_engine_save_weights_booleanVar.set(self.parameter_victim_engine_save_weights)

                    def _callback_victim_engine_save_weights():
                        self.parameter_victim_engine_save_weights = self.gui_victim_engine_save_weights_booleanVar.get()
                        print("Victim engine save weights = ", self.parameter_victim_engine_save_weights)
                    self.gui_callback_functions.append(_callback_victim_engine_save_weights)

                    self.gui_victim_engine_save_weights_checkbox = tkinter.Checkbutton(
                                                                    self.gui_victim_engine_save_weights_frame,
                                                                    text="Save victim engine weights. Destination folder = ",
                                                                    variable=self.gui_victim_engine_save_weights_booleanVar,
                                                                    onvalue=True,
                                                                    offvalue=False,
                                                                    command=_callback_victim_engine_save_weights)
                    self.gui_victim_engine_save_weights_checkbox.pack(side=tkinter.LEFT)

                    self.gui_victim_engine_save_weights_folder_entry = tkinter.Entry(self.gui_victim_engine_save_weights_frame,
                                                                                     width=25)
                    if len(self.gui_victim_engine_listbox.curselection()):
                        self.gui_victim_engine_save_weights_directory_name = "weights_victim_" + self.victim_ai_engines_list[int(self.gui_victim_engine_listbox.curselection()[0])][0]
                    else:
                        self.gui_victim_engine_save_weights_directory_name = "Engine is not selected!"
                        self.gui_victim_engine_save_weights_checkbox.deselect()
                    self.gui_victim_engine_save_weights_folder_entry.insert(0, self.gui_victim_engine_save_weights_directory_name)
                    self.gui_victim_engine_save_weights_folder_entry.pack(side=tkinter.LEFT)

                    def _entry_save_weights_set_button_callback():
                        self.gui_victim_engine_save_weights_directory_name = self.gui_victim_engine_save_weights_folder_entry.get()
                        print("Victim engine save weights directory name = ", self.gui_victim_engine_save_weights_directory_name)

                    self.gui_victim_engine_save_weights_set_button = tkinter.Button(
                                                                            self.gui_victim_engine_save_weights_frame,
                                                                            text="OK",
                                                                            command=_entry_save_weights_set_button_callback)
                    self.gui_victim_engine_save_weights_set_button.pack(side=tkinter.LEFT)

                def __make_load_weights_frame():
                    self.gui_load_weights_frame = tkinter.Frame(self.gui_ai_control_frame)
                    self.gui_load_weights_frame.pack(anchor=tkinter.NW, padx=10, pady=5)

                    self.gui_load_weights_label = tkinter.Label(self.gui_load_weights_frame,
                                                                text="Load weights by file dialog: ")
                    self.gui_load_weights_label.pack(side=tkinter.LEFT, padx=5)

                    def _callback_load_weights_predator():
                        self.gui_load_weights_directory_name_predator_filedialog = tkinter.filedialog.askopenfile(filetypes=[("HDF5 files", ".h5")])
                        if self.gui_load_weights_directory_name_predator_filedialog:
                            self.load_predator_engine_weights_filepath = self.gui_load_weights_directory_name_predator_filedialog.name
                            print("Load predator engine weights filepath = ", self.load_predator_engine_weights_filepath)
                            self.ready_to_load_weights_predator = True

                    self.gui_load_weights_predator_button = tkinter.Button(self.gui_load_weights_frame,
                                                                           text="Predator",
                                                                           command=_callback_load_weights_predator)
                    self.gui_load_weights_predator_button.pack(side=tkinter.LEFT)

                    def _callback_load_weights_victim():
                        self.gui_load_weights_directory_name_victim_filedialog = tkinter.filedialog.askopenfile(filetypes=[("HDF5 files", ".h5")])
                        if self.gui_load_weights_directory_name_victim_filedialog:
                            self.load_victim_engine_weights_filepath = self.gui_load_weights_directory_name_victim_filedialog.name
                            print("Load victim engine weights filepath = ", self.load_victim_engine_weights_filepath)
                            self.ready_to_load_weights_victim = True

                    self.gui_load_weights_victim_button = tkinter.Button(self.gui_load_weights_frame,
                                                                         text="Victim",
                                                                         command=_callback_load_weights_victim)
                    self.gui_load_weights_victim_button.pack(side=tkinter.LEFT, padx=5)

                self.gui_ai_control_engines_frame = tkinter.LabelFrame(self.gui_ai_control_frame, text="AI engines")
                self.gui_ai_control_engines_frame.pack(anchor=tkinter.NW, padx=10)

                __make_predator_engine_listbox()
                __make_victim_engine_listbox()
                __make_save_weights_frequency_frame()
                __make_predator_engine_save_weights_frame()
                __make_victim_engine_save_weights_frame()
                __make_load_weights_frame()

            self.gui_ai_control_frame = tkinter.LabelFrame(self.gui_prepared_frame, text="AI control")
            self.gui_ai_control_frame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

            __make_manual_control_checkbox()
            __make_play_collision_sound_checkbox()
            __make_turn_ai_control_checkbox()
            __make_turn_ai_learning_checkbox()
            __make_logging_checkbox()
            __make_remember_frequency_frame()
            __make_batch_size_frame()
            __make_agent_engines_frame()
        __make_ai_control_frame()

        def __make_game_summary_frame():
            def __make_current_episode_frame():
                self.gui_current_episode_frame = tkinter.Frame(self.gui_game_summary_frame)
                self.gui_current_episode_frame.pack(anchor=tkinter.NW)

                self.gui_current_episode_meta_label = tkinter.Label(self.gui_current_episode_frame,
                                                                    text="Current episode = ")
                self.gui_current_episode_value_label = tkinter.Label(self.gui_current_episode_frame,
                                                                     text=str(self.current_episode))
                self.gui_current_episode_meta_label.pack(side=tkinter.LEFT)
                self.gui_current_episode_value_label.pack(side=tkinter.LEFT)

            def __make_success_episodes_count_frame():
                self.gui_success_episodes_count_frame = tkinter.Frame(self.gui_game_summary_frame)
                self.gui_success_episodes_count_frame.pack(anchor=tkinter.NW)

                self.gui_success_episodes_count_meta_label = tkinter.Label(self.gui_success_episodes_count_frame,
                                                                           text="Success episodes count = ")
                self.gui_success_episodes_count_value_label = tkinter.Label(self.gui_success_episodes_count_frame,
                                                                            text=str(self.success_episodes_count))
                self.gui_success_episodes_count_meta_label.pack(side=tkinter.LEFT)
                self.gui_success_episodes_count_value_label.pack(side=tkinter.LEFT)

            self.gui_game_summary_frame = tkinter.LabelFrame(self.gui_prepared_frame, text="Game summary")
            self.gui_game_summary_frame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

            __make_current_episode_frame()
            __make_success_episodes_count_frame()
        __make_game_summary_frame()

        for gui_callback_function in self.gui_callback_functions:
            gui_callback_function()

    def start_threads(self):
        self.started = True

        self.game_ai_thread.start()
        self.__gui_thread()
        self.game_ai_thread.join()

    def __game_ai_thread(self):
        core_done = False
        once_for_render = True

        self.current_ai_agent_predator = self.predator_ai_engines_list[self.selected_predator_engine_index][1]
        self.current_ai_agent_victim = self.victim_ai_engines_list[self.selected_victim_engine_index][1]
        self.current_ai_agent_predator.init()
        self.current_ai_agent_victim.init()

        self.current_episode = 1
        self.success_episodes_count = 0
        self.success = False
        self.current_predator_reward = 0
        self.current_victim_reward = 0
        self.general_predator_reward = 0
        self.general_victim_reward = 0

        start_observation = self.env_core.reset()
        observation_predator = start_observation["agent_0"]
        observation_victim = start_observation["agent_1"]

        self.success_episodes_count = 0
        remember_frequency_counter = 0
        done_predator = None
        done_victim = None

        while self.started:
            self.env_renderer.draw_direction = self.parameter_draw_direction
            self.env_renderer.draw_trajectory = self.parameter_draw_trajectory
            if self.parameter_is_delay:
                time.sleep(self.parameter_render_delay)

            self.log(" ")
            self.log(" ")
            self.log("*** main iteration, timer is " + ("off" if not self.parameter_show_timer
                                                        else ("on, episode = " + str(self.current_episode))))

            if self.ready_to_load_weights_predator:
                self.ready_to_load_weights_predator = False
                self.current_ai_agent_predator.load_weights(self.load_predator_engine_weights_filepath)
                print("Predator weights loaded")
                self.log("Predator weights loaded")

            if self.ready_to_load_weights_victim:
                self.ready_to_load_weights_victim = False
                self.current_ai_agent_victim.load_weights(self.load_victim_engine_weights_filepath)
                print("Victim weights loaded")
                self.log("Victim weights loaded")

            if self.selected_predator_ai_engine_changed:
                self.selected_predator_ai_engine_changed = False
                self.current_ai_agent_predator.close()
                self.current_ai_agent_predator = self.predator_ai_engines_list[self.selected_predator_engine_index][1]
                self.current_ai_agent_predator.init()
                print("AI predator engine is changed to " + self.predator_ai_engines_list[self.selected_predator_engine_index][0])

            if self.selected_victim_ai_engine_changed:
                self.selected_victim_ai_engine_changed = False
                self.current_ai_agent_victim.close()
                self.current_ai_agent_victim = self.victim_ai_engines_list[self.selected_victim_engine_index][1]
                self.current_ai_agent_victim.init()
                print("AI victim engine is changed to " + self.victim_ai_engines_list[self.selected_victim_engine_index][0])

            self.log("core_done = " + str(core_done))
            if core_done:
                if self.current_episode % 300 == 0:
                    self.make_plots()
                print(self.current_episode)
                self.env_renderer.reset()
                core_done = False

                self.episode_n_gametime.append((self.current_episode, self.env_core.game_current_step))
                self.episode_n_reward_predator.append((self.current_episode, self.general_predator_reward))
                self.episode_n_reward_victim.append((self.current_episode, self.general_victim_reward))

                if self.parameter_turn_ai_learning:
                    if self.env_core.game_current_step < self.env_core.GAME_STEPS and done_predator and done_victim:
                        self.success = True
                        self.success_episodes_count += 1
                        self.episode_n_catchcount.append((self.current_episode, float(self.success_episodes_count) / float(self.current_episode)))
                        self.episode_n_catch_time.append((self.current_episode, self.env_core.game_current_step))
                        if self.parameter_play_collision_sound:
                            self.env_renderer.play_sound("Cannon2.wav")

                    self.log("success = " + str(self.success))

                    self.current_episode += 1
                    self.success = False
                    self.general_predator_reward = 0
                    self.general_victim_reward = 0

                    if self.parameter_predator_engine_save_weights and self.current_episode % self.parameter_save_weights_episode_count == 0:
                        self.log("saving weights predator: " + self.gui_predator_engine_save_weights_directory_name + str(self.current_episode))
                        print("saving weights predator: " + self.gui_predator_engine_save_weights_directory_name + str(self.current_episode))

                        if not os.path.exists(self.gui_predator_engine_save_weights_directory_name):
                            os.mkdir(self.gui_predator_engine_save_weights_directory_name)

                        self.current_ai_agent_predator.save_weights("./" + self.gui_predator_engine_save_weights_directory_name +
                                                                    '/' + self.gui_predator_engine_save_weights_directory_name +
                                                                    str(self.current_episode))
                    if self.parameter_victim_engine_save_weights and self.current_episode % self.parameter_save_weights_episode_count == 0:
                        self.log("saving weights victim: " + self.gui_victim_engine_save_weights_directory_name + str(self.current_episode))
                        print("saving weights victim: " + self.gui_victim_engine_save_weights_directory_name + str(self.current_episode))

                        if not os.path.exists(self.gui_victim_engine_save_weights_directory_name):
                            os.mkdir(self.gui_victim_engine_save_weights_directory_name)

                        self.current_ai_agent_victim.save_weights("./" + self.gui_victim_engine_save_weights_directory_name +
                                                                    '/' + self.gui_victim_engine_save_weights_directory_name +
                                                                    str(self.current_episode))

                if self.parameter_start_position_random:
                    start_observation = self.env_core.reset(True)
                    observation_predator = start_observation["agent_0"]
                    observation_victim = start_observation["agent_1"]
                else:
                    start_observation = self.env_core.reset(False,
                                                            self.parameter_predator_start_position,
                                                            self.parameter_victim_start_position)
                    observation_predator = start_observation["agent_0"]
                    observation_victim = start_observation["agent_1"]
                self.env_core.turn_timer = self.parameter_show_timer

                if self.parameter_turn_ai_learning:
                    self.log("start AI training: ")
                    self.current_ai_agent_predator.train(self.parameter_batch_size)
                    self.current_ai_agent_victim.train(self.parameter_batch_size)
            else:
                self.log("*** start AI instructions ******")

                manual_action_predator = 0
                manual_action_victim = 0
                if self.parameter_turn_manual_control:
                    manual_action_predator, manual_action_victim = self.__manual_control()

                agent_action_predator = 0
                agent_action_victim = 0
                if self.parameter_turn_ai_control:
                    agent_action_predator = self.current_ai_agent_predator.act(observation_predator)
                    agent_action_victim = self.current_ai_agent_victim.act(observation_victim)

                action_predator = manual_action_predator if manual_action_predator != 0 else agent_action_predator
                action_victim = manual_action_victim if manual_action_victim != 0 else agent_action_victim

                new_observation, new_rewards, new_dones, new_infos = self.env_core.step({
                    "agent_0": action_predator,
                    "agent_1": action_victim
                })

                old_observation_predator = observation_predator
                old_observation_victim = observation_victim

                self.current_observation = new_observation
                observation_predator = new_observation["agent_0"]
                observation_victim = new_observation["agent_1"]
                reward_predator = new_rewards["agent_0"]
                reward_victim = new_rewards["agent_1"]

                self.current_predator_reward = reward_predator
                self.current_victim_reward = reward_victim

                done_predator = new_dones["agent_0"]
                done_victim = new_dones["agent_1"]
                core_done = new_dones["__all__"]

                remember_frequency_counter += 1
                if core_done or remember_frequency_counter >= self.parameter_remember_frequency:
                    remember_frequency_counter = 0
                    if self.parameter_turn_ai_learning and self.parameter_turn_ai_control:
                        self.log("* start AI remembering: ")
                        self.general_predator_reward += reward_predator
                        self.general_victim_reward += reward_victim
                        self.current_ai_agent_predator.remember(
                            old_observation_predator,
                            observation_predator,
                            action_predator,
                            reward_predator,
                            done_predator
                        )
                        self.current_ai_agent_victim.remember(
                            old_observation_victim,
                            observation_victim,
                            action_victim,
                            reward_victim,
                            done_victim
                        )

                self.log("*** end AI instructions ******")

                if self.parameter_is_render:
                    once_for_render = True
                    self.env_core.render()
                elif once_for_render:
                    once_for_render = False
                    self.env_renderer.clear_screen()
        else:
            self.current_ai_agent_predator.close()
            self.current_ai_agent_victim.close()
        self.make_plots()
        print("FINISH!!!")

    def __gui_thread(self):
        while self.started:
            self.gui_root.update_idletasks()
            self.gui_root.update()
            self.__update_gui()
        else:
            self.gui_root.destroy()

    def __update_gui(self):
        self.parameter_render_delay = self.gui_delay_scale.get()
        self.parameter_remember_frequency = self.gui_remember_frequency_scale.get()

        if self.parameter_show_observation_info:
            self.gui_observation_info_predator_position_value_label['text'] = str(
                [format(round(i, 5), '.5f') for i in self.current_observation["agent_0"][1:3]])
            self.gui_observation_info_predator_velocity_value_label['text'] = str(
                [format(round(i, 3), '.3f') for i in self.current_observation["agent_0"][5:7]])
            self.gui_observation_info_victim_position_value_label['text'] = str(
                [format(round(i, 5), '.5f') for i in self.current_observation["agent_0"][3:5]])
            self.gui_observation_info_victim_velocity_value_label['text'] = str(
                [format(round(i, 3), '.3f') for i in self.current_observation["agent_0"][7:9]])
            self.gui_observation_info_predator_reward_value_label['text'] = str(self.current_predator_reward)
            self.gui_observation_info_victim_reward_value_label['text'] = str(self.current_victim_reward)

        if self.parameter_show_timer:
            self.env_core.GAME_STEPS = self.gui_timer_scale.get()
            self.gui_timer_count_value_label['text'] = str(self.env_core.game_current_step) + '/' + str(self.env_core.GAME_STEPS)

        self.gui_current_episode_value_label['text'] = str(self.current_episode)
        self.gui_success_episodes_count_value_label['text'] = str(self.success_episodes_count)

    def __event_on_closing(self):
        self.started = False
        self.game_ai_thread.join()
        self.make_plots()
        print("The program is finished...")

    def __manual_control(self):
        manual_action_predator = 0
        manual_action_victim = 0
        keys = pygame.key.get_pressed()

        if keys[pygame.K_KP1]:
            manual_action_predator = 1
        elif keys[pygame.K_KP4]:
            manual_action_predator = 2
        elif keys[pygame.K_KP7]:
            manual_action_predator = 3
        elif keys[pygame.K_KP8]:
            manual_action_predator = 4
        elif keys[pygame.K_KP9]:
            manual_action_predator = 5
        elif keys[pygame.K_KP6]:
            manual_action_predator = 6
        elif keys[pygame.K_KP3]:
            manual_action_predator = 7
        elif keys[pygame.K_KP2]:
            manual_action_predator = 8

        if keys[pygame.K_z]:
            manual_action_victim = 1
        elif keys[pygame.K_a]:
            manual_action_victim = 2
        elif keys[pygame.K_q]:
            manual_action_victim = 3
        if keys[pygame.K_w]:
            manual_action_victim = 4
        elif keys[pygame.K_e]:
            manual_action_victim = 5
        elif keys[pygame.K_d]:
            manual_action_victim = 6
        elif keys[pygame.K_c]:
            manual_action_victim = 7
        elif keys[pygame.K_x]:
            manual_action_victim = 8

        return manual_action_predator, manual_action_victim

    def log(self, text):
        if self.parameter_turn_ai_control:
            if self.parameter_is_logging:
                self.log_file.write(text + "\n")

    def make_plots(self):
        import matplotlib.pyplot as plt

        def smooth(y, cutoff=2000, fs=500000 * 0.20):
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

        fig, axs = plt.subplots(2, 2, figsize=(11, 7))

        axs[0, 0].plot(
            [i for (i, j) in self.episode_n_catchcount],
            smooth([(j * 100) for (i, j) in self.episode_n_catchcount]),
        )
        axs[0, 0].set_title('Кількість зустрічів')
        axs[0, 0].set_xlabel('Номер епізоду')
        axs[0, 0].set_ylabel('% епізодів')
        axs[0, 0].legend(loc="best")

        axs[0, 1].plot(
            [i for (i, j) in self.episode_n_catch_time],
            smooth([j for (i, j) in self.episode_n_catch_time]),
            'tab:orange')
        axs[0, 1].set_title('Тривалість епізоду')
        axs[0, 1].set_xlabel('Номер епізоду')
        axs[0, 1].set_ylabel('Кількість кадрів')
        axs[0, 1].legend(loc="best")

        axs[1, 0].plot(
            [i for (i, j) in self.episode_n_reward_victim],
            smooth([j for (i, j) in self.episode_n_reward_victim]),
            'tab:green')
        axs[1, 0].set_title('Сумарна нагорода за епізод (жертва)')
        axs[1, 0].set_xlabel('Номер епізоду')
        axs[1, 0].set_ylabel('Сумарна нагорода')
        axs[1, 0].legend(loc="best")

        axs[1, 1].plot(
            [i for (i, j) in self.episode_n_reward_predator],
            smooth([j for (i, j) in self.episode_n_reward_predator]),
            'tab:gray')
        axs[1, 1].set_title('Сумарна нагорода за епізод (хижак)')
        axs[1, 1].set_xlabel('Номер епізоду')
        axs[1, 1].set_ylabel('Сумарна нагорода')
        axs[1, 1].legend(loc="best")

        import pickle
        path = "experiments/tr/"
        with open(path + "episode_n_catchcount", 'wb') as f:
            pickle.dump(self.episode_n_catchcount, f)
        with open(path + "episode_n_catch_time", 'wb') as f:
            pickle.dump(self.episode_n_catch_time, f)
        with open(path + "episode_n_reward_victim", 'wb') as f:
            pickle.dump(self.episode_n_reward_victim, f)
        with open(path + "episode_n_reward_predator", 'wb') as f:
            pickle.dump(self.episode_n_reward_predator, f)

        plt.show()
        plt.close()

