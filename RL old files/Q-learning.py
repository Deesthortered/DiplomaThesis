import gym         # подгружаем библиотеку с нашими игровыми окружениями
import numpy as np # ну и просто математическую либу

# Тут описаны константы (гиперпараметры), которые использует алгоритм Q-learning
LEARNING_RATE = 0.1 # Скорость обучения - гиперпараметр, который указывает насколько сильно будет влиять новый опыт на обучение агента
DISCOUNT = 0.95     # Скидка (хз как назвать) - гиперпараметр, который указывает насколько сильно вгент будет ценить последующие вознаграждения по сравнению с текущим
EPISODES = 10000    # Количество эпизодов, в которых будет обучаться агент.
PRINT_CYCLE = 100  # Это константа, которая означает через сколько эпизодов выводить информацию об процессе. Нужна что бы не через раз узнавать инфу, ибо так часто это делать бесполезно.

def Game(): # Всю работу пихану в функцию, что бы можно было бы на Jypiter/Hydrogen-е запускать сразу
    env = gym.make("MountainCar-v0") # Подгружаем нашу среду, в которой будет обучаться агент.

    epsilone = 0.5
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // 2
    epsilone_decay_value = epsilone/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
    print("epsilone_decay_value = ", epsilone_decay_value)

    # Для обучения агента по алгоритму Q-learning, ему нужна таблица Q-table, которая будет выражать функцию Q(s, a) - функцию оценки действий агента в конкретном состоянии.
    # Много слов в книгах сказано что она нужна, но толком не рассказывали как ее делать и откуда она взялась, потому опишу подробно.
    # По факту - это функция (отображение) множеств состояний S и действий A(s) во множество Q, которое есть подмножеством R. Проще говоря - каждой возможной комбинации всех состояний
    # и действий задаем число, которое будет означать насколько хорошим является данное действие в данном состоянии. Изначально это случайное число, которое потом в процессе обучения
    # будет изменяться.
    # Состояния агента выражаются через его восприятие (obervation, пространство наблюдения - вектор чисел описывающие разные величины), а действия - непосредственно, пронумерованы числами.
    # Проблема в том, что множество состояний может быть континуальным и большого диапазона и надо что то с этим делать. Для этого надо его дискретизировать, но важно как именно это делать и
    # зависит это от самого окружения. Надо найти компромисс между точностью дискретизации (чем точнее, тем лучше обучится агент) и размером таблицы (чем меньше, тем быстрее обучится агент).

    # Конкретно в этом случае у нас obervation состоит из следующего количества вещественных чисел:
    print("Количество переменных = ", env.observation_space)
    # И эти переменные имеют следующие верхние границы:
    print("Верхние границы = ", env.observation_space.high)
    # И нижние границы:
    print("Нижние границы = ", env.observation_space.low)

    # Создаем нашу таблицу и промежуточные переменные
    DISCRETE_SIZE = [20] * len(env.observation_space.high)   # создаем массив 20-ток длинною в пространство наблюдения
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_SIZE   # массив дискретных шагов для каждой перменной пространства наблюдений
    q_table = np.random.uniform(low = -2, high = 0, size=(DISCRETE_SIZE + [env.action_space.n]))   # создаем многомерную таблицу (тензор если хотите) и заполняем случайными числами из [-2, 0]
    print("Шаг дискретности: ", discrete_os_win_size)
    print("Размерность Q-таблицы: ", q_table.shape)

    # Вспомогательная функция, которая будет превращать состояние в непрервном виде в дискретное - а именно в индексы массива Q-таблицы
    def getDicreteState(state):
        discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(discrete_state.astype(np.int))

    # Будем запускать игру много раз, где каждый раз будет называться эпизодом
    for episode in range(EPISODES):
        # Инициализируем необходимые переменные, получаем начальное состояние. env.reset() - сбрасвает окружение до стартового положения и возвращает заодно наблюдения.
        done = False
        discrete_state = getDicreteState(env.reset())

        # Тут м проверяем не пора ли вывести инфу
        render = False
        if episode % PRINT_CYCLE == 0:
            print(episode, epsilone)
            render = True

        # Запускаем цикл в котором мы будем на каждой итерации получать "снимок" среды и выполнять соответствующие действия до тех пор, пока игра не завершится
        # Игра долго не будет происходить, потому и нужны эпизоды
        while not done:

            if np.random.random() > 0.1:
                action = np.argmax(q_table[discrete_state]); # выбираем найлучшее действие для текущего состояния
            else:
                action = np.random.randint(0, env.action_space.n)
            new_state, reward, done, info = env.step(action) # выполняем действие и ловим фидбек от среды
            new_discrete_state = getDicreteState(new_state) # дискретизируем полученное новое состояние

            if render:
                env.render() # выводим на экран происходящее

            if not done: # если процесс еще не закончился
                max_future_q = np.max(q_table[new_discrete_state]) # получаем максимальную оценку из таблицы, которую может дать следующее состояние
                current_q = q_table[discrete_state + (action, )] # получаем оценку сделаного действия из таблицы
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q) # по главной формуле получаем новую оценку для сделаного действия
                q_table[discrete_state + (action, )] = new_q # и обновляем ее в таблице
            elif new_state[0] >= env.goal_position: # если процесс окончился и мы удостоверились, что мы пришли в искомое состояние (а именно чекаем положение телеги)
                print("Цель достигнута на", episode, "эпизоде")
                q_table[discrete_state + (action, )] = 0 # то этому состоянию + действию задаем максимальное вознаграждение. Оно определяется чисто по условию задачи - тут максимум равен нулю.
                                                         # То есть игра постоянно "наказывает" до тех пор, пока агент не выполнит необходимое условие и награда = 0 - то есть не наказывать.
            discrete_state = new_discrete_state # обновляем состояние

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilone -= epsilone_decay_value

    env.close() # все сделано, потому можем вырубать среду

if __name__ == "__main__": # Ну собсна запускаем тут работу
    Game()
