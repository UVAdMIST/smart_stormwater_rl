import numpy as np


class ReplayStacker:
    def __init__(self, columns, window_length=100):
        self._data = np.zeros((window_length, columns))
        self.capacity = window_length
        self.size = 0
        self.columns = columns

    def update(self, x):
        self._add(x)

    def _add(self, x):
        if self.size == self.capacity:
            self._data = np.roll(self._data, -1)
            self._data[self.size-1, :] = x
        else:
            self._data[self.size, :] = x
            self.size += 1

    def data(self):
        return self._data[0:self.size, :]


class ReplayMemoryAgent:
    def __init__(self, states_len, replay_window):
        self.states_len = states_len
        self.replay_window = replay_window

        # Initialize replay memory
        self.replay_memory = {'states': ReplayStacker(self.states_len,
                                                      self.replay_window),
                              'states_new': ReplayStacker(self.states_len,
                                                          self.replay_window),
                              'rewards': ReplayStacker(1,
                                                       self.replay_window),
                              'actions': ReplayStacker(1,
                                                       self.replay_window),
                              'terminal': ReplayStacker(1,
                                                        self.replay_window)}

    def replay_memory_update(self, states, states_new, rewards, actions, terminal):
        self.replay_memory['rewards'].update(rewards)
        self.replay_memory['states'].update(states)
        self.replay_memory['states_new'].update(states_new)
        self.replay_memory['actions'].update(actions)
        self.replay_memory['terminal'].update(terminal)
