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
    def __init__(self, states_len, action_space, replay_window):
        self.states_len = states_len
        self.action_space = action_space
        self.replay_window = replay_window
        # Initialize replay memory
        self.replay_memory = {'states': ReplayStacker(self.states_len, self.replay_window),
                              'states_new': ReplayStacker(self.states_len, self.replay_window),
                              'rewards': ReplayStacker(1, self.replay_window),
                              'actions': ReplayStacker(self.action_space, self.replay_window),
                              'terminal': ReplayStacker(1, self.replay_window)}

    def replay_memory_update(self, states, states_new, rewards, actions, terminal):
        self.replay_memory['rewards'].update(rewards)
        self.replay_memory['states'].update(states)
        self.replay_memory['states_new'].update(states_new)
        self.replay_memory['actions'].update(actions)
        self.replay_memory['terminal'].update(terminal)


def random_indx(sample_size, replay_size):
    # get indices randomly
    indx = np.linspace(0, replay_size-1, sample_size)
    indx = np.random.choice(indx, sample_size, replace=False)
    indx.tolist()
    indx = list(map(int, indx))
    return indx


def create_minibatch(random_index, memory_agent, batch_size, action_space):
    indx = random_index
    states_len = memory_agent.replay_memory['states'].data().shape[1]

    # create minibatch dict structure
    training_batch = {'states': np.zeros((batch_size, states_len)),
                      'states_new': np.zeros((batch_size, states_len)),
                      'actions': np.zeros((batch_size, action_space)),
                      'rewards': np.zeros((batch_size, 1)),
                      'terminal': np.zeros((batch_size, 1))}

    for i in training_batch.keys():
        temp = memory_agent.replay_memory[i].data()
        training_batch[i] = temp[indx]

    return training_batch
