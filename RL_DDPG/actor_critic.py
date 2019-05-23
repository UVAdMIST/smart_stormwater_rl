"""
Creates RL algorithm classes

Author: Benjamin Bowes
Date: May 10, 2019

Based on code from:
https://github.com/kLabUM/rl-storm-control/blob/master/flux_training_code/flux_version_system_run/pond_net.py
and
https://github.com/germain-hug/Deep-RL-Keras/blob/master/DDPG/actor.py
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Reshape, Lambda, BatchNormalization, GaussianNoise, Flatten, Activation, Dropout
from keras.optimizers import RMSprop, Adam
from keras.initializers import RandomUniform

np.random.seed(1)


class Actor:  # Actor Network for the DDPG Algorithm
    def __init__(self, action_value_model, target_model, states_len, replay_memory, policy, tau,
                 batch_size=64, target_update=10000, train=True):

        self.states_len = states_len
        self.ac_model = action_value_model
        self.target_model = target_model
        self.replay = replay_memory
        self.batch_size = batch_size
        self.policy = policy
        self.train = train
        self.target_update = target_update
        self.tau = tau

        self.state_vector = np.zeros((1, self.states_len))
        self.state_new_vector = np.zeros((1, self.states_len))
        self.rewards_vector = np.zeros(1)
        self.terminal_vector = np.zeros(1)
        self.action_vector = np.zeros(1)

        self.training_batch = {'states': np.zeros((self.batch_size,
                                                   self.states_len)),
                               'states_new': np.zeros((self.batch_size,
                                                       self.states_len)),
                               'actions': np.zeros((self.batch_size, 1)),
                               'rewards': np.zeros((self.batch_size, 1)),
                               'terminal': np.zeros((self.batch_size, 1))}

    def _random_sample(self):
        indx = randombatch(self.batch_size, len(self.replay['states'].data()))
        for i in self.training_batch.keys():
            temp = self.replay[i].data()
            self.training_batch[i] = temp[indx]

    def _update_target_model(self):  # transfer weights to target model with a factor of tau
        W, target_W = self.ac_model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def _train(self):
        temp_states_new = self.training_batch['states_new']
        temp_states = self.training_batch['states']
        temp_rewards = self.training_batch['rewards']
        temp_terminal = self.training_batch['terminal']
        temp_actions = self.training_batch['actions']
        q_values_train_next = self.target_model.predict_on_batch(temp_states_new)
        target = self.ac_model.predict_on_batch(temp_states)
        for i in range(self.batch_size):
            action_idx = int(temp_actions[i])
            if temp_terminal[i]:
                target[i][action_idx] = temp_rewards[i]
            else:
                target[i][action_idx] = temp_rewards[i] + 0.99 * np.max(
                    q_values_train_next[i])

        self.ac_model.fit(temp_states, target, batch_size=64, epochs=1, verbose=0)

    def train_q(self, update):
        self._random_sample()
        if update:
            self._update_target_model()
        self._train()


class Critic:  # Critic Network for the DDPG Algorithm
    def __init__(self, action_value_model, target_model, states_len, replay_memory, policy, tau,
                 batch_size=64, target_update=10000, train=True):

        self.states_len = states_len
        self.ac_model = action_value_model
        self.target_model = target_model
        self.replay = replay_memory
        self.batch_size = batch_size
        self.policy = policy
        self.train = train
        self.target_update = target_update
        self.tau = tau

        self.state_vector = np.zeros((1, self.states_len))
        self.state_new_vector = np.zeros((1, self.states_len))
        self.rewards_vector = np.zeros(1)
        self.terminal_vector = np.zeros(1)
        self.action_vector = np.zeros(1)

        self.training_batch = {'states': np.zeros((self.batch_size,
                                                   self.states_len)),
                               'states_new': np.zeros((self.batch_size,
                                                       self.states_len)),
                               'actions': np.zeros((self.batch_size, 1)),
                               'rewards': np.zeros((self.batch_size, 1)),
                               'terminal': np.zeros((self.batch_size, 1))}

    def _random_sample(self):
        indx = randombatch(self.batch_size, len(self.replay['states'].data()))
        for i in self.training_batch.keys():
            temp = self.replay[i].data()
            self.training_batch[i] = temp[indx]

    def _update_target_model(self):  # transfer weights to target model with a factor of tau
        W, target_W = self.ac_model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def _train(self):
        temp_states_new = self.training_batch['states_new']
        temp_states = self.training_batch['states']
        temp_rewards = self.training_batch['rewards']
        temp_terminal = self.training_batch['terminal']
        temp_actions = self.training_batch['actions']
        q_values_train_next = self.target_model.predict_on_batch(temp_states_new)
        target = self.ac_model.predict_on_batch(temp_states)
        for i in range(self.batch_size):
            action_idx = int(temp_actions[i])
            if temp_terminal[i]:
                target[i][action_idx] = temp_rewards[i]
            else:
                target[i][action_idx] = temp_rewards[i] + 0.99 * np.max(
                    q_values_train_next[i])

        self.ac_model.fit(temp_states, target, batch_size=64, epochs=1, verbose=0)

    def train_q(self, update):
        self._random_sample()
        if update:
            self._update_target_model()
        self._train()


def build_network(input_states, output_states, hidden_layers, neuron_count, activation_function, dropout):
    # Build and initialize the neural network with a choice for dropout
    model = Sequential()
    model.add(Dense(neuron_count, input_dim=input_states))
    model.add(Activation(activation_function))
    model.add(Dropout(dropout))
    for i_layers in range(0, hidden_layers - 1):
        model.add(Dense(neuron_count))
        model.add(Activation(activation_function))
        model.add(Dropout(dropout))
    model.add(Dense(output_states))
    model.add(Activation('linear'))
    sgd = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


def randombatch(sample_size, replay_size):
    indx = np.linspace(0, replay_size-1, sample_size)
    indx = np.random.choice(indx, sample_size, replace=False)
    indx.tolist()
    indx = list(map(int, indx))
    return indx
