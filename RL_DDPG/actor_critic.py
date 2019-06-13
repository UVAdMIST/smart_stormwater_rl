"""
Create Actor and Critic for Deep Deterministic Policy Gradient RL method

Author: Benjamin Bowes
Date: May 10, 2019

Originally based on code from Hugo Germain:
https://github.com/germain-hug/Deep-RL-Keras/tree/master/DDPG
"""

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten


class Actor:
    def __init__(self, inp_dim, out_dim, act_range, lr, tau):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.act_range = act_range
        self.tau = tau
        self.lr = lr
        self.model = self.network()
        self.target_model = self.network()
        self.adam_optimizer = self.optimizer()

    def network(self):
        """
        Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        inp = Input(shape=self.env_dim)

        x = Dense(100, activation='relu')(inp)
        x = GaussianNoise(1.0)(x)

        # x = Flatten()(x)
        x = Dense(100, activation='relu')(x)
        x = GaussianNoise(1.0)(x)

        out = Dense(self.act_dim, activation='sigmoid', kernel_initializer=RandomUniform())(x)
        # out = Lambda(lambda i: i * self.act_range)(out)  # sigmoid scales to (0,1) so don't need to adjust range

        return Model(inp, out)

    def predict(self, state):
        """ Action prediction
        """
        return self.model.predict(np.expand_dims(state, axis=0))

    def target_predict(self, inp):
        """ Action prediction (target network)
        """
        return self.target_model.predict(inp)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def train(self, states, actions, grads):
        """ Actor Training
        """
        self.adam_optimizer([states, grads])

    def optimizer(self):
        """ Actor Optimizer
        """
        action_gdts = K.placeholder(shape=(None, self.act_dim))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)  # gradients are made negative here for gradient ascent
        grads = zip(params_grad, self.model.trainable_weights)
        return K.function([self.model.input, action_gdts], [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)][1:])

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)


class Critic:
    def __init__(self, inp_dim, out_dim, lr, tau):
        # Dimensions and Hyperparams
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau, self.lr = tau, lr
        # Build critic and target models
        self.model = self.network()
        self.target_model = self.network()
        self.model.compile(Adam(self.lr), 'mse')
        self.target_model.compile(Adam(self.lr), 'mse')
        # Function to compute Q-value gradients (Actor Optimization)
        self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output,
                                                                                               [self.model.input[1]]))

    def network(self):
        """ Assemble Critic network to predict q-values
        """
        state = Input(self.env_dim)
        action = Input((self.act_dim,))
        x = Dense(200, activation='relu')(state)
        # x = concatenate([Flatten()(x), action])
        x = concatenate([x, action])
        x = Dense(200, activation='relu')(x)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform())(x)
        return Model([state, action], out)

    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        return self.action_grads([states, actions])

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.target_model.predict(inp)

    def train_on_batch(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        return self.model.train_on_batch([states, actions], critic_target)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
