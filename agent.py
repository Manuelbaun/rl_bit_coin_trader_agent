from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras import Model
from keras.models import load_model
from keras.layers import Dense, LSTM, concatenate, Input, Flatten, Concatenate
from keras.optimizers import Adam

import math
import numpy as np
from enum import Enum


class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


class Agent:
    def __init__(
        self,
        state_space=10,
        action_space=3,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        max_memory=1000,
        learning_rate=0.001,
        epsilon_decay=0.995,
        name="Trader",
        train_random=False,
    ):
        """
        `state_space`: Inputs, die das DQN Netzwerk aufnimmt \n
        `action_space`: Die Aktionen, die gewählt werden können, hier nur 3, `buy`, `sell`, `hold/sit`\n
        `gamma`: Verzögern für ein kurz- bzw. langfristigen Erfolg\n
        `max_memory`: Das maximum an Erfahrung bevor alte gelöscht werden um platz für neue zu schaffen\n
        `lr`: Learning Rate für den Agenten/Neuronale Netz\n
        """
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_memory = max_memory
        self.lr = learning_rate
        # Der memory Replay Memory Buffer
        self.memory = []
        self.inventory = []
        self.name = name

        self.train_random = train_random

        # Loss, die Genauigkeit des Netzwerks
        self.loss = 0
        # ein Paar Settings für interne Zwecke
        self.total_profit = 0

        # Baue das Model, oder TODO: lade ein bestehendes
        self.model = self.build_network()

    def build_network(self):
        # input1 : open, close, high, low, volume, date..... x mal window_size
        input = Input(shape=(self.state_space,), name="state")

        # Hidden Units are defined!
        hidden_size = self.state_space * 2
        x = Dense(hidden_size, activation="relu")(input)
        x = Dense(128, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        output = Dense(self.action_space, activation="relu", name="actions")(x)

        model = Model(inputs=input, outputs=output, name=self.name)

        self.opt = Adam(learning_rate=self.lr)
        # opt=RMSprop(lr=0.02, rho=0.9, epsilon=None, decay=0),
        model.compile(optimizer=self.opt, loss="mse")  # metrics=["mse"]

        return model

    def load_checkpoint(self, path):
        self.model = load_model(path)
        self.model.compile(optimizer=self.opt, loss="mse")
        self.model.summary()

    def save_checkpoint(self, path):
        self.model.save(path)

    def reset_for_next_train(self):
        self.total_profit = 0
        self.memory = []
        self.loss = 0

    def decay_epsilon(self, epoch):
        """ 
        #### Epsilon
        reduziere das Epsilon pro Epoche
        evtl. eine andere Formel
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action(self, state, env_initial_action: Action):
        """
        #### Get Action
        Greedy-Epsilon Strategie. Wird Epsilon immer kleiner, so
        werden immer häufiger Aktionen aus den Erfahrungen gewählt.
        Balance zwischen Erkundung (exploration) und Bedienung (exploitation)
        return 0: hold, 1: buy, 2: sell
        """
        action = Action.HOLD

        # Falls die Daten Lücken ausweisen, da nicht alles geloggt wurde
        if state.size < self.state_space:
            return action

        # Zufällige Aktion
        if np.random.rand() <= self.epsilon:
            a = np.random.choice(self.action_space)
            action = Action(a)

        elif env_initial_action == Action.HOLD:
            q = self.model.predict(state)
            action = Action(np.argmax(q[0]))

        elif env_initial_action != Action.HOLD:
            q = self.model.predict(state)
            action = Action(np.argmax(q[0]))

        return action

    # TODO: Löschen könnte zu teuer sein, evtl. wären zwei Zähler besser
    def add_memory(self, state):
        """state = [state_t, action_t, reward_t, state_t+1, game_over?]"""
        self.memory.append(state)
        # Lösche alte Erfahrungen
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def train_exp_replay(self, batch_size=10):
        """ Trainiere anhand der gemachten Erfahrungen
            Wird auch als Experience Replay bezeichnet """

        len_memory = len(self.memory)
        length = min(len_memory, batch_size)

        # define size
        states = np.zeros((length, self.state_space))
        targets = np.zeros((length, self.action_space))

        # Wähle die letzten batch_size Erinnerungen aus
        memory_idx = range(len_memory - length, len_memory)

        for i, idx in enumerate(memory_idx):
            state, action, reward, state_next, game_over = self.memory[idx]

            states[i] = state
            # TODO: Was ist das hier?
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deepNN
            # TODO: Setze zu null, die anderen?
            targets[i] = self.model.predict(state)[0]

            # if game_over is True
            if game_over:
                targets[i, action.value] = reward
            else:
                # reward_t + gamma * max_a' Q(s', a')
                Q_sa_max = np.max(self.model.predict(state_next)[0])
                targets[i, action.value] = reward + self.gamma * Q_sa_max

        return self.model.train_on_batch(states, targets)
