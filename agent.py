from __future__ import absolute_import, division, print_function, unicode_literals
import time

import numpy as np
from keras.models import load_model
from keras.layers import Dense, LSTM, concatenate, Input, Flatten, Concatenate, Activation
from keras import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from collections import deque
import datetime
import random
from modified_tensorboard import ModifiedTensorBoard
from enum import Enum


class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


# Hyperparameter
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
GAMMA = 0.95
MIN_REPLAY_MEMORY_SIZE = 1_000
REPLAY_MEMORY_SIZE = 50_000


class DQNAgent:
    def __init__(
        self,
        state_space=10,
        action_space=3,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        learning_rate=0.001,
        epsilon_decay=0.995,
        name="Trader",
    ):
        """
        `state_space`: Inputs, die das DQN Netzwerk aufnimmt \n
        `action_space`: Die Aktionen, die gewählt werden können, hier nur 3, `buy`, `sell`, `hold/sit`\n
        `gamma`[0-1]: Gamma/Discountfaktor. Verzögern für ein kurz[-> 0]- bzw. langfristigen[-> 1] Erfolg. \n
        `lr`[alpha]: Learning Rate für den Agenten/Neuronale Netz\n
        `epsilon`: [epsilon-greedy]
        """
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate

        # Der memory Replay Memory Buffer
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # Zähler für das Updaten des Target-Models
        self.traget_update_counter = 0

        self.name = name

        # Baue das Model
        self.model = self.build_network()

        # Target model für das Doppel DQN, mit diesem .predict!!!
        self.target_model = self.build_network()
        self.target_model.set_weights(self.model.get_weights())

        log_dir = f"logs/{self.name}-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard = TensorBoard(log_dir=log_dir)

    def build_network(self):

        model = Sequential()
        model.add(Dense(self.state_space * 2, input_shape=(self.state_space,), name="state"))
        model.add(Activation("relu"))
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dense(self.action_space, name="actions"),)
        # opt=RMSprop(lr=0.02, rho=0.9, epsilon=None, decay=0),
        self.opt = Adam(lr=self.lr)
        model.compile(loss="mse", optimizer=self.opt, metrics=["accuracy"])

        return model

    def load_checkpoint(self, path):
        self.model = load_model(path)
        self.model.compile(optimizer=self.opt, loss="mse")
        self.model.summary()

    def save_checkpoint(self, path):
        self.model.save(path)

    def decay_epsilon(self):
        """ 
        #### Epsilon
        reduziere das Epsilon e-Greedy, evtl. eine andere Formel
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

        state = np.array(state).reshape(-1, *state.shape)

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

    def add_memory(self, transition):
        """transition = (state, action, reward, state_next, game_over)"""
        self.replay_memory.append(transition)

    def get_Action(self, state):
        state_ = np.array(state).reshape(-1, *state.shape)  #  TODO normalisieren
        return np.argmax(self.model.predict(state_)[0])

    def train(self, terminal_state, step):
        """ 
        Trainiere anhand der gemachten Erfahrungen. Wird auch als Experience Replay bezeichnet. 
        Wird erst angewandt, wenn mindestens MIN_REPLAY_MEMORY_SIZE erreicht ist
        """

        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Wählt zufällig Erinnerungen aus.
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array(
            [transition[0] for transition in minibatch]
        )  # TODO: normalisieren
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array(
            [transition[3] for transition in minibatch]
        )  # TODO: normalisieren

        # Wende das Dopple DQN an!
        # TODO: check: predict_batch?
        future_qs_list = self.target_model.predict(new_current_states)

        # X und Y
        X_states = []
        Y_target_qs = []

        for index, (state, action, reward, state_t_next, done) in enumerate(minibatch):
            # hier kommt die Formel für das Q-Leaning zum Einsatz
            if not done:
                Q_sa_max = np.max(future_qs_list[index])
                Q_sa_new = reward + self.gamma * Q_sa_max
            else:
                Q_sa_new = reward

            current_qs = current_qs_list[index]
            current_qs[action.value] = Q_sa_new
            # targets[i, action_t.value] = reward_t

            X_states.append(state)
            Y_target_qs.append(current_qs)

        loss = self.model.fit(
            np.array(X_states),  # TODO: normalisieren
            np.array(Y_target_qs),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
            # Fit das Model, wenn terminal state existiert, sonst nicht!
            # callbacks=[self.tensorboard] if terminal_state else None,
        )

        # Überprüfen, ob das Target_model mit den neuen Werten aktualisiert werden soll
        if terminal_state:
            self.traget_update_counter += 1

        if self.traget_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.traget_update_counter = 0

        return loss
