import os
import random
import numpy as np
from collections import deque
from keras.models import Model, load_model


class DQN:
    def __init__(self, model=None,
                 exploration_rate=1.0,
                 discount_rate=0.95,
                 exploration_rate_min=0.01,
                 exploration_rate_decay=0.995,
                 memory_size=2000):
        self._memory = deque(maxlen=memory_size)
        self._gamma = discount_rate
        self._epsilon = exploration_rate
        self._epsilon_min = exploration_rate_min
        self._epsilon_decay = exploration_rate_decay
        self._model = model if isinstance(model, Model) else None

    @property
    def memory_size(self):
        return len(self._memory)

    @property
    def action_size(self):
        return None if not self._model else self._model.output_shape[1]

    @property
    def state_size(self):
        return None if not self._model else self._model.input_shape[1]

    def remember(self, state, action, reward, next_state, done):
        self._memory.append((state, action, reward, next_state, done))

    def pred_action(self, state):
        if self._model:
            action = self._model.predict(state)
            return np.argmax(action[0])  # returns action with highest prob

    def act(self, state):
        if self._model:
            if np.random.rand() <= self._epsilon:
                return random.randrange(self.action_size)
            return self.pred_action(state)

    def train(self, batch_size):
        batch = random.sample(self._memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self._gamma * np.amax(self._model.predict(next_state)[0])
            target_f = self._model.predict(state)
            target_f[0][action] = target
            self._model.fit(state, target_f, epochs=1, verbose=0)
        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def load(self, filepath, weights_only=False):
        if not weights_only:
            self._model = load_model(filepath)
            print('Model loaded.')
        else:
            if self._model:
                self._model.load_weights(filepath)
                print('Model weights loaded.')

    def save(self, filepath, weights_only=False):
        path = os.path.dirname(filepath)
        if not os.path.exists(path):
            os.makedirs(path)
        if weights_only:
            self._model.save_weights(filepath)
            print('Model weights saved.')
        else:
            self._model.save(filepath)
            print('Model saved.')
