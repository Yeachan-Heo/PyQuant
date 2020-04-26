from PyQuant.rl import normalized, get_random_data
from typing import *
import numpy as np
import gym


class AmountTradingEnv(gym.Env):
    def __init__(self,
                 seed_money = 100000000,
                 observation_size:int=100,
                 use_random_data:bool=True,
                 data:Optional[np.ndarray]=None):
        self.seed_money = seed_money
        self.use_random_data = use_random_data
        self.observation_space = gym.spaces.Space(shape=(1, observation_size))
        self.action_space = gym.spaces.Box(shape=(1, ), low=0, high=1)
        self.data = data
        pass

    def reset(self):
        self.idx = 0
        self.stock = 0
        self.money = self.seed_money
        self.prev_value = self.money
        self.data = get_random_data()
        self.price_data = self.data["close"].to_numpy()
        return self._get_observation()

    def step(self, action:float):
        if action > 0: self._buy(action)
        elif action < 0: self._sell(action)
        self.idx += 1
        reward = self._get_reward()
        observation = self._get_observation()
        done = self._get_terminal()
        return observation, reward, done

    def _get_observation(self):
        observation = np.expand_dims(
            self.price_data[self.idx-self.observation_space.shape[-1]:self.idx], 0)
        normalized_observation = normalized(observation, 1)
        return normalized_observation

    def _get_reward(self):
        return self._current_value / self.prev_value

    def _get_terminal(self):
        return True if self.idx+1 == self.price_data.shape[0] else False

    def _buy(self, amount:float):
        amount = self._preprocess_action(amount)
        amount = np.minimum(amount, self._buy_limit)
        self.money -= amount * self._stock_price
        self.stock += amount

    def _sell(self, amount:float):
        amount = self._preprocess_action(amount)
        amount = np.minimum(amount, self._sell_limit)
        self.money += amount * self._stock_price
        self.stock -= amount

    def _preprocess_action(self, action):
        action = np.round(np.abs(action)).astype(int)
        return action

    @property
    def _buy_limit(self):
        return self.money // self._stock_price

    @property
    def _sell_limit(self):
        return self.stock

    @property
    def _stock_price(self):
        return self.price_data[self.idx]

    @property
    def _current_value(self):
        return self._stock_price*self.stock + self.money

    def render(self, mode='human'):
        print("\n"*2)
        print("portfolio value: ", self._current_value)
        print("stock price: ", self._stock_price)
        print("amount_stock: ", self._stock_price * self.stock/self._current_value*100, "%")
        print("amount_money: ", self.money/self._current_value*100, "%")

if __name__ == '__main__':
    env = AmountTradingEnv()
    for epi in range(100):
        print("\n"*100)
        d = False
        s = env.reset()
        while not d:
            env.render()
            r, ns, d = env.step(np.random.randint(-1000, 1000))









