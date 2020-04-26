import PyQuant.rl.envs.AmountTradingEnv as base
import numpy as np

class PercentageTradingEnv(base.AmountTradingEnv):
    def __init__(self, *args, **kwargs):
        super(PercentageTradingEnv, self).__init__(*args, **kwargs)

    def _buy(self, amount:float):
        super(PercentageTradingEnv, self)._buy(amount)

    def _preprocess_action(self, action):
        if action < 0: action = np.round(np.abs(self.stock * action))
        elif action > 0: action = np.round(np.abs(self.money / self._stock_price * action))
        return action

if __name__ == '__main__':
    env = PercentageTradingEnv()
    for epi in range(100):
        d = False
        s = env.reset()
        while not d:
            env.render()
            r, s, d = env.step(np.random.uniform(-1, 1))
