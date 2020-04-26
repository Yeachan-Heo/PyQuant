import PyQuant.rl.envs.PercentageTradingEnv as base
import numpy as np

class BuySellEnv(base.PercentageTradingEnv):
    def __init__(self, *args, **kwargs):
        super(BuySellEnv, self).__init__(*args, **kwargs)

    def _preprocess_action(self, action):
        action = action - 1
        action = super(BuySellEnv, self)._preprocess_action(action)
        return action


if __name__ == '__main__':
    env = BuySellEnv()
    for epi in range(100):
        d = False
        s = env.reset()
        while not d:
            env.render()
            a = epi % 3
            print(a)
            r, s, d = env.step(a)
