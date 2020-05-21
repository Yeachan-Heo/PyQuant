from PyQuant.backtest import *
import pandas as pd

def sell(engine:BackTestCore, pos_thr=20, neg_thr=-3):
    for code in engine.sellable_stocks:
        if ((engine.portfolio[code]["yield(%)"] >= pos_thr) or
            (engine.portfolio[code]["yield(%)"] <= neg_thr)):
            engine.sell(code, engine.portfolio[code]["amount"])

if __name__ == '__main__':
    seed_money = 100000000
    codes = pd.read_csv(path.code_data + "kospi.csv")["code"].to_list()
    engine = BackTestCore(seed_money, print_log=False)
    for p in os.listdir(path.kospi_price_data):
        engine.register_data_path(p[:-4], path.kospi_price_data + p)
    engine.lock()
    engine.move_index(y=11)
    i = 0
    break_flag = False
    while not break_flag:
        buyable = list(engine.buyable_stocks)

        for x in range(random.randint(0, len(buyable))):
            randombuy = random.choice(buyable)
            amount = engine.max_amount(randombuy)
            engine.buy(randombuy, random.randint(0, amount))
            buyable.pop(buyable.index(randombuy))

        if engine.move_index(m=1): break_flag = True

        if engine.sellable_stocks:
            sell(engine)

        if i % 100 == 0:
            pp.pprint(engine.portfolio)
            engine.pv_log()
        i += 1
    engine.update_yield()
    pp.pprint(engine.portfolio)
    json.dump(engine.portfolio, open("portfolio.json", "w"))
    print(engine.pv / seed_money * 100-100, "% 수익!")