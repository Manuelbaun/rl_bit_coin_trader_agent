import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from prepare_data import *

current_dir = Path(Path(__file__).resolve()).parent
data_paths = current_dir / "data"

# Konvertiere String Date to Datetime
row_df = pd.read_csv(data_paths / "gemini_BTCUSD_2019_1min.csv")
# reindex
df = row_df.iloc[::-1].reset_index(drop=True)
df["Date"] = pd.to_datetime(df["Date"])


curr_idx = 100
start = df["Date"][10]
curr_time = df["Date"][400]

_time_of_day = (curr_time.hour * 60 + curr_time.minute) / (24 * 60)
day_of_week = curr_time.dayofweek / 6

t1 = df["Date"][1]
t0 = df["Date"][0]
d2 = t1 - t0
curr_price = df["Close"][1]


# _open, hight, low, close, volume = df.loc[0, ["Open", "High", "Low", "Close", "Volume"]]
Open, Close, High, Low, Volume = df.loc[0, ["Open", "High", "Low", "Close", "Volume"]]

row0 = df.loc[0, ["Open", "High", "Low", "Close", "Volume"]]
row5 = df.loc[5, ["Open", "High", "Low", "Close", "Volume"]]

diff1 = row5 - row0
print(diff1)
# print(row.Open, row.Close, row.High, row.Low, row.Volume)
print(Open, Close, High, Low, Volume)
print(t1, t0, d2)

print(curr_price)


from enum import Enum


class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


action = 1
print(Action(1))


state = np.array([])

state = np.append(state, action)
state = np.append(state, np.sign(123.5))
state = np.append(state, np.sign(-123.5))

print(state)
mean = np.mean(state)
std = np.std(state)
print(mean, std)
state = (np.array(state) - mean) / std
print(state)


def get_diff_of_last_time_steps(t):
    """ Kalculiert den Close diff der letzen `t` Zeitschritte"""
    old = 1000 - t
    row_old = df.loc[old, ["Open", "High", "Low", "Close", "Volume"]]
    row_now = df.loc[1000, ["Open", "High", "Low", "Close", "Volume"]]
    row_diff = row_now - row_old
    return row_diff.Open, row_diff.Close, row_diff.High, row_diff.Low, row_diff.Volume


min5 = get_diff_of_last_time_steps(5)
min10 = get_diff_of_last_time_steps(10)

state = np.append(state, get_diff_of_last_time_steps(5))
state = np.append(state, get_diff_of_last_time_steps(10))
state = np.append(state, get_diff_of_last_time_steps(60))
state = np.append(state, get_diff_of_last_time_steps(60 * 3))
print(state)


df2 = getBitCoinData("gemini_BTCUSD_2019_1min.csv")
print(df2)
curr_time = df2.index[0]

_k = list(map(float, str(curr_time.time()).split(":")[:2]))
price = df2["Close"][0]


def build(df):
    window = 1
    last5m, last1h, last1d = _get_last_N_timebars(window, df2.index[2000])
    bars = [last5m, last1h, last1d]

    state = []
    columns = last5m.keys()
    candles = {j: {k: np.array([]) for k in columns} for j in range(len(bars))}

    for j, bar in enumerate(bars):

        for col in columns:
            arr = np.asarray(bar[col])
            candles[j][col] = arr
            state += list(arr)[-window:]

    print("CAndels", candles)
    print("State", state)
    print("Length", len(state))
    return 0


df_5m = df2["Close"].resample("5T", label="right", closed="right").ohlc().dropna()
df_1h = df2["Close"].resample("1H", label="right", closed="right").ohlc().dropna()
df_1d = df2["Close"].resample("1D", label="right", closed="right").ohlc().dropna()


def _get_last_N_timebars(windows_size, curr_time):
    d5m = curr_time - timedelta(minutes=windows_size * 5)
    d1h = curr_time - timedelta(hours=windows_size)
    d1d = curr_time - timedelta(days=windows_size)

    last5m = df_5m[d5m:curr_time][-windows_size:]
    last1h = df_1h[d1h:curr_time][-windows_size:]
    last1d = df_1d[d1d:curr_time][-windows_size:]

    start = np.array(last5m)[-5:]
    frame1 = start[0]
    tt = start.shape[0]
    em = np.array([])
    for i in range(tt, windows_size):
        start = np.insert(start, 0, frame1)
        # start[:0] = frame1
        print(i, start.size)

    return last5m, last1h, last1d


initial_index = 10 * 24 * 60

# for i in range(1000):
#     curr_time = df2.index[initial_index]
#     a, b, c = _get_last_N_timebars(10, curr_time)
#     initial_index += 1

#     print(curr_time, np.array(a).size, np.array(b).size, np.array(c).size)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

max_x = 5
max_rand = 10

x = np.arange(0, max_x)
ax.set_ylim(0, max_rand)
(line,) = ax.plot(x, np.random.randint(0, max_rand, max_x))


def init():  # give a clean slate to start
    line.set_ydata([np.nan] * len(x))
    return (line,)


def animate(i):  # update the y values (every 1000ms)
    line.set_ydata(np.random.randint(0, max_rand, max_x))
    return (line,)


ani = animation.FuncAnimation(fig, animate, init_func=init, interval=16, blit=True, save_count=10)

plt.show()
