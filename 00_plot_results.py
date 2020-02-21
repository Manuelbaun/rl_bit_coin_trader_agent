import glob, os
import sys
from pathlib import Path
import re
import pandas
from pandas import DataFrame
from common import *
import datetime
import numpy as np
import matplotlib.dates as mdates

# Import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


project_root_dir = Path(Path(__file__).resolve()).parent
models_dir = project_root_dir / "models"
data_dir = project_root_dir / "data"
data_gamini = data_dir / "gemini"
trading_res = project_root_dir / "trading"


def plot_year(TRADER_NAME, YEAR):
    path = trading_res / TRADER_NAME / YEAR

    df = get_bit_coin_data(data_gamini / f"gemini_BTCUSD_{YEAR}_1min.csv")

    df_trades = []
    for ep in range(10000, 100001, 10000):
        df_trade = get_trading_logs(path / f"ep_{ep}.csv")
        df_trades.append(df_trade)

    sns.set()
    sns.set_context("paper")

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex="col")
    # AXIS FORMATTER
    locator = mdates.MonthLocator()
    formatter = mdates.DateFormatter("%b")

    start_date = np.datetime64(df.index.min())
    end_date = np.datetime64(df.index.max())

    # Set X-Axis formetter
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlim([start_date, end_date])

    axes[0].set_xlabel("Datum")
    axes[0].set_ylabel("Kursverlauf in [$]")

    # Limit x Axis
    # axes[1].set_yticks([-5000, 0, 5000, 10000, 15000])
    axes[1].set_ylabel("Kapital in [$]")

    axes[0].grid(linestyle="-", linewidth="0.5")
    axes[1].grid(linestyle="-", linewidth="0.5")

    axes[0].set_axisbelow(True)
    axes[1].set_axisbelow(True)

    df["Close"].plot(ax=axes[0], label="Bitcoin")
    axes[0].set_title(f"Bitcoin Kurs {YEAR}")

    # Shrink current axis by 20%
    box = axes[0].get_position()
    axes[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ############### PLOT dataframes
    for index, df_trade in enumerate(df_trades):
        df_trade["Balance [$]"].plot(ax=axes[1], label=f"{index+1}0k")

    fig.autofmt_xdate()
    fig.legend(loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)

    plt.savefig(format="svg", fname="plots/" + TRADER_NAME + f"_{YEAR}.svg", dpi=300)
    plt.close()


plt.close("all")
TRADER_NAME = "ai_trader_100000_seq_pass_2"

for year in range(1, 4, 1):
    plt.figure(year)
    plot_year(TRADER_NAME, f"{2016+year}")

plt.show()


# path = trading_res / "ai_trader_100000_random"

# df17 = get_bit_coin_data(data_gamini / f"gemini_BTCUSD_2017_1min.csv")
# df18 = get_bit_coin_data(data_gamini / f"gemini_BTCUSD_2018_1min.csv")
# df19 = get_bit_coin_data(data_gamini / f"gemini_BTCUSD_2019_1min.csv")

# df = pd.concat([df17, df18, df19])
# df_trades = get_trading_logs(path / f"best_.csv")


# sns.set()
# sns.set_context("paper")
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex="col")

# df["Close"].plot(ax=ax1, label="Bitcoin")
# df_trades["Balance [$]"].plot(ax=ax2, style="g", label="80k")

# start_date = np.datetime64(df.index.min())
# end_date = np.datetime64(df.index.max())


# # AXIS FORMATTER
# ax1.set_title(f"Bitcoin Kurs 2017-2019")
# ax1.set_xlabel("Datum")
# ax1.set_ylabel("Kursverlauf in [$]")
# ax2.set_ylabel("Kapital in [$]")

# # Set X-Axis formetter
# ax2.set_xlim([start_date, end_date])
# ax1.set_xlim([start_date, end_date])


# ax2.xaxis.set_minor_locator(mdates.MonthLocator())
# ax2.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

# ax2.xaxis.set_major_locator(mdates.YearLocator(1))
# ax2.xaxis.set_major_formatter(mdates.DateFormatter("\n\n%Y"))


# for tick in ax2.get_xticklabels():
#     tick.set_rotation(0)

# for tick in ax2.xaxis.get_minorticklabels():
#     tick.set_rotation(90)
#     tick.set_horizontalalignment("left")

# plt.xticks(ha="center")

# fig.legend(loc=7)
# fig.tight_layout()
# fig.subplots_adjust(right=0.85)
# plt.savefig(format="svg", fname="plots/ai_trader_100000_random_best_2017-2019.svg", dpi=300)

