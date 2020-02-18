import os
import re
import pandas as pd

from pandas import DataFrame
from traiding_gym import TradingGym


def get_bit_coin_data(file_path):
    df = pd.read_csv(file_path, header=0, encoding="ascii", skipinitialspace=True)
    df.info()
    df["Date"] = pd.to_datetime(df["Date"])

    # created an extra index column => why?
    df = df.sort_values(by="Date", ascending=True).reset_index()
    df.set_index("Date", inplace=True)
    df = df.drop(["Unix Timestamp", "Symbol", "index"], axis=1)

    return df


def get_lastest_model_path(path_to_folder):
    """returns path of model in given folder and name,
    and its version number ep{number}"""
    file = ""
    version_num = 0

    regex = re.compile("ep(\d+)")
    with os.scandir(path_to_folder) as it:
        for entry in it:
            z = re.search(regex, entry.name)
            if z and (int(z.group(1)) >= version_num):
                version_num = int(z.group(1))
                file = entry.path

    return file, version_num


# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


def log_state(epoch, loss, epsilon, env: TradingGym, profits_absolut, profits_norm):

    stats = ""
    stats += "Epoch: {:03d},".format(epoch)
    stats += "Loss: {:.2f},".format(loss)
    stats += "Epsilon: {:.5f},".format(epsilon)
    stats += "IN: {:5},".format(env.initial_action.name)
    stats += "$: {:8.2f},".format(env.entry_price)
    stats += "OUT: {:5},".format(env.game_over_action.name)
    stats += "$: {:8.2f},".format(env.curr_price)
    stats += "Delta-$ {:10.2f},".format(env.profit_loss_absolute)
    stats += "Balance-$: {:10.2f},".format(profits_absolut)
    stats += "PNL-Sum: {:5.2f},".format(profits_norm * 100)
    stats += "PNL-Trade: {:5.2f},".format(env.profit_loss_norm * 100)
    stats += "Reward: {:8.5f},".format(env.reward)
    stats += "Duration[Min]: {:5},".format(env.trade_length())
    stats += "Start: {},".format(env.start_time)
    stats += "End: {},".format(env.curr_time)

    print(stats)


file_exits = False


def log_state_file(epoch, loss, epsilon, env: TradingGym, profits_absolut, profits_norm, filename):
    global file_exits

    if not file_exits:
        file_exits = log_state_file_head(filename)

    stats = ""
    stats += "{:03d},".format(epoch)
    stats += "{:.2f},".format(loss)
    stats += "{:.5f},".format(epsilon)
    stats += "{:5},".format(env.initial_action.name)
    stats += "{:8.2f},".format(env.entry_price)
    stats += "{:5},".format(env.game_over_action.name)
    stats += "{:8.2f},".format(env.curr_price)
    stats += "{:10.2f},".format(env.profit_loss_absolute)
    stats += "{:10.2f},".format(profits_absolut)
    stats += "{:5.2f},".format(profits_norm * 100)
    stats += "{:5.2f},".format(env.profit_loss_norm * 100)
    stats += "{:8.5f},".format(env.reward)
    stats += "{:5},".format(env.trade_length())
    stats += "{},".format(env.start_time)
    stats += "{},".format(env.curr_time)

    fid = open(filename, "a")
    fid.write(stats + "\n")
    fid.close()


def log_state_file_head(filename):
    stats_head = ""
    stats_head += "Epoch,"
    stats_head += "Loss,"
    stats_head += "Epsilon,"
    stats_head += "IN,"
    stats_head += "IN [$],"
    stats_head += "OUT,"
    stats_head += "Out [$],"
    stats_head += "Delta [$],"
    stats_head += "Balance [$],"
    stats_head += "PNL-Sum [%],"
    stats_head += "PNL-Delta [%],"
    stats_head += "Reward,"
    stats_head += "Duration [Min],"
    stats_head += "Start,"
    stats_head += "End,"

    try:
        f = open(filename)
        f.close()
        # Do something with the file
        return True
    except FileNotFoundError:
        print("File not accessible")

    fid = open(filename, "a")
    fid.write(stats_head + "\n")
    fid.close()
    return True
