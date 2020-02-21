import glob, os
import sys
from pathlib import Path
import re
import pandas as pd
from pandas import DataFrame


# Setze Paths
project_root_dir = Path(Path(__file__).resolve()).parent
models_dir = project_root_dir / "models"
trade_logs_dir = project_root_dir / "trading"
data_dir = project_root_dir / "data"
data_gamini = data_dir / "gemini"


def get_bit_coin_data(file_path):
    df = pd.read_csv(file_path, header=0, encoding="ascii", skipinitialspace=True)
    df.info()
    df["Date"] = pd.to_datetime(df["Date"])

    # created an extra index column => why?
    df = df.sort_values(by="Date", ascending=True).reset_index()
    df.set_index("Date", inplace=True)
    df = df.drop(["Unix Timestamp", "Symbol", "index"], axis=1)

    return df


def get_trading_logs(file_path):
    df = pd.read_csv(file_path, header=0, encoding="ascii", skipinitialspace=True)
    df.info()
    df["Start"] = pd.to_datetime(df["Start"])
    df["End"] = pd.to_datetime(df["End"])

    df.set_index("End", inplace=True)
    df = df.drop(["Epsilon", "Epoch"], axis=1)

    return df


def get_all_bit_coin_data():
    # Lade die Bitcoin Data
    df15 = get_bit_coin_data(data_gamini / "gemini_BTCUSD_2015_1min.csv")
    df16 = get_bit_coin_data(data_gamini / "gemini_BTCUSD_2016_1min.csv")
    df17 = get_bit_coin_data(data_gamini / "gemini_BTCUSD_2017_1min.csv")
    df18 = get_bit_coin_data(data_gamini / "gemini_BTCUSD_2018_1min.csv")
    df19 = get_bit_coin_data(data_gamini / "gemini_BTCUSD_2019_1min.csv")

    return pd.concat([df15, df16, df17, df18, df19])


def get_bit_coin_data_df(file):
    data_gamini = data_dir / "gemini"
    return get_bit_coin_data(data_gamini / file)


def get_lastest_model_path(path_to_folder):
    """returns path of model in given folder and name,
    and its version number ep{number}"""
    latest_model = ""

    regex = re.compile("ep(\d+)")
    try:
        files = os.listdir(path_to_folder)
        paths = [os.path.join(path_to_folder, basename) for basename in files]
        latest_model = max(paths, key=os.path.getctime)
    except:
        print("Kein Model gefunden, ein neues wird erstellt")

    with os.scandir(path_to_folder) as it:
        for entry in it:
            z = re.search(regex, entry.name)
            if z and (int(z.group(1)) >= version_num):
                version_num = int(z.group(1))
                file = entry.path

    return latest_model


def get_all_models_with_epoch(path_to_folder, epoch=10000):
    """returns path of model in given folder and name,
    and its version number ep{number}"""
    files = []

    regex = re.compile("ep_(\d+)")
    try:
        with os.scandir(path_to_folder) as it:

            for entry in it:
                z = re.search(regex, entry.name)
                if z:
                    ep = int(z.group(1))
                    choose = ep % epoch
                    if choose == 0:
                        files.append((entry.path, ep))

    except Exception as e:
        print("Kein Model gefunden", e)

    return files


# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


def log_state(epoch, epsilon, env, profits_absolut, profits_norm):

    stats = ""
    stats += "Epoch: {:03d},".format(epoch)
    # stats += "Loss: {:.2f},".format(loss)
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


def log_state_file(epoch, epsilon, env, profits_absolut, profits_norm, filename):
    global file_exits

    if not file_exits:
        file_exits = log_state_file_head(filename)

    stats = ""
    stats += "{:03d},".format(epoch)
    # stats += "{:.2f},".format(loss)
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
    # stats_head += "Loss,"
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


def log_console(
    epsilon,
    env,
    epoch,
    counter_loss,
    counter_win,
    counter_pass,
    profit_total,
    reward_sum,
    profit_sum_since_last,
    length_sum,
):

    st = f"\nEpoch: {(epoch-20):5}-{epoch:5} | "
    st += f"Epsilon: {epsilon:5.4f} | "

    # st = f"Steps: {(step:5} | "
    st += f"Duration: {length_sum:8} | "
    st += f"Wins: {counter_win:4} | "
    st += f"Loss: {counter_loss:4} | "
    st += f"Pass: {counter_pass:4} | "
    st += f"Reward: {reward_sum:8.2f} | "
    st += f"PNL: $ {profit_sum_since_last:8.2f} | "
    st += f"Kapital: $ {profit_total:12.2f} | "
    st += f"Date: {env.curr_time}"
    print(st)
