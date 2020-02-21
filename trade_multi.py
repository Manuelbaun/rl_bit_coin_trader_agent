# Hier werden verschiedene Modelle gleichzeitig zum traden aufgerufen


import os
import sys

#  enable only CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


from traiding_gym import TradingGym
from tqdm import tqdm_notebook, tnrange, tqdm, trange
from common import *
from agent import DQNAgent, Action
import random

from multiprocessing import Process


if len(sys.argv) != 3:
    print("Usage: python main.py [trader_name] [YEAR]")
    exit()


def Trade(TRADER_NAME, model_path, trader_path, YEAR):
    # setup
    df = get_bit_coin_data_df(f"gemini_BTCUSD_{YEAR}_1min.csv")

    # PARAMETER
    WINDOW_SIZE = 10  # 10 für trader
    ONE_DAY = 60 * 24  # In Minuten
    START_IDX = WINDOW_SIZE * ONE_DAY
    MAX_GAME_LENGTH = 1000

    # Erstelle Gym und ermittle State size
    env = TradingGym(
        df, window_size=WINDOW_SIZE, max_game_length=MAX_GAME_LENGTH, initial_index=START_IDX
    )

    # bevor alles andere, reset!
    env.reset()

    # create Agent
    agent = DQNAgent(env.OBSERVATION_SPACE, name=TRADER_NAME, epsilon=0.0)
    agent.load_checkpoint(model_path)

    # START trade
    profits_absolute = []
    profits_norm = []

    # Die Länge des dataframes,
    trading_max_index = len(env.df)
    trades_counter = 0

    with tqdm(initial=env.curr_index, total=trading_max_index, ascii=True, unit=" index") as pbar:

        # Handle durch die gesamte Zeit hindurch
        while env.curr_index < trading_max_index:
            # Neuer Start Trading Zeitpunkt ist das Ende vom letzten Trading
            env.set_trading_time_index(env.curr_index)
            epoch_reward = 0
            game_over = False
            trades_counter += 1
            current_state = env.reset()

            # eine Neue Handelsepisode beginnt
            while not game_over:
                if not env.trade_max_iteration_reached():
                    # Zeige AGent den State und erhalte Action
                    # env.initial_action gibt an, welche Aktion
                    # zum Handelsstart geführt haben, kann dann Hold oder Gegenaktion
                    action = agent.get_action(current_state, env.initial_action)
                    state_next, reward, game_over = env.step(Action(action))

                    current_state = state_next
                else:
                    # Erzwinge ein Runde zu beenden, da max_game_length erreicht wurde
                    state_next, reward, game_over = env.step(env.game_over_action)
                    break

                pbar.update(1)

            ##################### #############
            # update every iteration !!
            # pbar.update(env.trade_length())
            # Now Game is over
            profits_absolute.append(env.profit_loss_absolute)
            profits_norm.append(env.profit_loss_norm)

            # Log to File
            log_state_file(
                trades_counter,
                agent.epsilon,
                env,
                sum(profits_absolute),
                sum(profits_norm),
                trader_path,
            )

    pbar.close()


TRADER_NAME = sys.argv[1]
YEAR = sys.argv[2]
models_paths = get_all_models_with_epoch(models_dir / TRADER_NAME, 10000)


def Test_Func(arg1, arg2, arg3):
    print("Hallo", arg1, arg2, arg3)


if __name__ == "__main__":
    process = []

    trading_logs_path = project_root_dir / "trading" / TRADER_NAME / YEAR

    try:
        access_rights = 0o755
        os.mkdir(trading_logs_path, access_rights)
    except OSError:
        print("Creation of the directory %s failed" % trading_logs_path)
    else:
        print("Successfully created the directory %s" % trading_logs_path)

    for model in models_paths:

        filename = trading_logs_path / f"ep_{model[1]}.csv"

        p = Process(target=Trade, args=(TRADER_NAME, model[0], filename, YEAR,))

        p.start()
        process.append(p)

    for p in process:
        p.join()

