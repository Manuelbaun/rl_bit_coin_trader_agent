import os
import sys
from traiding_gym import TradingGym
from tqdm import tqdm_notebook, tnrange, tqdm, trange
from common import *
from agent import DQNAgent, Action
import random

from multiprocessing import Process

#  enable only CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


def trade(env: TradingGym, agent: DQNAgent, trader_path):

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

            ##################### #############
            # update every iteration !!
            pbar.update(env.trade_length())
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


####################################################################
# get args

if len(sys.argv) != 2:
    print("Usage: python main.py [trader_name]")
    exit()


TRADER_NAME = sys.argv[1]
# Data Bitcoin data
df17 = get_bit_coin_data_df("gemini_BTCUSD_2017_1min.csv")
df18 = get_bit_coin_data_df("gemini_BTCUSD_2018_1min.csv")
df19 = get_bit_coin_data_df("gemini_BTCUSD_2019_1min.csv")

df = pd.concat([df17, df18, df19])


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


def Test_Func(name):
    print(name)


latest_model_path = get_lastest_model_path(models_dir / TRADER_NAME)
latest_model_path = models_dir / "ai_trader_100000_random" / "ep_80000"


if latest_model_path:
    # nutze nur das trainierte Netzwerk
    agent = DQNAgent(env.OBSERVATION_SPACE, name=TRADER_NAME, epsilon=0.0)
    agent.load_checkpoint(latest_model_path)

    trader_path = trade_logs_dir / TRADER_NAME / "best_.csv"
    trade(env, agent, trader_path)
else:
    print("Model nicht gefunden")
