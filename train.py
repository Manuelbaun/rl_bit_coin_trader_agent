import os
from traiding_gym import TradingGym
from tqdm import tqdm_notebook, tnrange, tqdm, trange
from common import log_state_file, log_state, log_console, get_lastest_model_path, get_bit_coin_data
from agent import DQNAgent, Action
import random


import glob, os
import sys
from pathlib import Path
from tqdm import tqdm_notebook, tnrange, tqdm, trange
from time import perf_counter
import pandas as pd
from agent import DQNAgent, Action
from traiding_gym import TradingGym
from common import get_lastest_model_path, get_all_bit_coin_data
from train import train
from trade import trade


# Setze Paths
project_root_dir = Path(Path(__file__).resolve()).parent
models_dir = project_root_dir / "models"
trade_logs_dir = project_root_dir / "trade_logs"
data_dir = project_root_dir / "data"

#  enable only CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

####################################################################

if len(sys.argv) != 4:
    print("Usage: python main.py [trader_name] [trading_strategy: sequential/random] [epochs]")
    exit()

TRADER_NAME, trading_strategy, EPOCHS = sys.argv[1], sys.argv[2], int(sys.argv[3])


df = get_all_bit_coin_data()


# Training stuff
WINDOW_SIZE = 10  # 10 für trader
ONE_DAY = 60 * 24  # In Minuten
START_IDX = WINDOW_SIZE * ONE_DAY
MAX_GAME_LENGTH = 1000


## Setup den Agenten, mit den richtigen Dimensionen etc.

# Erstelle Gym und ermittle State size
env = TradingGym(
    df, window_size=WINDOW_SIZE, max_game_length=MAX_GAME_LENGTH, initial_index=START_IDX
)

# bevor alles andere, reset!
env.reset()

# Erstelle den Agenten>
load_agent = True
agent = DQNAgent(env.OBSERVATION_SPACE, name=TRADER_NAME, epsilon=1.0)
tf.keras.utils.plot_model(agent.model, trade_logs_dir / (agent.name + ".png"), show_shapes=True)
agent.model.summary()

#  Setup paths
model_path = models_dir / agent.name
trader_path = trade_logs_dir / (agent.name + ".csv")

try:
    access_rights = 0o755
    os.mkdir(model_path, access_rights)
except OSError:
    print("Creation of the directory %s failed" % model_path)
else:
    print("Successfully created the directory %s" % model_path)


if load_agent:
    # lade das letzte Modell
    latest_model_path = get_lastest_model_path(model_path)
    if latest_model_path:
        agent.load_checkpoint(latest_model_path)


# starte Training
# train(env, agent, epochs, model_path, trader_path, trading_strategy)


def train(env: TradingGym, agent: DQNAgent, epochs, model_path, trader_path, training="sequential"):

    profits_absolute = []
    profits_norm = []

    # pro Plot Ausgabe
    profit_bucket = []
    profit_bucket_norm = []
    length_bucket = []
    reward_bucket = []

    profit_sum = 0
    counter_win = 0
    counter_loss = 0
    counter_pass = 0
    last_duration_in_minutes = 0

    # minimum des ZeitIndex
    min_index = env.trading_time_index

    # Der Nächste Index, von wo das Training gestartet werden soll
    next_trading_index = env.trading_time_index

    # Die Länge des dataframes,
    trading_max_index = len(env.df)

    for epoch in tqdm(range(1, epochs + 1), ascii=True, unit="epoch"):
        agent.tensorboard.step = epoch

        # Setze nächsten Startpunkt  => Könnte auch Zufällig sein ?
        if training == "sequential":
            next_trading_index += last_duration_in_minutes
        elif training == "random":
            next_trading_index = random.randint(min_index, trading_max_index - env.max_game_length)
        else:
            assert f"Training strategy {training} unknown.  Use 'sequential' or 'random'"
            print(f"Training strategy {training} unknown. Use 'sequential' or 'random'")
            break

        env.set_trading_time_index(next_trading_index)

        if next_trading_index >= trading_max_index - 1:
            print("End of Trainigsdata reached")
            log_console(
                agent,
                env,
                epoch,
                counter_loss,
                counter_win,
                counter_pass,
                profit_sum,
                sum(reward_bucket),
                sum(profit_bucket),
                sum(length_bucket),
            )
            save_model(agent, model_path, sum(reward_bucket), epoch, True)
            break

        epoch_reward = 0
        step = 1
        current_state = env.reset()

        game_over = False

        while not game_over:
            # Zeige AGent den State und erhalte Action
            # env.initial_action gibt an, welche Aktion
            # zum Handelsstart geführt haben, kann dann Hold oder Gegenaktion
            action = agent.get_action(current_state, env.initial_action)

            # Erzwinge ein Runde zu beenden, da max_game_length erreicht wurde
            if env.trade_max_iteration_reached() or env.curr_index >= trading_max_index - 1:
                action = env.game_over_action

            # Zeige Environment State und erhalte die Response
            # Done = terminal_state
            state_next, reward, game_over = env.step(Action(action))

            if game_over:
                if reward > 0:
                    counter_win += 1
                elif reward < 0:
                    counter_loss += 1
                else:
                    counter_pass += 1
                # accum reward
                reward_bucket.append(reward)
                epoch_reward = reward

            agent.add_memory((current_state, action, reward, state_next, game_over))
            agent.train(game_over, step)

            current_state = state_next
            step += 1

        ##################################
        #  Decay Policy anwenden
        agent.decay_epsilon()

        # Now Game is over
        profits_absolute.append(env.profit_loss_absolute)
        profit_bucket.append(env.profit_loss_absolute)
        profits_norm.append(env.profit_loss_norm)
        profit_bucket_norm.append(env.profit_loss_norm)
        # ermittle die Momentane Summe
        profit_sum = sum(profits_absolute)
        profit_sum_norm = sum(profits_norm)

        # setze Profit_sum_norm in der env.
        # env.profit_loss_sum_norm = profit_sum_norm
        last_duration_in_minutes = env.trade_length()
        length_bucket.append(last_duration_in_minutes)
        # Log to File
        # log_state_file(epoch, agent.epsilon, env, profit_sum, profit_sum_norm, trader_path)

        # saves model only if current reward is bigger then de last one
        save_model(agent, model_path, epoch_reward, epoch)

        # Speicher
        # TODO: tensorboard finish
        if epoch % 20 == 0:

            log_console(
                agent.epsilon,
                env,
                epoch,
                counter_loss,
                counter_win,
                counter_pass,
                profit_sum,
                sum(reward_bucket),
                sum(profit_bucket),
                sum(length_bucket),
            )
            profit_bucket = []
            profit_bucket_norm = []
            length_bucket = []
            reward_bucket = []
            counter_win = 0
            counter_loss = 0
            counter_pass = 0

        if epoch % 500 == 0:
            save_model(agent, model_path, epoch_reward, epoch, True)


last_max_reward = 0


def save_model(agent, path, current_reward, epoch, should_force_to_save=False):
    global last_max_reward

    if last_max_reward < current_reward:
        last_max_reward = current_reward
        # Speichere das momentane Model
        path_to_save = path / "rev_{:.3f}".format(current_reward)
        agent.save_checkpoint(path_to_save)

    elif should_force_to_save:
        path_to_save = path / "ep_{}".format(epoch)
        agent.save_checkpoint(path_to_save)


# start trading
trade(env, agent, EPOCHS, model_path, trader_path)
