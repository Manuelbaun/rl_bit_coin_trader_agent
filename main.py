import glob, os
from pathlib import Path
from tqdm import tqdm_notebook, tnrange, tqdm, trange
from time import perf_counter
import pandas as pd
from agent import DQNAgent, Action
from traiding_gym import TradingGym
from common import get_lastest_model_path, get_bit_coin_data
from train import train


# Setze Paths
project_root_dir = Path(Path(__file__).resolve()).parent
models_dir = project_root_dir / "models"
trade_logs_dir = project_root_dir / "trade_logs"
data_dir = project_root_dir / "data"

#  enable only CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

####################################################################


# Lade die Bitcoin Data
data_gamini = data_dir / "gemini"

df15 = get_bit_coin_data(data_gamini / "gemini_BTCUSD_2015_1min.csv")
df16 = get_bit_coin_data(data_gamini / "gemini_BTCUSD_2016_1min.csv")
df17 = get_bit_coin_data(data_gamini / "gemini_BTCUSD_2017_1min.csv")
df18 = get_bit_coin_data(data_gamini / "gemini_BTCUSD_2018_1min.csv")
df19 = get_bit_coin_data(data_gamini / "gemini_BTCUSD_2019_1min.csv")

df = pd.concat([df15, df16, df17, df18, df19])

# Training stuff
window_size = 10  # 10 für trader
one_day = 60 * 24  # In Minuten
start_idx = window_size * one_day

batch_size = 32
epochs = 30000

max_game_length = 1000
traderName = "trader_10_full"


## Setup den Agenten, mit den richtigen Dimensionen etc.

# Erstelle Gym und ermittle State size
env = TradingGym(
    df, window_size=window_size, max_game_length=max_game_length, initial_index=start_idx
)

# bevor alles andere, reset!
env.reset()

# Erstelle den Agenten>
load_agent = True
agent = DQNAgent(env.OBSERVATION_SPACE, name=traderName, epsilon=1.0)
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


version_num = 0
if load_agent:
    # lade das letzte Modell
    latest_model_path, version_num = get_lastest_model_path(model_path)
    if latest_model_path:
        agent.load_checkpoint(latest_model_path)


# starte Training
train(env, agent, epochs, model_path, trader_path)

