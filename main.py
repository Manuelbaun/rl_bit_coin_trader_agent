import os

from pathlib import Path
from tqdm import tqdm_notebook, tnrange, tqdm, trange
from time import perf_counter
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
# df = get_bit_coin_data(data_dir / "gemini_BTCUSD_2015_1min.csv")
# df = get_bit_coin_data(data_dir / "gemini_BTCUSD_2016_1min.csv")
# df = get_bit_coin_data(data_dir / "gemini_BTCUSD_2017_1min.csv")
# df = get_bit_coin_data(data_dir / "gemini_BTCUSD_2018_1min.csv")
df = get_bit_coin_data(data_dir / "gemini_BTCUSD_2019_1min.csv")

# Training stuff
window_size = 10  # 10 für trader
one_day = 60 * 24  # In Minuten
start_idx = window_size * one_day

batch_size = 32
epochs = 10000

max_game_length = 1000
traderName = "trader_10"


## Setup den Agenten, mit den richtigen Dimensionen etc.

# Erstelle Gym und ermittle State size
env = TradingGym(
    df, window_size=window_size, max_game_length=max_game_length, initial_index=start_idx
)

# bevor alles andere, reset!
env.reset()

# Erstelle den Agenten
agent = DQNAgent(env.OBSERVATION_SPACE, name=traderName, epsilon=1.0)
tf.keras.utils.plot_model(agent.model, trade_logs_dir / (agent.name + ".png"), show_shapes=True)
agent.model.summary()

version_num = 0
# if load_agent:
#     # lade das letzte Modell
#     latest_model_path, version_num = get_lastest_model_path(models_dir / agent.name)
#     if latest_model_path:
#         agent.load_checkpoint(latest_model_path)

# starte Training
train(env, agent, epochs, trade_logs_dir, models_dir)

