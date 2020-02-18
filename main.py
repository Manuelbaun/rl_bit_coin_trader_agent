import os

from pathlib import Path
from tqdm import tqdm_notebook, tnrange, tqdm, trange
from time import perf_counter
from agent import Agent, Action
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


def setup_agent(
    df,
    name="trader",
    init_start_index=10 * 24 * 60,
    window_size=10,
    max_game_length=1000,
    epsilon=1.0,
    load_agent=True,
):
    # Erstelle Gym und ermittle State size
    env = TradingGym(
        df, window_size=window_size, max_game_length=max_game_length, initial_index=init_start_idx
    )

    env.reset()
    state_space = env.get_observation_size()

    # Erstelle den Agenten
    agent = Agent(state_space, name=name, epsilon=epsilon)

    tf.keras.utils.plot_model(agent.model, trade_logs_dir / (agent.name + ".png"), show_shapes=True)
    agent.model.summary()

    version_num = 0
    if load_agent:
        # lade das letzte Modell
        latest_model_path, version_num = get_lastest_model_path(models_dir / agent.name)
        if latest_model_path:
            agent.load_checkpoint(latest_model_path)

    return agent, version_num


# Lade die Bitcoin Data
# df = get_bit_coin_data(data_dir / "gemini_BTCUSD_2015_1min.csv")
# df = get_bit_coin_data(data_dir / "gemini_BTCUSD_2016_1min.csv")
# df = get_bit_coin_data(data_dir / "gemini_BTCUSD_2017_1min.csv")
# df = get_bit_coin_data(data_dir / "gemini_BTCUSD_2018_1min.csv")
df = get_bit_coin_data(data_dir / "gemini_BTCUSD_2019_1min.csv")

# Training stuff
window_size = 10  # 10 für trader

one_day = 60 * 24  # In Minuten
init_start_idx = window_size * one_day

batch_size = 32
epochs = 10000

max_game_length = 3000
traderName = "trader_10"


## Setup den Agenten, mit den richtigen Dimensionen etc.
agent, version_num = setup_agent(
    df,
    window_size=window_size,
    max_game_length=max_game_length,
    init_start_index=init_start_idx,
    name=traderName,
    epsilon=1.0,
)

# starte Training
train(
    df,
    epochs,
    init_start_idx,
    version_num,
    window_size,
    max_game_length,
    agent,
    batch_size,
    trade_logs_dir,
    models_dir,
)

