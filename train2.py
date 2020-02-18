import os
import sys
from pathlib import Path
from tqdm import tqdm_notebook, tnrange, tqdm, trange
from time import perf_counter
from custom_gym import Custom_Gym, Action
from agent import Agent
from functions import *
from prepare_data import getBitCoinData
import traceback

# Setze Paths
project_root_dir = Path(Path(__file__).resolve()).parent
models_dir = project_root_dir / "models"
data_dir = project_root_dir / "data"

#  enable only CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

####################################################################


# Definiere Learning batch_size
model_name = "trader"
model_folder = models_dir / model_name


try:
    access_rights = 0o755
    os.mkdir(model_folder, access_rights)
except OSError:
    print("Creation of the directory %s failed" % model_folder)
else:
    print("Successfully created the directory %s" % model_folder)


epsilon = 1.0
window_size = 10
initial_Start_time = 10 * 24 * 60  # in Minuten


# Lade die Bitcoin Data
df = getBitCoinData("gemini_BTCUSD_2018_1min.csv")

# Erstelle Gym und ermittle State size
env = Custom_Gym(
    df, window_size=window_size, max_game_length=1000, initial_index=initial_Start_time
)
env.reset()
state_t = env.observe()
state_space = state_t.shape[1]
# Erstelle den Agenten
agent = Agent(state_space, name=model_name, epsilon=float(epsilon), train_random=False)

# lade das letzte Modell
latest_model_path, version_num = get_lastest_model_path(model_folder)
if latest_model_path:
    agent.load_checkpoint(latest_model_path)


# Training stuff
counter_win = 0
counter_loss = 0
batch_size = 64
epochs = 10000
max_game_length = 1000


profits_absolute = []
profits_norm = []
for i in range(epochs):
    epoch = i + version_num

    agent.reset_for_next_train()
    agent.decay_epsilon(epoch=epoch)

    initial_Start_time += env.step_counter
    # Resets env
    env = Custom_Gym(
        df, window_size=window_size, max_game_length=1000, initial_index=initial_Start_time
    )

    env.reset()
    loss = 0.0
    game_over = False

    # Initialer State
    state_t = env.observe()

    pbar = tqdm(total=max_game_length, desc="Epoch {:5}".format(epoch))

    for counter in range(max_game_length):
        # Zeige agent den State und erhalte Action
        action = agent.get_action(state_t, env.initial_action)

        # Erzwinge ein Runde zu beenden, da max_game_length erreicht wurde
        if counter >= max_game_length - 1 or env.max_game_length_reached():
            action = env.game_over_action

        # Zeige Environment State und erhalte die Response
        state_next, reward, game_over = env.step(action)

        if reward > 0:
            counter_win += 1
        elif reward < 0:
            counter_loss += 1

        agent.add_memory((state_t, action, reward, state_next, game_over))

        # if action or len(agent.memory) < 20 or np.random.rand() < 0.1:
        loss += agent.train_exp_replay(batch_size)
        state_t = state_next

        if game_over:
            break

        pbar.update(1)

    ##################################
    # Close Progress bar
    pbar.close()

    # Now Game is over
    profits_absolute.append(env.profit_loss_absolute)
    profits_norm.append(env.profit_loss_norm)
    env.profit_loss_sum_norm = sum(profits_norm)

    # Logger
    log_state_file(
        epoch,
        loss,
        agent.epsilon,
        env,
        sum(profits_absolute),
        sum(profits_norm),
        model_folder / "_logs.csv",
    )

    if epoch % 10 == 0:
        path = model_folder / "ep{}".format(epoch)
        log_state(epoch, loss, agent.epsilon, env, sum(profits_absolute), sum(profits_norm), True)
        print("++ --- Save Model --- ++")
        agent.save_checkpoint(path)

