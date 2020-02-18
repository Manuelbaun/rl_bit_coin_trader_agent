import os
from traiding_gym import TradingGym
from tqdm import tqdm_notebook, tnrange, tqdm, trange
from common import log_state_file, log_state, get_lastest_model_path, get_bit_coin_data
from agent import Agent, Action


def train(
    dataframe,
    epochs,
    initial_index,
    version_num,
    window_size,
    max_game_length,
    agent: Agent,
    batch_size,
    trade_path,
    models_path,
):

    # setup path to store the model
    model_path = models_path / agent.name
    trader_path = trade_path / (agent.name + ".csv")

    try:
        access_rights = 0o755
        os.mkdir(model_path, access_rights)
    except OSError:
        print("Creation of the directory %s failed" % model_path)
    else:
        print("Successfully created the directory %s" % model_path)

    profits_absolute = []
    profits_norm = []
    counter_win = 0
    counter_loss = 0

    last_iteration = 0
    for i in range(epochs):
        # Ändere Epoch für den Checkpoint
        epoch = i + version_num

        agent.reset_for_next_train()
        agent.decay_epsilon(epoch=epoch)

        # Setze nächsten Startpunkt  => Könnte auch Zufällig sein ?
        initial_index += last_iteration
        loss = 0.0
        game_over = False

        # Resets Environment und Initialer State
        env = TradingGym(
            dataframe,
            window_size=window_size,
            max_game_length=max_game_length,
            initial_index=initial_index,
        )
        env.reset()
        # set init state
        state_t = env.get_state()

        pbar = tqdm(total=max_game_length, desc="Epoch {:5}".format(epoch))

        # Das eigentliche Spiel fängt hier an
        while not game_over:
            # Zeige agent den State und erhalte Action
            action = agent.get_action(state_t, env.initial_action)

            # Erzwinge ein Runde zu beenden, da max_game_length erreicht wurde
            if env.trade_max_iteration_reached():
                action = env.game_over_action

            # # Zeige Environment State und erhalte die Response
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
        # ermittle die Momentane Summe
        profit_sum = sum(profits_absolute)
        profit_sum_norm = sum(profits_norm)

        # setze Profit_sum_norm in der env.
        env.profit_loss_sum_norm = profit_sum_norm
        last_iteration = env.trade_length()

        # Log to File
        log_state_file(epoch, loss, agent.epsilon, env, profit_sum, profit_sum_norm, trader_path)

        # Speicher
        if epoch % 10 == 0:
            log_state(epoch, loss, agent.epsilon, env, profit_sum, profit_sum_norm)
            # Speichere das momentane Model
            agent.save_checkpoint(path=model_path / "ep{}".format(epoch))
