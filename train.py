import os
from traiding_gym import TradingGym
from tqdm import tqdm_notebook, tnrange, tqdm, trange
from common import log_state_file, log_state, get_lastest_model_path, get_bit_coin_data
from agent import DQNAgent, Action


def train(env: TradingGym, agent: DQNAgent, epochs, trade_path, models_path):

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
    last_duration_in_minutes = 0

    start_trading_time_index = env.trading_time_index

    for epoch in tqdm(range(1, epochs + 1), ascii=True, unit="epoch"):
        agent.tensorboard.step = epoch

        # Setze nächsten Startpunkt  => Könnte auch Zufällig sein ?
        start_trading_time_index += last_duration_in_minutes
        env.set_trading_time_index(start_trading_time_index)

        epoch_reward = 0
        step = 1
        current_state = env.reset()

        done = False

        while not done:
            # Zeige agent den State und erhalte Action
            # env.initial_action gibt an, welche Aktion
            # zum Handelsstart geführt haben, => Gegenaktion
            action = agent.get_action(current_state, env.initial_action)

            # Erzwinge ein Runde zu beenden, da max_game_length erreicht wurde
            if env.trade_max_iteration_reached():
                action = env.game_over_action

            # Zeige Environment State und erhalte die Response
            # Done = terminal_state
            state_next, reward, done = env.step(Action(action))

            epoch_reward += reward

            agent.add_memory((current_state, action, reward, state_next, done))
            agent.train(done, step)

            current_state = state_next
            step += 1

        ##################################
        #  Decay Policy anwenden
        agent.decay_epsilon()

        # Now Game is over
        profits_absolute.append(env.profit_loss_absolute)
        profits_norm.append(env.profit_loss_norm)
        # ermittle die Momentane Summe
        profit_sum = sum(profits_absolute)
        profit_sum_norm = sum(profits_norm)

        # setze Profit_sum_norm in der env.
        # env.profit_loss_sum_norm = profit_sum_norm
        last_iteration = env.trade_length()

        # Log to File
        log_state_file(epoch, agent.epsilon, env, profit_sum, profit_sum_norm, trader_path)

        # Speicher
        # TODO: tensorboard finish
        if epoch % 10 == 0:
            # agent.tensorboard.update_stats(
            #     profit=profit_sum,
            #     profit_norm=profit_sum_norm,
            #     # reward_min=min_reward,
            #     # reward_max=max_reward,
            #     epsilon=agent.epsilon,
            # )
            log_state(epoch, agent.epsilon, env, profit_sum, profit_sum_norm)
            # Speichere das momentane Model
            agent.save_checkpoint(path=model_path / f"ep{epoch}")
