import os
from traiding_gym import TradingGym
from tqdm import tqdm_notebook, tnrange, tqdm, trange
from common import log_state_file, log_state, get_lastest_model_path, get_bit_coin_data
from agent import DQNAgent, Action


def train(env: TradingGym, agent: DQNAgent, epochs, model_path, trader_path):

    profits_absolute = []
    profits_norm = []
    # pro Plot ausgabe
    profit_bucket = []
    profit_bucket_norm = []
    length_bucket = []
    reward_bucket = []

    profit_sum = 0
    counter_win = 0
    counter_loss = 0
    counter_pass = 0
    last_duration_in_minutes = 0

    start_trading_time_index = env.trading_time_index
    trading_max_index = len(env.df)

    for epoch in tqdm(range(1, epochs + 1), ascii=True, unit="epoch"):
        agent.tensorboard.step = epoch

        # Setze nächsten Startpunkt  => Könnte auch Zufällig sein ?
        start_trading_time_index += last_duration_in_minutes
        env.set_trading_time_index(start_trading_time_index)

        if start_trading_time_index >= trading_max_index - 1:
            print("End of Trainigsdata reached")
            log_and_save(
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
                path_save=model_path / f"ep{epoch}",
            )
            break

        epoch_reward = 0
        step = 1
        current_state = env.reset()

        game_over = False

        while not game_over:
            # Zeige agent den State und erhalte Action
            # env.initial_action gibt an, welche Aktion
            # zum Handelsstart geführt haben, => Gegenaktion
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
        log_state_file(epoch, agent.epsilon, env, profit_sum, profit_sum_norm, trader_path)

        # Speicher
        # TODO: tensorboard finish
        if epoch % 20 == 0:
            log_and_save(
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
                path_save=model_path / f"ep{epoch}",
            )
            profit_bucket = []
            profit_bucket_norm = []
            length_bucket = []
            reward_bucket = []
            counter_win = 0
            counter_loss = 0
            counter_pass = 0


def log_and_save(
    agent,
    env,
    epoch,
    counter_loss,
    counter_win,
    counter_pass,
    profit_total,
    reward_sum,
    profit_sum_since_last,
    length_sum,
    path_save,
):
    # agent.tensorboard.update_stats(
    #     profit=profit_sum,
    #     profit_norm=profit_sum_norm,
    #     # reward_min=min_reward,
    #     # reward_max=max_reward,
    #     epsilon=agent.epsilon,
    # )

    st = f"\nEpoch: {(epoch-20):5}-{epoch:5} | "
    # st = f"Steps: {(step:5} | "
    st += f"Duration: {length_sum:8} | "
    st += f"Wins: {counter_win:4} | "
    st += f"Loss: {counter_loss:4} | "
    st += f"Pass: {counter_pass:4} | "
    st += f"Reward: $ {reward_sum:8.2f} | "
    st += f"PNL: $ {profit_sum_since_last:8.2f} | "
    st += f"Kapital: $ {profit_total:12.2f} | "

    st += f"Date: {env.curr_time}"
    print(st)

    # log_state(epoch, agent.epsilon, env, profit_sum, profit_sum_norm)

    # Speichere das momentane Model
    agent.save_checkpoint(path_save)

