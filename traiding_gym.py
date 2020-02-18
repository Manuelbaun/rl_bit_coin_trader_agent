import gym
from gym import spaces
import numpy as np
from datetime import datetime, timedelta
from agent import Action

# https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e


class TradingGym(gym.Env):
    """Custom Environment that follows gym interface\n
    
    Ein Spielverlauf ist die Auswahl einer Aktion, kaufen oder verkaufen
    und dessen Umkehraktion. \n
    1. Wenn verkauft wurde => wird wieder zurückgekauft.\n
    2. Wenn gekauft wurde => wird wieder verkauft.\n

    Dann ist das Spiel vorbei, und der Reward wird ermittelt:\n
    1. Profit :1
    2. Verlust : -1
    3. sonst: 0
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, window_size, max_game_length, initial_index):
        """
        - `df`: ein DataFrame mit DateTime index!
        - `window_size`: Die Zeit-Fenster-Größe,
        - `max_game_length`: die maximale Spiellänge in Minuten, danach wird die GameOver Aktion ausgeführt
        - `initial_index`: Der Index, von wo aus, die Daten aus der CSV geliefert werden
        """
        super(TradingGym, self).__init__()
        # setup the rest
        self.df = df

        # 5 Min resample
        self.df_5m = self.df["Close"].resample("5T", label="right", closed="right").ohlc().dropna()
        # 1 Std. resample
        self.df_1h = self.df["Close"].resample("1H", label="right", closed="right").ohlc().dropna()
        # 1 Tag resample
        self.df_1d = self.df["Close"].resample("1D", label="right", closed="right").ohlc().dropna()

        self.window_size = window_size

        # 3 Aktionen die man ausführen kann
        self.action_space = 3
        # basierend auf window_size=10 => wird später geändert
        self.state_space = 125

        self.initial_index = initial_index
        self.curr_index = self.initial_index
        self.reward = 0
        self.curr_time = 0
        self.curr_price = 0
        self.profit_loss_norm = 0
        self.profit_loss_absolute = 0
        self.profit_loss_sum_norm = 0

        self.max_game_length = max_game_length

    def step(self, action: Action):
        """ 
        0: hold, 1: buy,  2: sell\n
        Execute one time step within the environment"""
        self.curr_index += 1

        self._update_state(action)

        return self.observe(), self.reward, self.game_over

    def observe(self):
        return np.array([self.state])

    def reset(self):
        """Muss am Anfang aufgerufen werden.
        Reset the state of the environment to an initial state"""

        # Die länge einer Zeiteinheit in Sekunden
        self.time_unit_in_secs = (self.df.index[1] - self.df.index[0]).seconds

        # Die Zeit in Minuten normiert zwischen 0-1
        self.norm_time_of_day = 0
        # Tag in der Woche Normiert zwischen 0-1
        self.norm_day_of_week = 0
        # Gewinn oder Verlust einer Spielrunde
        self.profit_loss_norm = 0
        self.profit_loss_absolute = 0
        # Preis, zu dem entweder gekauft, oder verkauft wurde
        self.entry_price = 0

        self.game_over = False

        # Momentaner Index
        self.curr_index = self.initial_index

        # Start index des Handels
        self.start_index = self.curr_index
        #
        self.curr_time = self.df.index[self.curr_index]
        self.start_time = self.df.index[self.start_index]
        # State stuff
        self.state = []

        # Die Aktion, die der Agent am Anfang des Spiels ausgeführt hat.
        # hat der Agent am Anfang gekauft oder verkauft =>
        self.initial_action = Action.HOLD
        self.game_over_action = Action.HOLD
        # Update state => hold/sit Aktion
        self._update_state(Action.HOLD)

        #  TODO: rest fertig machen
        return 0

    def render(self, mode="human", close=False):
        return 0

    def trade_max_iteration_reached(self):
        return self.trade_length() >= self.max_game_length

    def _update_state(self, action: Action):
        """Update den momentanen State des Gyms """

        self.curr_time = self.df.index[self.curr_index]
        self.curr_price = self.df["Close"][self.curr_index]

        # Ermittelt den zu erwartenden Gewinn/Verlust
        pnl_norm, pnl = self._calc_current_Profit_Loss()
        self.profit_loss_norm = pnl_norm
        self.profit_loss_absolute = pnl

        self._build_state()

        # skip update falls die state länge kleiner ist,da Zeiten fehlen
        if self.state.size < self.state_space:
            print(
                "######################## Lücke in den Zeit Daten",
                self.state.size,
                self.curr_index,
                self.curr_time,
            )
            return

        # Normiere Zeiten
        self.norm_time_of_day = (self.curr_time.hour * 60 + self.curr_time.minute) / (24 * 60)
        self.norm_day_of_week = self.curr_time.dayofweek / 6

        # Normiere Epoch

        time_delta = self.curr_time - self.start_time
        self.norm_epoch = time_delta.seconds / self.time_unit_in_secs

        """ Spielregeln definieren und Position updaten"""

        if action == Action.HOLD:  # hold/sit => nichts tun
            pass

        elif action == Action.BUY:
            # Sell Stock
            if self.initial_action == Action.SELL:
                # Wurde die Aktion BUY ausgewählt, muss vorher verkauft
                # worden sein. Nun wird zurückgekauft => Game_over
                self.game_over = True
                self._get_reward()

            # Buy Stock
            elif self.initial_action == Action.HOLD:
                # Wenn vorher noch nichts passiert ist,
                # wird jetzt hier jetzt gekauft.
                self.initial_action = Action.BUY
                self.game_over_action = Action.SELL
                self.entry_price = self.curr_price
                self.start_index = self.curr_index
                self.start_time = self.df.index[self.start_index]
            else:
                pass

        elif action == Action.SELL:
            # Buy Stock
            if self.initial_action == Action.BUY:
                # Wurde die Aktion SELL ausgewählt, muss vorher eingekauft
                # worden sein. Nun  wird verkauft => Game_over
                self.game_over = True
                self._get_reward()

            # Sell Stock
            elif self.initial_action == Action.HOLD:
                # Wenn vorher noch nichts passiert ist,
                # wird jetzt hier jetzt verkauft.
                self.initial_action = Action.SELL
                self.game_over_action = Action.BUY
                self.entry_price = self.curr_price
                self.start_index = self.curr_index
                self.start_time = self.df.index[self.start_index]
            else:
                pass

    def _build_state(self):
        """ hier wird der state zusammen gesetzt"""

        state = np.array([])
        # Füge den momentanen Kurzverlauf der letzten window size Kurse an
        # open, high, low, close
        last5m, last1h, last1d = self._get_last_window_size_data()

        state = np.append(state, np.array(last5m))
        state = np.append(state, np.array(last1h))
        state = np.append(state, np.array(last1d))

        # # füge die Differenz der letzten 5 min an
        # state = np.append(state, self.get_diff_of_last_time_steps(5))
        # # füge die Differenz der letzten 60 min an
        # state = np.append(state, self.get_diff_of_last_time_steps(60))
        # # füge die Differenz der letzten 24 Std an
        # state = np.append(state, self.get_diff_of_last_time_steps(60 * 24))

        state = np.append(state, self.initial_action.value)
        # füge die Summe bisheriger Profit/Loss zum state?
        # als -1, 0, oder 1
        state = np.append(state, np.sign(self.profit_loss_sum_norm))
        # füge den momentanen normierten Gewinn/Verlust an
        state = np.append(state, self.profit_loss_norm)
        #  TODO: Sollte der Gesamtverlust/gewinn ??
        # füge die normierte Zeit des Tages an
        state = np.append(state, self.norm_time_of_day)
        # füge den normierten Tag der Woche an
        state = np.append(state, self.norm_day_of_week)

        # normiere den state!
        # Muss das so sein
        self.state = (np.array(state) - np.mean(state)) / np.std(state)

    def _get_last_window_size_data(self):

        d5m = self.curr_time - timedelta(minutes=self.window_size * 5)
        d1h = self.curr_time - timedelta(hours=self.window_size)
        d1d = self.curr_time - timedelta(days=self.window_size)

        # Die Zeiten variieren, warum auch immer
        last5m = self.df_5m[d5m : self.curr_time][-self.window_size :]
        last1h = self.df_1h[d1h : self.curr_time][-self.window_size :]
        last1d = self.df_1d[d1d : self.curr_time][-self.window_size :]

        t = np.array(last5m)
        t2 = np.array(last1h)
        t3 = np.array(last1d)

        # Füge padding hinzu, mit dem ältesten Wert
        if t.size != self.window_size * 4:
            if t.size > 0:
                frame = t[0]
                for i in range(t.shape[0], 10):
                    t = np.insert(t, 0, frame)
            else:
                frame = self.df.loc[self.curr_time, ["Open", "Close", "High", "Low"]]
                for i in range(0, 10):
                    t = np.insert(t, 0, frame)

        # Füge padding hinzu, mit dem ältesten Wert
        if t2.size != self.window_size * 4:
            if t2.size > 0:
                frame = t2[0]
                for i in range(t2.shape[0], 10):
                    t2 = np.insert(t2, 0, frame)
            else:
                frame = self.df.loc[self.curr_time, ["Open", "Close", "High", "Low"]]
                for i in range(0, 10):
                    t2 = np.insert(t2, 0, frame)

        # Füge padding hinzu, mit dem ältesten Wert
        if t3.size != self.window_size * 4:
            if t3.size > 0:
                frame = t3[0]
                for i in range(t3.shape[0], 10):
                    t3 = np.insert(t3, 0, frame)
            else:
                frame = self.df.loc[self.curr_time, ["Open", "Close", "High", "Low"]]
                for i in range(0, 10):
                    t3 = np.insert(t3, 0, frame)

        return t, t2, t3

    def get_diff_of_last_time_steps(self, t):
        """ Kalkuliert den Close diff der letzten `t` Zeitschritte"""
        old = self.curr_index - t
        row_old = self.df.loc[old, ["Open", "High", "Low", "Close", "Volume"]]
        row_now = self.df.loc[self.curr_index, ["Open", "High", "Low", "Close", "Volume"]]
        row_diff = row_now - row_old
        return row_diff.Open, row_diff.Close, row_diff.High, row_diff.Low, row_diff.Volume

    def trade_length(self):
        return self.curr_index - self.start_index

    def _calc_current_Profit_Loss(self):
        """Kalkuliert den momentan zu erwartenden Gewinn oder Verlust, 
        wenn Umkehraktion ausgeführt würde in Normiert \n
        `Handel-Differenz / Handel-Einstiegspreis`
        """
        pnl_norm = 0
        pnl = 0
        # Bei jetziger Aktion steht der zu erwartende Gewinn/Verlust an:
        if self.initial_action == Action.BUY:  # Bei Verkauf
            pnl = self.entry_price - self.curr_price
            pnl_norm = pnl / self.entry_price
        elif self.initial_action == Action.SELL:  # Bei Zurückkauf
            pnl = -self.entry_price + self.curr_price
            pnl_norm = pnl / self.entry_price
        else:
            # Vorher wurde weder eingekauft noch verkauft => kein Gewinn/Verlust
            pnl = 0

        return pnl_norm, pnl

    def _get_reward(self):
        """Reward wird am Ende einer Spielsession zurückgegeben
            - Gewinn      : 1
            - Nichts      : 0 
            - Verlust    : -1
        """

        self.reward = np.sign(self.profit_loss_norm) if self.game_over else self.reward
        # print(self.reward + self.profit_loss_norm*100)
        # Lasse den profit_Loss mit einfließen
        self.reward += self.profit_loss_norm * 100
        return self.reward

    def _assemble_state(self):
        return 0
