# https://towardsdatascience.com/visualizing-artificial-neural-networks-anns-with-just-one-line-of-code-b4233607209e

# Create your first MLP in Keras
from keras.layers import Dense, LSTM, concatenate, Input, Flatten, Concatenate, Activation
from keras import Sequential
from keras.optimizers import Adam
from keras.utils import plot_model


# Viz NN
from ann_visualizer.visualize import ann_viz


input_dim = 124
action_space = 3
model = Sequential()
model.add(Dense(input_dim * 2, input_dim=input_dim, activation="relu", name="state",))
model.add(Dense(128, activation="relu"))
model.add(Dense(action_space, name="actions", activation="linear"),)
model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])


# Plot nn
ann_viz(model, title="AI Trader", filename="plots/network_trader.gv")
# plot_model(model, to_file="network_trader.png")
