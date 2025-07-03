###  1. LSTM RNN Model (Multi-feature OHLCV with Relative Strength Index (RSI)) + Backtesting ###

# %pip install protobuf==3.20.3
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import time

# Data Loading from Yahoo Finance
def load_yahoo_data(ticker="AAPL", start="2024-05-28", end="2025-05-27"):  # Parameters
    df = yf.download(ticker, start=start, end=end)
    return df

df = load_yahoo_data()
# print(df)
# print(df["Close"])
# print(len(df["Close"].values))

# RSI Computation
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df["RSI"] = compute_rsi(df["Close"])

# Drop initial NaNs from RSI
df = df.dropna()

# Feature Scaling
features = ["Close", "Open", "High", "Low", "Volume", "RSI"]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
target_scaler = MinMaxScaler()
target = target_scaler.fit_transform(df[["Close"]])

# Sequence Preparation
X, y = [], []
seq_len = 20
for i in range(seq_len, len(scaled)):
    X.append(scaled[i - seq_len : i])
    y.append(target[i])
X, y = np.array(X), np.array(y)

# Train/Test Split
test_size = 20
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# LSTM Model Definition
model = Sequential()
model.add(
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))
)
model.add(LSTM(64))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

start = time.time()
model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=0)
end = time.time()
print(f"Training time: {(end - start) / 60:.2f} minutes")

# Prediction
pred = model.predict(X_test)
pred_inv = target_scaler.inverse_transform(pred)
y_test_inv = target_scaler.inverse_transform(y_test)

# Plotting
plt.plot(y_test_inv, label="Actual", color="blue")
plt.plot(pred_inv, label="Predicted", color="red")
plt.title("LSTM Price Prediction (OHLCV + RSI)")
plt.legend()
plt.show()

# RMSE Calculation
rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, pred_inv))
print("LSTM Root-Mean-Squared-Error (RMSE):", rmse_lstm)

# Directional Accuracy Calculation
direction_true = np.sign(y_test_inv[1:] - y_test_inv[:-1])
direction_pred = np.sign(pred_inv[1:] - pred_inv[:-1])
accuracy = np.mean(direction_true == direction_pred)
print(f"LSTM Directional Accuracy: {accuracy * 100:.2f}%")



###  2. Transformer Neural Network Model + Backtesting ###

import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt


# === Transformer Model Definition ===
class TransformerModel(layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        seq_length,
        num_transformer_blocks,
        dropout_rate=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.seq_length = seq_length
        self.num_transformer_blocks = num_transformer_blocks

        self.positional_encoding = self.positional_encoding(seq_length)

        self.transformer_blocks = [
            self.transformer_block(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_transformer_blocks)
        ]

        self.dropout_1 = layers.Dropout(dropout_rate)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1)

    def call(self, inputs, training=True):
        seq_length = inputs.shape[1]
        word_emb = inputs

        # Tile the positional encoding to match the batch size
        pos_encoding_tiled = tf.tile(
            self.positional_encoding, [tf.shape(inputs)[0], 1, 1]
        )

        # Add positional encoding to word embeddings
        word_emb += pos_encoding_tiled

        x = self.dropout_1(word_emb, training=training)
        for i in range(self.num_transformer_blocks):
            x = self.transformer_blocks[i](x, training=training)
        x = self.flatten(x)
        output = self.dense(x)
        return output

    def transformer_block(self, embed_dim, num_heads, ff_dim, dropout_rate):
        inputs = layers.Input(shape=(None, embed_dim))
        attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )(inputs, inputs)
        attention = layers.Dropout(dropout_rate)(attention)
        attention = layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

        outputs = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(
            attention
        )
        outputs = layers.Dropout(dropout_rate)(outputs)
        outputs = layers.Conv1D(filters=embed_dim, kernel_size=1)(outputs)
        outputs = layers.Dropout(dropout_rate)(outputs)
        outputs = layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

        return Model(inputs=inputs, outputs=outputs)

    def positional_encoding(self, seq_length):
        pos = tf.cast(tf.range(seq_length)[:, tf.newaxis], dtype=tf.float32)
        i = tf.cast(tf.range(self.embed_dim)[tf.newaxis, :], dtype=tf.float32)
        angle_rads = pos / tf.pow(
            10000, 2 * (i // 2) / tf.cast(self.embed_dim, tf.float32)
        )
        angle_rads = tf.where(
            tf.math.equal(i % 2, 0), tf.sin(angle_rads), tf.cos(angle_rads)
        )
        pos_encoding = angle_rads[tf.newaxis, ...]
        return pos_encoding


# === Data Loading from Yahoo Finance ===
def load_yahoo_data(ticker="AAPL", start="2024-05-28", end="2025-05-27"):  # Parameters
    df = yf.download(ticker, start=start, end=end)
    return df


df_raw = load_yahoo_data()
stock_data = df_raw.copy()


# === Generate Training Data ===
def generate_data(stock_data, seq_length):
    closing_prices = stock_data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(closing_prices)
    X, y = [], []
    for i in range(len(scaled_prices) - seq_length):
        X.append(scaled_prices[i : i + seq_length])
        y.append(scaled_prices[i + seq_length])
    return np.array(X), np.array(y), scaler


# === Hyperparameters ===
embed_dim = 32
num_heads = 2
ff_dim = 32
seq_length = 20
num_transformer_blocks = 2
dropout_rate = 0.1
learning_rate = 0.001
batch_size = 64
epochs = 100

# === Prepare Data ===
X_train, y_train, scaler = generate_data(stock_data, seq_length)

# === Initialize Transformer Model ===
model = TransformerModel(
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    seq_length=seq_length,
    num_transformer_blocks=num_transformer_blocks,
    dropout_rate=dropout_rate,
)

# === Loss and Optimizer ===
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate)


# === Training Step ===
@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# === Training Loop ===
num_batches = len(X_train) // batch_size
for epoch in range(epochs):
    total_loss = 0.0
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_X = X_train[start_idx:end_idx]
        batch_y = y_train[start_idx:end_idx]
        loss = train_step(batch_X, batch_y)
        total_loss += loss
    average_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss.numpy():.6f}")

# === Predict Next 10 Days ===
next_days = 10
predicted_prices = []
last_sequence = X_train[-1]
for _ in range(next_days):
    predicted_price = model(last_sequence.reshape(1, seq_length, 1))
    predicted_prices.append(predicted_price.numpy()[0][0])
    last_sequence = np.concatenate(
        (last_sequence[1:], [[predicted_price.numpy()[0][0]]])
    )

# === Inverse Scaling ===
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.plot(
    stock_data.index[-20:],
    stock_data["Close"].values[-20:],
    label="Actual Prices",
    color="blue",
)
plt.plot(
    stock_data.index[-10:], predicted_prices, label="Predicted Prices", color="red"
)
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Actual vs. Predicted Prices for the Next 10 Days")
plt.legend()
plt.show()

# === Get actual closing prices for the next 10 days ===
y_true = stock_data["Close"].values[-10:]

# === RMSE Calculation ===
rmse_transformer = np.sqrt(mean_squared_error(y_true, predicted_prices))
print("Transformer Root-Mean-Squared-Error (RMSE):", rmse_transformer)