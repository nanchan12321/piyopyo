import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam

# --- ダミーの不正取引データを作成 ---
np.random.seed(42)
num_samples = 500
feature_dim = 5  # 実際の取引データの特徴量数

fraud_data = pd.DataFrame({
    'Feature1': np.random.randn(num_samples),
    'Feature2': np.random.randn(num_samples) * 2,
    'Feature3': np.random.uniform(0, 1, num_samples),
    'Feature4': np.random.uniform(10, 500, num_samples),
    'Feature5': np.random.randint(0, 2, num_samples),
})

# --- GANの構築 ---
noise_dim = 10

# Generator
generator = Sequential([
    Dense(16, input_dim=noise_dim),
    LeakyReLU(alpha=0.2),
    BatchNormalization(),
    Dense(32),
    LeakyReLU(alpha=0.2),
    BatchNormalization(),
    Dense(feature_dim, activation='tanh')
])

# Discriminator
discriminator = Sequential([
    Dense(32, input_dim=feature_dim),
    LeakyReLU(alpha=0.2),
    Dense(16),
    LeakyReLU(alpha=0.2),
    Dense(1, activation='sigmoid')
])

# コンパイル
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
discriminator.trainable = False

# GANの統合
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# --- GANの学習 ---
epochs = 10000
batch_size = 64

for epoch in range(epochs):
    # 本物の不正データ
    idx = np.random.randint(0, fraud_data.shape[0], batch_size)
    real_data = fraud_data.iloc[idx].values

    # 偽のデータ（Generatorが生成）
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    fake_data = generator.predict(noise)

    # ラベル設定
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    # Discriminatorの学習
    d_loss_real = discriminator.train_on_batch(real_data, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Generatorの学習
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    # 1000エポックごとにログ出力
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: D Loss = {d_loss[0]:.4f}, G Loss = {g_loss:.4f}")

# --- 生成データの取得 ---
num_generated = 1000
noise = np.random.normal(0, 1, (num_generated, noise_dim))
generated_fraud_data = generator.predict(noise)

# データフレーム化
generated_df = pd.DataFrame(generated_fraud_data, columns=fraud_data.columns)

# 生成データの表示
print("生成された不正取引データのサンプル")
print(generated_df.head())
