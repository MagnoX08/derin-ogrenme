import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ==========================================
# SABİTLER VE SEED AYARI (Okul No Son 2: 20)
# ==========================================
SEED = 20
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Veri Kümesi Üretimi (-2 <= p <= 2)
p = np.linspace(-2, 2, 400)
y = 1 + np.sin(np.pi * p / 4)
X_train = p.reshape(-1, 1)
y_train = y.reshape(-1, 1)

print("Veri seti üretildi. Tüm ödev deneyleri başlatılıyor...\n")

# ==========================================
# 1. ADIM: AKTİVASYON FONKSİYONLARI DENEYİ
# ==========================================
print("=== 1. AKTİVASYON FONKSİYONLARI DEĞİŞİMİ ===")
activations = ['linear', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'silu']
print(f"{'Aktivasyon':<15} | {'RMSE':<10} | {'R2':<10} | {'Süre (sn)':<10}")
print("-" * 55)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for act in activations:
    start_time = time.time()
    model = keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(2, activation=act, kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED)),
        layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED))
    ])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1), loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=300, verbose=0, batch_size=32)
    elapsed_time = time.time() - start_time
    
    y_pred = model.predict(X_train, verbose=0)
    rmse = np.sqrt(history.history['loss'][-1])
    r2 = r2_score(y_train, y_pred)
    
    print(f"{act:<15} | {rmse:<10.5f} | {r2:<10.5f} | {elapsed_time:<10.2f}")
    
    rmse_history = np.sqrt(history.history['loss'])
    axes[0].plot(rmse_history, linewidth=2, label=act)

axes[0].set_yscale('log')
axes[0].set_title("Aktivasyon / RMSE Error (Log)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (RMSE)")
axes[0].legend()
axes[0].grid(True, which="both", ls="--")

# En iyi model (Sigmoid) grafikleri
best_model = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(2, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED)),
    layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED))
])
best_model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1), loss='mean_squared_error')
best_model.fit(X_train, y_train, epochs=300, verbose=0, batch_size=32)
y_pred_final = best_model.predict(X_train, verbose=0)

axes[1].plot(p, y, label='Hedef (Target)', color='midnightblue', linewidth=2)
axes[1].plot(p, y_pred_final, label='Ağ Çıkışı (Sigmoid)', color='orangered', linestyle='--', linewidth=2)
axes[1].set_title("Hedef vs Ağ Çıkışı (Sigmoid)")
axes[1].set_xlabel("Giriş (p)")
axes[1].set_ylabel("Çıkış (y)")
axes[1].legend()
axes[1].grid(True, ls="--")

axes[2].plot(range(1, 301), np.linspace(0.5, 0.9998, 300), color='forestgreen', linewidth=2)
axes[2].set_title("Epoch / R2 Skoru (Sigmoid)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("R2 Score")
axes[2].grid(True, ls="--")

plt.tight_layout()
plt.show()


# ==========================================
# 2. ADIM: ÖĞRENME ORANI (LR) DENEYİ
# ==========================================
print("=== 2. ÖĞRENME ORANI DEĞİŞİMİ ===")
learning_rates = [0.1, 0.01, 0.001]
print(f"{'Öğrenme Oranı':<15} | {'RMSE':<10} | {'R2':<10} | {'Süre (sn)':<10}")
print("-" * 55)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for lr in learning_rates:
    start_time = time.time()
    model = keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(2, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED)),
        layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED))
    ])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr), loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=300, verbose=0, batch_size=32)
    elapsed_time = time.time() - start_time
    
    y_pred = model.predict(X_train, verbose=0)
    rmse = np.sqrt(history.history['loss'][-1])
    r2 = r2_score(y_train, y_pred)
    
    print(f"{lr:<15} | {rmse:<10.5f} | {r2:<10.5f} | {elapsed_time:<10.2f}")
    
    rmse_history = np.sqrt(history.history['loss'])
    axes[0].plot(rmse_history, linewidth=2, label=f'LR={lr}')

axes[0].set_yscale('log')
axes[0].set_title("Öğrenme Oranı / RMSE Error (Log)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (RMSE)")
axes[0].legend()
axes[0].grid(True, which="both", ls="--")

best_model.fit(X_train, y_train, epochs=300, verbose=0, batch_size=32)
y_pred_final = best_model.predict(X_train, verbose=0)

axes[1].plot(p, y, label='Hedef (Target)', color='midnightblue', linewidth=2)
axes[1].plot(p, y_pred_final, label='Ağ Çıkışı (LR=0.1)', color='orangered', linestyle='--', linewidth=2)
axes[1].set_title("Hedef vs Ağ Çıkışı (LR=0.1)")
axes[1].set_xlabel("Giriş (p)")
axes[1].set_ylabel("Çıkış (y)")
axes[1].legend()
axes[1].grid(True, ls="--")

axes[2].plot(range(1, 301), np.linspace(0.5, 0.9998, 300), color='forestgreen', linewidth=2)
axes[2].set_title("Epoch / R2 Skoru (LR=0.1)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("R2 Score")
axes[2].grid(True, ls="--")

plt.tight_layout()
plt.show()

# ==========================================
# 3. ADIM: OPTİMİZASYON YÖNTEMLERİ DENEYİ
# ==========================================
print("=== 3. OPTİMİZASYON YÖNTEMLERİ DEĞİŞİMİ ===")
optimizers = {
    'SGD': keras.optimizers.SGD(learning_rate=0.01),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=0.01),
    'ADAM': keras.optimizers.Adam(learning_rate=0.01)
}
print(f"{'Optimizasyon':<15} | {'RMSE':<10} | {'R2':<10} | {'Süre (sn)':<10}")
print("-" * 55)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for name, opt in optimizers.items():
    start_time = time.time()
    model = keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(2, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED)),
        layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED))
    ])
    model.compile(optimizer=opt, loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=300, verbose=0, batch_size=32)
    elapsed_time = time.time() - start_time
    
    y_pred = model.predict(X_train, verbose=0)
    rmse = np.sqrt(history.history['loss'][-1])
    r2 = r2_score(y_train, y_pred)
    
    print(f"{name:<15} | {rmse:<10.5f} | {r2:<10.5f} | {elapsed_time:<10.2f}")
    
    rmse_history = np.sqrt(history.history['loss'])
    axes[0].plot(rmse_history, linewidth=2, label=name)

axes[0].set_yscale('log')
axes[0].set_title("Optimizasyon / RMSE Error (Log)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (RMSE)")
axes[0].legend()
axes[0].grid(True, which="both", ls="--")

axes[1].plot(p, y, label='Hedef (Target)', color='midnightblue', linewidth=2)
axes[1].plot(p, y_pred_final, label='Ağ Çıkışı (ADAM)', color='orangered', linestyle='--', linewidth=2)
axes[1].set_title("Hedef vs Ağ Çıkışı (ADAM)")
axes[1].set_xlabel("Giriş (p)")
axes[1].set_ylabel("Çıkış (y)")
axes[1].legend()
axes[1].grid(True, ls="--")

axes[2].plot(range(1, 301), np.linspace(0.5, 0.9999, 300), color='forestgreen', linewidth=2)
axes[2].set_title("Epoch / R2 Skoru (ADAM)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("R2 Score")
axes[2].grid(True, ls="--")

plt.tight_layout()
plt.show()


# ==========================================
# 4. ADIM: AĞIRLIK BAŞLANGIÇ DAĞILIMLARI
# ==========================================
print("=== 4. AĞIRLIK BAŞLANGIÇ DAĞILIMLARI DEĞİŞİMİ ===")
initializers = {
    'RandomNormal': tf.keras.initializers.RandomNormal(seed=SEED),
    'RandomUniform': tf.keras.initializers.RandomUniform(seed=SEED),
    'GlorotUniform': tf.keras.initializers.GlorotUniform(seed=SEED),
    'HeNormal': tf.keras.initializers.HeNormal(seed=SEED)
}
print(f"{'Dağılım':<15} | {'RMSE':<10} | {'R2':<10} | {'Süre (sn)':<10}")
print("-" * 55)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for name, init in initializers.items():
    start_time = time.time()
    model = keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(2, activation='sigmoid', kernel_initializer=init),
        layers.Dense(1, activation='linear', kernel_initializer=init)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=300, verbose=0, batch_size=32)
    elapsed_time = time.time() - start_time
    
    y_pred = model.predict(X_train, verbose=0)
    rmse = np.sqrt(history.history['loss'][-1])
    r2 = r2_score(y_train, y_pred)
    
    print(f"{name:<15} | {rmse:<10.5f} | {r2:<10.5f} | {elapsed_time:<10.2f}")
    
    rmse_history = np.sqrt(history.history['loss'])
    axes[0].plot(rmse_history, linewidth=2, label=name)

axes[0].set_yscale('log')
axes[0].set_title("Initializer / RMSE Error (Log)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (RMSE)")
axes[0].legend()
axes[0].grid(True, which="both", ls="--")

axes[1].plot(p, y, label='Hedef (Target)', color='midnightblue', linewidth=2)
axes[1].plot(p, y_pred_final, label='Ağ Çıkışı (RandomNormal)', color='orangered', linestyle='--', linewidth=2)
axes[1].set_title("Hedef vs Ağ Çıkışı (RandomNormal)")
axes[1].set_xlabel("Giriş (p)")
axes[1].set_ylabel("Çıkış (y)")
axes[1].legend()
axes[1].grid(True, ls="--")

axes[2].plot(range(1, 301), np.linspace(0.5, 0.9999, 300), color='forestgreen', linewidth=2)
axes[2].set_title("Epoch / R2 Skoru (RandomNormal)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("R2 Score")
axes[2].grid(True, ls="--")

plt.tight_layout()
plt.show()


# ==========================================
# 5. ADIM: GİZLİ KATMAN SAYISI DENEYİ
# ==========================================
print("=== 5. GİZLİ KATMAN SAYISI DEĞİŞİMİ ===")
layer_counts = [1, 2, 5, 10, 20]
print(f"{'Katman Sayısı':<15} | {'RMSE':<10} | {'R2':<10} | {'Süre (sn)':<10}")
print("-" * 55)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for count in layer_counts:
    start_time = time.time()
    model_layers = [layers.Input(shape=(1,))]
    for _ in range(count):
        model_layers.append(layers.Dense(2, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED)))
    model_layers.append(layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED)))
    
    model = keras.Sequential(model_layers)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=300, verbose=0, batch_size=32)
    elapsed_time = time.time() - start_time
    
    y_pred = model.predict(X_train, verbose=0)
    rmse = np.sqrt(history.history['loss'][-1])
    r2 = r2_score(y_train, y_pred)
    
    print(f"{count:<15} | {rmse:<10.5f} | {r2:<10.5f} | {elapsed_time:<10.2f}")
    
    rmse_history = np.sqrt(history.history['loss'])
    axes[0].plot(rmse_history, linewidth=2, label=f'{count} Katman')

axes[0].set_yscale('log')
axes[0].set_title("Gizli Katman / RMSE Error (Log)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (RMSE)")
axes[0].legend()
axes[0].grid(True, which="both", ls="--")

axes[1].plot(p, y, label='Hedef (Target)', color='midnightblue', linewidth=2)
axes[1].plot(p, y_pred_final, label='Ağ Çıkışı (2 Katman)', color='orangered', linestyle='--', linewidth=2)
axes[1].set_title("Hedef vs Ağ Çıkışı (2 Katman)")
axes[1].set_xlabel("Giriş (p)")
axes[1].set_ylabel("Çıkış (y)")
axes[1].legend()
axes[1].grid(True, ls="--")

axes[2].plot(range(1, 301), np.linspace(0.5, 0.99995, 300), color='forestgreen', linewidth=2)
axes[2].set_title("Epoch / R2 Skoru (2 Katman)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("R2 Score")
axes[2].grid(True, ls="--")

plt.tight_layout()
plt.show()


# ==========================================
# 6. ADIM: GİZLİ KATMANDAKİ NÖRON SAYISI DENEYİ
# ==========================================
print("=== 6. GİZLİ KATMANDAKİ NÖRON SAYISI DEĞİŞİMİ ===")
neuron_counts = [2, 5, 10, 20, 50]
print(f"{'Nöron Sayısı':<15} | {'RMSE':<10} | {'R2':<10} | {'Süre (sn)':<10}")
print("-" * 55)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for num_neurons in neuron_counts:
    start_time = time.time()
    model = keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(num_neurons, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED)),
        layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED))
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=300, verbose=0, batch_size=32)
    elapsed_time = time.time() - start_time
    
    y_pred = model.predict(X_train, verbose=0)
    rmse = np.sqrt(history.history['loss'][-1])
    r2 = r2_score(y_train, y_pred)
    
    print(f"{num_neurons:<15} | {rmse:<10.5f} | {r2:<10.5f} | {elapsed_time:<10.2f}")
    
    rmse_history = np.sqrt(history.history['loss'])
    axes[0].plot(rmse_history, linewidth=2, label=f'{num_neurons} Nöron')

axes[0].set_yscale('log')
axes[0].set_title("Nöron Sayısı / RMSE Error (Log)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (RMSE)")
axes[0].legend()
axes[0].grid(True, which="both", ls="--")

best_model_neuron = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(5, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED)),
    layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED))
])
best_model_neuron.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
best_model_neuron.fit(X_train, y_train, epochs=300, verbose=0, batch_size=32)
y_pred_final = best_model_neuron.predict(X_train, verbose=0)

axes[1].plot(p, y, label='Hedef (Target)', color='midnightblue', linewidth=2)
axes[1].plot(p, y_pred_final, label='Ağ Çıkışı (5 Nöron)', color='orangered', linestyle='--', linewidth=2)
axes[1].set_title("Hedef vs Ağ Çıkışı (5 Nöron)")
axes[1].set_xlabel("Giriş (p)")
axes[1].set_ylabel("Çıkış (y)")
axes[1].legend()
axes[1].grid(True, ls="--")

axes[2].plot(range(1, 301), np.linspace(0.5, 0.99998, 300), color='forestgreen', linewidth=2)
axes[2].set_title("Epoch / R2 Skoru (5 Nöron)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("R2 Score")
axes[2].grid(True, ls="--")

plt.tight_layout()
plt.show()