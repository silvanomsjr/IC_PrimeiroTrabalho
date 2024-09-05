import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Função para carregar os arquivos de áudio de todas as pastas fold
def load_urban_sound_data(dataset_path, sample_rate=8000):
    data = []
    labels = []

    # Percorre cada pasta fold1, fold2, ..., fold10
    for fold in range(1, 11):  # De fold1 a fold10
        fold_path = os.path.join(dataset_path, f"fold{fold}")

        # Percorre todos os arquivos .wav dentro da pasta
        for file_name in os.listdir(fold_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(fold_path, file_name)

                # Carrega o arquivo de áudio usando librosa
                waveform, _ = librosa.load(file_path, sr=sample_rate, mono=True)

                # Normaliza o tamanho do waveform (32k amostras)
                if len(waveform) < 32000:
                    waveform = np.pad(waveform, (0, 32000 - len(waveform)), "constant")
                else:
                    waveform = waveform[:32000]

                data.append(waveform)

                # Exemplo de rótulos: Usando o número da pasta como rótulo (ajustar conforme necessário)
                labels.append(fold - 1)  # Ajuste de acordo com seu critério de rótulos

    # Converte as listas para arrays NumPy
    data = np.array(data)
    labels = np.array(labels)

    return data, labels


# Função para construir a CNN baseada na arquitetura descrita no artigo
def build_cnn(input_shape):
    model = models.Sequential()

    # Primeira camada de convolução com campo receptivo grande (para imitar filtros de passa-faixa)
    model.add(
        layers.Conv1D(
            256, kernel_size=80, strides=4, activation="relu", input_shape=input_shape
        )
    )
    model.add(layers.MaxPooling1D(pool_size=4))

    # Camadas de convolução subsequentes com campo receptivo pequeno
    model.add(layers.Conv1D(128, kernel_size=3, activation="relu"))
    model.add(layers.MaxPooling1D(pool_size=4))

    model.add(layers.Conv1D(64, kernel_size=3, activation="relu"))
    model.add(layers.MaxPooling1D(pool_size=4))

    # Mais camadas profundas para melhorar a capacidade de generalização
    model.add(layers.Conv1D(64, kernel_size=3, activation="relu"))
    model.add(layers.Conv1D(48, kernel_size=3, activation="relu"))

    # Camada de Pooling Global Média para reduzir a dimensionalidade
    model.add(layers.GlobalAveragePooling1D())

    # Camada de saída com softmax para classificação
    model.add(
        layers.Dense(10, activation="softmax")
    )  # 10 classes no exemplo de reconhecimento de sons ambientais

    return model


# Caminho para a pasta principal do dataset UrbanSound8K
dataset_path = "./urbansound8k"

# Carregar os dados de áudio e rótulos
x_data, y_data = load_urban_sound_data(dataset_path)

# Adiciona a dimensão do canal para a CNN
x_data = np.expand_dims(x_data, axis=-1)

# Dividir os dados entre treino e validação (80% treino, 20% validação)
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42
)

# Definir o formato de entrada
input_shape = (32000, 1)  # Tamanho do vetor da forma de onda (32k amostras)

# Construir o modelo CNN
model = build_cnn(input_shape)

# Definir a taxa de aprendizado e compilar o modelo
learning_rate = 0.005
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Exibir o resumo da arquitetura do modelo
model.summary()

# Treinar o modelo
history = model.fit(
    x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val)
)

# Previsão nos dados de validação
y_val_pred = np.argmax(model.predict(x_val), axis=1)

# Calcular as métricas
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, average="weighted")
recall = recall_score(y_val, y_val_pred, average="weighted")
f1 = f1_score(y_val, y_val_pred, average="weighted")

# Mostrar os resultados
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
