# ---- IMPORTAÇÃO DE MÓDULOS ----
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import os
import kagglehub



# ---- BASE DE DADOS ---- 

# Caminho da pasta do usuário
home_dir = os.path.expanduser("~")

# Caminho onde os datasets do kagglehub são armazenados
dataset_name = "raddar/tuberculosis-chest-xrays-montgomery"
dataset_folder = dataset_name.replace("/", os.sep)  
dataset_path = os.path.join(home_dir, ".cache", "kagglehub", "datasets", dataset_folder, "versions", "1")

# Verifica se o dataset já foi instalado
if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
    print("Baixando o dataset...")
    path = kagglehub.dataset_download("raddar/tuberculosis-chest-xrays-montgomery")
    print("Download concluído! Arquivos em:", path)
else:
    print("Dataset já instalado. Usando arquivos locais em:", dataset_path)



# ----- TESTE -----
exams_results = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]

exams_result_path = os.path.join(dataset_path, exams_results[0])
df_exams_results = pd.read_csv(exams_result_path)

df_exams_results


# ---- CRIANDO IA CONVOLUCIONAL ----
# # Download latest version
# path = kagglehub.dataset_download("raddar/tuberculosis-chest-xrays-montgomery")

# print("Path to dataset files:", path)

# # Definir diretório dos dados (precisa baixar o Montgomery Set manualmente)
# DATASET_PATH = "./MontgomerySet/"
# TRAIN_DIR = os.path.join(DATASET_PATH, "train")
# VALID_DIR = os.path.join(DATASET_PATH, "valid")

# # Parâmetros
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 16
# EPOCHS = 10

# # Data Augmentation e Pré-processamento
# datagen = ImageDataGenerator(
#     rescale=1.0/255, 
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2
# )

# train_generator = datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='training'
# )

# valid_generator = datagen.flow_from_directory(
#     VALID_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='validation'
# )

# # Carregar MobileNetV2 pré-treinado
# base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# base_model.trainable = False  # Congelar pesos do modelo base

# # Construir modelo
# x = GlobalAveragePooling2D()(base_model.output)
# x = Dense(128, activation='relu')(x)
# x = Dense(1, activation='sigmoid')(x)  # Saída binária

# model = Model(inputs=base_model.input, outputs=x)

# # Compilar modelo
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Treinar modelo
# history = model.fit(
#     train_generator,
#     validation_data=valid_generator,
#     epochs=EPOCHS
# )

# # Salvar modelo treinado
# model.save("tb_classifier.h5")

# print("Modelo treinado e salvo!")
