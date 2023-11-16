import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adamax
import pandas as pd
import json

# Définition des chemins
model_path = 'misc/model/EfficientNetB0-525-(224 X 224)- 98.97.h5'
train_dir = 'misc/train'
valid_dir = 'misc/valid'

# Fonction pour créer un DataFrame à partir des dossiers d'images
def create_df(directory):
    filepaths = []
    labels = []

    for fold in os.listdir(directory):
        foldpath = os.path.join(directory, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            filepaths.append(os.path.join(foldpath, file))
            labels.append(fold)

    return pd.concat([pd.Series(filepaths, name='filepaths'), pd.Series(labels, name='labels')], axis=1)

# Création des DataFrames
train_df = create_df(train_dir)
valid_df = create_df(valid_dir)

# Paramètres de base
batch_size = 32
img_size = (224, 224)
channels = 3

# Création des générateurs de données
tr_gen = ImageDataGenerator()
ts_gen = ImageDataGenerator()

train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', 
                                       target_size=img_size, class_mode='categorical', 
                                       color_mode='rgb', shuffle=True, batch_size=batch_size)

valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', 
                                       target_size=img_size, class_mode='categorical', 
                                       color_mode='rgb', shuffle=True, batch_size=batch_size)

# Chargement du modèle
model = tf.keras.models.load_model(model_path)

# Réentraînement du modèle
epochs = 10  # Nombre d'époques pour le réentraînement
model.fit(x=train_gen, epochs=epochs, validation_data=valid_gen, shuffle=False)

# Sauvegarde du modèle réentraîné
model.save(model_path)  # Écrase l'ancien modèle
class_indices = train_gen.class_indices

# Sauvegarder cette information dans un fichier JSON
with open('classes.json', 'w') as f:
    json.dump(class_indices, f)