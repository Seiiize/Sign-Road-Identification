# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 18:14:05 2022

@author: Youcef
"""
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
from numpy import unique, empty, arange
from random import seed as seed2
from tensorflow import random
from numpy.random import seed
from PIL import Image
from matplotlib.pyplot import plot, xlim, ylim, xlabel, ylabel, title, legend, subplot, figure,show,subplots
import seaborn as sns
# classe utilisé pour arreter l'apprentissage lorsque la precision de la validation atteint un seuil qu'on précise
class MyThresholdCallback(Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_accuracy"]
        tr_acc=logs['accuracy']
        if val_acc >= self.threshold and tr_acc >=0.995 :
            self.model.stop_training = True
threshold_callback = MyThresholdCallback(threshold=0.995)
# Setting seed for reproducibility
seed2(42)
seed(42)
random.set_seed(42)

# Définir le chemin des images
path = "C:\\Users\\Lenovo\\Desktop\\TPIA\\TP5\\sign_road_image\\"


#Importer le CSV 
data = read_csv(path + "32_road_sign_dataset.csv")
# Attribuer les noms de classes et le nom du fichier à deux variables
file_name = data['Filename']
class_name = data['Sign']
fig, ax = subplots()
sns.countplot(x='Sign', data=data, ax=ax) 
# fig=sns.countplot(data=df)      
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2., p.get_height(), f'{p.get_height()}', 
            ha='center', va='bottom', fontsize=10)
# Print le nombre de classes
figure()
print(unique(class_name))

# Division de nos données en données de training, validation et de test
file_temp, file_test, y_temp, y_test = train_test_split(file_name, class_name, test_size=0.40, random_state=32)
file_train, file_val, y_train, y_val = train_test_split(file_temp, y_temp, test_size=0.30, random_state=32)

# Encoder les données d'apprentissage et de validation et de test
y_encoder =  LabelEncoder() 
y_train_encoder = y_encoder.fit_transform(y_train)
y_val_encoder = y_encoder.fit_transform(y_val)
y_test_encoder = y_encoder.fit_transform(y_test)
print(y_train_encoder) 

# Calcul de la taille de chacune des données
n_images_train = len(file_train)
n_images_val = len(file_val)
n_images_test = len(file_test)

# Création de matrices vides
X_train = empty((n_images_train, 32, 32, 3), dtype="float")
X_val = empty((n_images_val, 32, 32, 3), dtype="float")
X_test = empty((n_images_test, 32, 32, 3), dtype="float")

# Remplissage des matrices
i = 0
for file in file_train:
    X_train[i,:,:,:] = Image.open(path + file)
    i = i + 1
i = 0
for file in file_val:
    X_val[i,:,:,:] = Image.open(path + file)
    i = i + 1
i=0
for file in file_test:
    X_test[i,:,:,:] = Image.open(path + file)
    i = i + 1

# Normalizing data
X_train = X_train
X_val = X_val
X_test = X_test

# Definir la taille des images
size_image = 32 

# Configuration du modéle 
model = Sequential()
#Ajout d'une couche convolutive
model.add(Conv2D(2, (3,3), activation="relu", input_shape=(size_image,size_image,3)))
#Ajout d'une couche Maxpooling
model.add(MaxPooling2D(2,2))
# Ajout d'un dropout
model.add(Dropout(0.4))
#Ajout d'une normalisation de Batch
model.add(BatchNormalization())
#Ajout d'une couche convolutive
model.add(Conv2D(2, (2, 2), activation="relu"))
#Ajout d'une couche Maxpooling
model.add(MaxPooling2D(2, 2))
#Ajout d'une normalisation de Batch
model.add(BatchNormalization())
#Ajout d'une classe Flatten 
model.add(Flatten())
# Ajout d'un dropout
model.add(Dropout(0.15))
#Ajout d'une couche de sortie
model.add(Dense(4, activation = 'softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
model.summary()

# Training the model and retrieving performance versus epochs
n_epochs = 1022
b_size = 25

performance = model.fit(X_train, y_train_encoder, validation_data=(X_val, y_val_encoder), batch_size=b_size, epochs=n_epochs,callbacks=[threshold_callback])

# Retrieving Loss and Accuracy for Training and Validation datasets
loss_train = performance.history['loss']
loss_val = performance.history['val_loss']

accuracy_train = performance.history['accuracy']
accuracy_val = performance.history['val_accuracy']

# Plotting Loss and Accuracy curves for Training et Validation datasets
figure(1)
epoch = arange(1,len(loss_train)+1)

subplot(1,2,1)
plot(epoch, loss_train, label='Train')
plot(epoch, loss_val, label='Val')
xlim(1,len(epoch))
xlabel('Epoch', fontsize=10)
ylabel('Loss', fontsize=10)
title('Multiclass Cross Entropy', fontsize=10)
legend()

subplot(1,2,2)
plot(epoch, accuracy_train, label='Train')
plot(epoch, accuracy_val, label='Val')
xlim(1,len(epoch))
ylim(0,1)
xlabel('Epoch', fontsize=10)
ylabel('Accuracy', fontsize=10)
title('Multiclass Cross Entropy', fontsize=10)
legend()

# Evaluating the model on Training and Validation datasets
pred_y_train = model.predict(X_train).argmax(axis=-1)
cm_train = confusion_matrix(y_train_encoder, pred_y_train)
cmd_train = ConfusionMatrixDisplay(cm_train)
cmd_train.plot()
cmd_train.ax_.set_title("Confusion Matrix for Training Dataset")
c_report_train = classification_report(y_train_encoder, pred_y_train)
print("Confusion matrix performed on Training dataset:", cm_train)
print("Classification report performed on Training dataset:", c_report_train)

pred_y_val = model.predict(X_val).argmax(axis=-1)
cm_val = confusion_matrix(y_val_encoder, pred_y_val)
cmd_val = ConfusionMatrixDisplay(cm_val)
cmd_val.plot()
cmd_val.ax_.set_title("Confusion Matrix for Validation Dataset")
c_report_val = classification_report(y_val_encoder, pred_y_val)
print("Confusion matrix performed on Validation dataset:", cm_val)
print("Classification report performed on Validation dataset:", c_report_val)
pred_y_test=model.predict(X_test).argmax(axis=-1)
acctest=accuracy_score(y_test_encoder,pred_y_test)
print(acctest)
show()