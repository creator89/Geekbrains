from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adamax
from keras.utils import np_utils
import matplotlib.pyplot as plt

np.random.seed(1672)

#сеть и её обучение
NB_EPOCH = 6
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adamax(lr=0.0032, beta_1=0.9, beta_2=0.999, decay=0.00005)
N_HIDDEN = 256
VALIDATION_SPLIT = 0.2 #какая часть обучающего набора зарезерирована для контроля
DROPOUT = 0.3 #прореживание

(x_train, y_train), (x_test, y_test) = mnist.load_data()
RESHAPED = 784

x_train = x_train.reshape(60000, RESHAPED)
x_test = x_test.reshape(10000, RESHAPED)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#нормировать
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train semples')
print(x_test.shape[0], 'test semples')

#преобразовать векторы классов в бинрные матрицы классов
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('selu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

#компиляция модели
model.compile(loss = 'categorical_crossentropy',
optimizer = OPTIMIZER,
metrics = ['accuracy'])

#обучение модели
history = model.fit(x_train, y_train,
batch_size=BATCH_SIZE, epochs=NB_EPOCH,
verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

#проверка точности
score =model.evaluate(x_test, y_test, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1]*100, '%')

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
