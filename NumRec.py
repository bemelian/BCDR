from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

IMG_ROWS = 20
IMG_COLS = 20
BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = Adam()

# ПОДГОТОВКА ТРЕНИРОВОЧНЫХ ДАННЫХ
train = open("creditcard.csv").read()
train = train.split("\n")[1:-1]  # Разделение файла на строки
train = [i.split(",") for i in train]  # Разделение строк на слова
a = np.array([[int(i[j]) for j in range(12, len(i))] for i in train])  # Массив без вспомогательных данных
y_train = np.array([int(i[2]) for i in train])  # Массив меток (цифра + 49)
X_train = np.zeros((19999, 20, 20))
X_train = a.reshape(19999, 20, 20)

# ПОДГОТОВКА ТЕСТОВЫХ ДАННЫХ
test = open("creditcardtest.csv").read()
test = test.split("\n")[1:-1]
test = [i.split(",") for i in test]
b = np.array([[int(i[j]) for j in range(12, len(i))] for i in test])
y_test = np.array([int(i[2]) for i in test])
X_test = np.zeros((len(b), 20, 20))
X_test = b.reshape(len(b), 20, 20)

# НАСТРОЙКА ПАРАМЕТРОВ РАСШИРЕНИЯ ДАТАСЕТА
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.1,
    zoom_range=0.2,
    fill_mode='nearest'
)

print('X_train shape: ', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# ПРЕОБРАЗОВАНИЕ ВЕКТОРОВ КЛАССОВ В ДВОИЧНУЮ МАТРИЦУ КЛАССОВ
Y_train = np_utils.to_categorical(y_train-48, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test-48, NB_CLASSES)

# ПЕРЕВОД В Ч/Б ФОРМАТ
X_train = X_train.reshape(X_train.shape[0], 20, 20, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 20, 1)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# ПОСТРОЕНИЕ МОДЕЛИ
model = Sequential()  # Линейное соединение слоев
model.add(Conv2D(20, (3, 3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, 1)))  # Сверточный слой
model.add(Activation('relu'))  # Функция активации ReLu
model.add(Conv2D(20, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Пулинговый слой
model.add(Dropout(0.25))  # Dropout слой
model.add(Conv2D(40, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(40, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Dense(512))  # Полносвязный слой
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())  # Сглаживание ввода (изменение формы данных в обычный массив)
model.add(Dense(NB_CLASSES))  # Выходной слой
model.add(Activation('softmax'))  # Функция активации SoftMax
model.summary()  # Вывод краткого представления модели
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])  # Конфигурация модели

# ГЕНЕРАЦИЯ ДОПЛЬНИТЕЛЬНЫХ ПРИМЕРОВ ТРЕНИРОВОЧНОГО ДАТАСЕТА
datagen.fit(X_train)

# ОБУЧЕНИЕ МОДЕЛИ
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# ТЕСТИРОВАНИЕ МОДЕЛИ
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])
model.save('NumRec.h5')
