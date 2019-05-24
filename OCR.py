import skimage
from keras.models import load_model
from keras.optimizers import RMSprop
from skimage import filters, measure
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import white_tophat
from skimage.color import rgb2gray, gray2rgb
from skimage import feature
from skimage import exposure
import numpy as np
import scipy.misc
import imageio
OPTIM = RMSprop()

img_name = 'card.jpg'

img = np.transpose(scipy.misc.imresize(imageio.imread(img_name), (540, 855)), (0, 1, 2))
img = rgb2gray(img)

'''
# СЕГМЕНТАЦИЯ ИЗОБРАЖЕНИЯ
#  img = exposure.equalize_hist(img)
#  img = filters.scharr(img)
#  img = white_tophat(img)

th = img > threshold_otsu(img)
for i in range(540):
    for j in range(855):
        if th[i][j]:
            img[i][j] = 1
        else:
            img[i][j] = 0
imageio.imwrite('cardtest.jpg', img)
'''

# СЕГМЕНТАЦИЯ ЦИФР НОМЕРА КАРТЫ
imgtest = np.zeros((16, 50, 40))
for a in range(4):
    for b in range(4):
        ii = 0
        for i in range(300, 350):
            jj = 0
            for j in range(80+a*180+b*38, 120+a*180+b*38):
                imgtest[a*4+b][ii][jj] = img[i][j]
                jj += 1
            ii += 1

imgs = np.zeros((16, 20, 20))
for i in range(16):
    imgs[i] = scipy.misc.imresize(imgtest[i], (20, 20), interp='lanczos', mode='L')
    #  imageio.imwrite('cardtestnum' + str(i) + '.jpg', imgs[i])

th = img > threshold_local(img, 31, method='mean')
for i in range(540):
    for j in range(855):
        if th[i][j]:
            img[i][j] = 255
        else:
            img[i][j] = 0

imageio.imwrite('cardtest.jpg', img)
'''
# СЕГМЕНТАЦИЯ БУКВ ИМЕНИ ДЕРЖАТЕЛЯ КАРТЫ 4374
th = img > threshold_local(img, 31, method='mean')
for i in range(540):
    for j in range(855):
        if th[i][j]:
            img[i][j] = 1
        else:
            img[i][j] = 0

imageio.imwrite('cardtest.jpg', img)

imgtest1 = np.zeros((14, 40, 25))
for a in range(7):
    ii = 0
    for i in range(435, 475):
        jj = 0
        for j in range(50+a*25, 75+a*25):
            imgtest1[a][ii][jj] = img[i][j]
            jj += 1
        ii += 1

for a in range(7, 14):
    ii = 0
    for i in range(435, 475):
        jj = 0
        for j in range(80+a*25, 105+a*25):
            imgtest1[a][ii][jj] = img[i][j]
            jj += 1
        ii += 1
'''

# СЕГМЕНТАЦИЯ БУКВ ИМЕНИ ДЕРЖАТЕЛЯ КАРТЫ 7217

imgtest1 = np.zeros((14, 36, 22))
for a in range(7):
    ii = 0
    for i in range(435, 471):
        jj = 0
        for j in range(64+a*25, 86+a*25):
            imgtest1[a][ii][jj] = img[i][j]
            jj += 1
        ii += 1

for a in range(7, 14):
    ii = 0
    for i in range(435, 471):
        jj = 0
        for j in range(94+a*25, 116+a*25):
            imgtest1[a][ii][jj] = img[i][j]
            jj += 1
        ii += 1
imgs1 = np.zeros((14, 20, 20))
for i in range(14):
    imgs1[i] = scipy.misc.imresize(imgtest1[i], (20, 20), interp='lanczos', mode='L')
    imageio.imwrite('cardtestnum' + str(i) + '.jpg', imgs1[i])

# РАСПОЗНАВАНИЕ НОМЕРА КАРТЫ
imgs = imgs.reshape(16, 20, 20, 1)
model = load_model('NumRec.h5')
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])

predictions = model.predict_classes(imgs)
print(predictions)

'''
imgs1 = np.zeros((14, 20, 20))
train = open("ocrb.csv").read()
train = train.split("\n")[1:-1]  # Разделение файла на строки
train = [i.split(",") for i in train]  # Разделение строк на слова
x = np.array([[int(i[j]) for j in range(12, len(i))] for i in train])  # Массив без вспомогательных данных
X_train = np.zeros((len(x), 20, 20))
X_train = x.reshape(len(x), 20, 20)
X_train = X_train.reshape(X_train.shape[0], 20, 20, 1)
X_train = X_train.astype("float32")
X_train /= 255
for i in range(14):
    imgs1[i] = X_train[i].reshape(20, 20)
    imageio.imwrite('cardtestnum' + str(i) + '.jpg', imgs1[i])
'''
# РАСПОЗНАВАНИЕ ИМЕНИ ДЕРЖАТЕЛЯ КАРТЫ
imgs1 = imgs1.reshape(14, 20, 20, 1)
model1 = load_model('LetRec.h5')
model1.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])

predictions1 = model1.predict_classes(imgs1)
print([chr(predictions1[i]+65) for i in range(14)])
