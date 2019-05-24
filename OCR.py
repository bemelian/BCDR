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
from PIL import Image

OPTIM = RMSprop()

img_name = 'img/card.jpg'

img = np.transpose(Image.fromarray(imageio.imread(img_name)).resize((855, 540), Image.LANCZOS), (0, 1, 2))
img = rgb2gray(img)

'''
# СЕГМЕНТАЦИЯ ИЗОБРАЖЕНИЯ
#  img = exposure.equalize_hist(img)
#  img = filters.scharr(img)
#  img = white_tophat(img)

th = img > threshold_otsu(img)
for i in range(540):
    for j in range(855):
        if th:
            img[i][j] = 1
        else:
            img[i][j] = 0
'''
imageio.imwrite('img/cardtest.jpg', img)


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
    imgs[i] = Image.fromarray(imgtest[i]).resize((20, 20), Image.LANCZOS)
    imageio.imwrite('img/cardtestnum' + str(i) + '.jpg', imgs[i])

'''
# СЕГМЕНТАЦИЯ БУКВ ИМЕНИ ДЕРЖАТЕЛЯ КАРТЫ 4374
th = img > threshold_local(img, 31, method='mean')
for i in range(540):
    for j in range(855):
        if th[i][j]:
            img[i][j] = 1
        else:
            img[i][j] = 0

imageio.imwrite('img/cardtest.jpg', img)

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
th = img > threshold_local(img, 31, method='mean')
for i in range(540):
    for j in range(855):
        if th[i][j]:
            img[i][j] = 1
        else:
            img[i][j] = 0

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
for a in range(14):
    imgs1[a] = Image.fromarray(imgtest1[a]).resize((20, 20), Image.LANCZOS)
    # th = imgs1[a] > threshold_otsu(imgs1[a])
    # for i in range(20):
    #     for j in range(20):
    #         if th[i][j]:
    #             imgs1[a][i][j] = 1
    #         else:
    #             imgs1[a][i][j] = 0
    imageio.imwrite('img/cardtestnum' + str(a) + '.jpg', imgs1[a])

# РАСПОЗНАВАНИЕ НОМЕРА КАРТЫ
imgs = imgs.reshape(16, 20, 20, 1)
model = load_model('NumRec.h5')
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])

predictions = model.predict_classes(imgs)
print(predictions)

# РАСПОЗНАВАНИЕ ИМЕНИ ДЕРЖАТЕЛЯ КАРТЫ
imgs1 = imgs1.reshape(14, 20, 20, 1)
model1 = load_model('LetRec.h5')
model1.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])

predictions1 = model1.predict_classes(imgs1)
print([chr(predictions1[i]+65) for i in range(14)])
