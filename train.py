import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer

data = []
label = []
for j in range (1,4):
  for i in range (1,91):
    filename = './dataset/anh'+ str(j) + '.'  + str(i) + '.jpg'
    Img = cv2.imread(filename)
    if Img is None:
        continue  # Skip this file
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    Img = cv2.resize(src=Img, dsize=(100,100))
    Img = np.array(Img)
    data.append(Img)
    label.append(j-1)
# Preprocessing
data1 = np.array(data)
label = np.array(label)
data1 = data1.reshape(-1, 100, 100, 1)
X_train = data1/255

lb = LabelBinarizer()
trainY =lb.fit_transform(label)

Model = Sequential()
shape = (100,100, 1)
Model.add(Conv2D(32,(3,3),padding="same",input_shape=shape))
Model.add(Activation("relu"))
Model.add(Conv2D(32,(3,3), padding="same"))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Conv2D(64,(3,3), padding="same"))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Flatten())
Model.add(Dense(512))
Model.add(Activation("relu"))
Model.add(Dense(3))
Model.add(Activation("softmax"))
Model.summary()
Model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("start training")
Model.fit(X_train,trainY,batch_size=5,epochs=10)
Model.save("FaceRecognition.h5")
