import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from tensorflow import keras
from fg import freeze_graph
from keras.models import Sequential
from keras import optimizers
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM
from keras import backend as K
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.models import load_model
from keras.models import model_from_json

#   导入数据，注意路径
mndata = MNIST('data')
X_train, y_train = mndata.load('./emnist/emnist-byclass-train-images-idx3-ubyte',
                               './emnist/emnist-byclass-train-labels-idx1-ubyte')
X_test, y_test = mndata.load('emnist/emnist-byclass-test-images-idx3-ubyte',
                             'emnist/emnist-byclass-test-labels-idx1-ubyte')

#   处理数据
#   这里原作者用了魔法来处理数据，我就不改动了
X_train = np.array(X_train) / 255.0
y_train = np.array(y_train)
X_test = np.array(X_test) / 255.0
y_test = np.array(y_test)
X_train = X_train.reshape(X_train.shape[0], 28, 28)
X_test = X_test.reshape(X_test.shape[0], 28, 28)
for t in range(X_train.shape[0]):
    X_train[t] = np.transpose(X_train[t])
for t in range(X_test.shape[0]):
    X_test[t] = np.transpose(X_test[t])
X_train = X_train.reshape(X_train.shape[0], 784,1)
X_test = X_test.reshape(X_test.shape[0], 784,1)
def resh(ipar):
    opar = []
    for image in ipar:
        opar.append(image.reshape(-1))
    return np.asarray(opar)
train_images = X_train.astype('float32')
test_images = X_test.astype('float32')
train_images = resh(train_images)
test_images = resh(test_images)
train_labels = np_utils.to_categorical(y_train, 62)
test_labels = np_utils.to_categorical(y_test, 62)

#   模型建立
K.set_learning_phase(1)

model = Sequential()

model.add(Reshape((28,28,1), input_shape=(784,)))

model.add(Convolution2D(32, (5,5),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.5))

model.add(Dense(62, activation='softmax'))

opt = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())  #   打印模型结构

#   这可能花费数小时
history = model.fit(train_images,train_labels,validation_data=(test_images, test_labels), batch_size=128, epochs=20)

#   summarize history for accuracy
print(history.history.keys())

scores = model.evaluate(test_images,test_labels, verbose = 0)

#   测试集精确度
print("Accuracy: %.2f%%"%(scores[1]*100))

#   保存pb文件
frozen_graph = freeze_graph(K.get_session(), output_names=[model.output.op.name])
tf.train.write_graph(frozen_graph,'.','PBfile8953.pb',as_text=False)
print(model.input.op.name)
print(model.output.op.name)

#   测试你的测试集
m = X_test[258].reshape(28,28)
plt.imshow(m)
plt.show()
print('prediction: '+str(model.predict_classes(X_test[258].reshape(1,784))))

#   如果训练好了，就可以保存模型，注意文件名
model_json = model.to_json()
with open("my_model.json", "w") as json_file:
    json_file.write(model_json)
# saves the model info as json file
    
model.save_weights("my_model.h5")
# Creates a HDF5 file 'model.h5'