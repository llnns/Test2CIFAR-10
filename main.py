# For CSCI 340
# Lucas Nesi
# Theano FLAG: set THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float64,lib.cnmem=0.5

import numpy as np
import sys
import skimage
import matplotlib.pyplot as plt
import array
import logging
from sklearn.utils import  as_float_array
from sklearn.base import *
from sknn.mlp import Classifier,Convolution, Native, Layer
from lasagne import layers as lasagne, nonlinearities as nl
from PIL import Image
from keras.layers import Input, Dense
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img,array_to_img
from keras.models import Model
from scipy import misc,linalg
from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import data, io, segmentation, color, measure,feature
import extrafunctions as ef
#################
#Configure Logger
#################
log = logging.getLogger('preparation')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%I:%M:%S')
ch.setFormatter(formatter)
log.addHandler(ch)
#################

##########################
#Configurating Constants
##########################

N_TRAIN = 45000 #Number Images for training
N_TRAIN_TEST = 5000 #Number Images for testing

#Configurating Keras Preposesing
datagen = ImageDataGenerator(featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.0,
                 height_shift_range=0.0,
                 shear_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False)
			 
#########################

log.info('Libs loaded and Consts configurated.')

#Reading Labels from CSV
log.info('Reading labels.')
labels = []
f = open("trainLabels.csv")
for line in f:
	labels.append( '"' + line.rstrip().split(',')[1] + '"' )
f.close()


#Global function for reading Images
def load_imgs(adr):
	global datagen
	img = img_to_array(load_img(adr,True))
	#img = ef.global_contrast_normalize(img[0])
	#final = np.zeros((1,32,32))
	#final[0] = img
	final = img
	return final


#Reading Training Images
log.info('Loading Training images.')	
train_images_full_info = []
for i in range(1, N_TRAIN+1):
	#print '\rLoading:',i,'/',N_TRAIN,'.',
	train_images_full_info.append(load_imgs('train/' + str(i) + '.png'))
print '\n'

Train_Input = np.array(train_images_full_info)
#Configurating labels
Train_Labels = np.array(labels[1:N_TRAIN+1])

#Preprosesing the Tran images
log.info('Preprosseing Train images.')
datagen.fit(Train_Input,False,3)
#Pre-Process and save
datagen.flow(Train_Input, Train_Labels, N_TRAIN,False,0, "./lol/","tt","png")
#Execute
(H,Y) = datagen.next()

#Reductic 1 Layer -- Error in Layer -- Verify!
Train_Input = np.zeros((len(H),1,32,32))
for i in range(len(H)):
	Train_Input[i] = H[i]

for i in range(0):
	datagen.flow(Train_Input, Train_Labels, N_TRAIN,False,0)
	(H,Y) = datagen.next()
	Train_Labels = np.concatenate((Train_Labels, Y), axis=0)
	Train_Input = np.concatenate((Train_Input, H), axis=0)



#Reading Testing Images
log.info('Loading Testing images.')	
test_images_full_info = []
for i in range(N_TRAIN+1, N_TRAIN+N_TRAIN_TEST+1):
	#print '\rLoading:',i-(N_TRAIN),'/',N_TRAIN_TEST,'.',
	test_images_full_info.append(load_imgs('train/' + str(i) + '.png'))
print '\n'

#Preprossesing Test Images
log.info('Preprosseing Test images.')
ZZ = np.array(test_images_full_info)
Z = np.zeros((len(ZZ),1,32,32))
for i in range(len(ZZ)):
	Z[i] = datagen.standardize(ZZ[i])
y_test = np.array(labels[N_TRAIN+1:N_TRAIN+1+N_TRAIN_TEST])
	
log.info('Starting Classifier.')
nn = Classifier(
    layers=[
	Convolution("Rectifier",channels=64,kernel_shape=(3,3)),
	Native(lasagne.Conv2DLayer, num_filters=32,filter_size=(3,3), nonlinearity=nl.leaky_rectify),
	Native(lasagne.MaxPool2DLayer, pool_size=(2,2)),
	Native(lasagne.DropoutLayer, p=0.25),
	Native(lasagne.Conv2DLayer, num_filters=64,filter_size=(3,3), nonlinearity=nl.leaky_rectify),
	Native(lasagne.Conv2DLayer, num_filters=64,filter_size=(3,3), nonlinearity=nl.leaky_rectify),
	Native(lasagne.MaxPool2DLayer, pool_size=(2,2)),


	#Convolution("Rectifier",channels=100,kernel_shape=(30,30)),
    Layer("Softmax")],
    learning_rate=0.0001,
    n_iter=30,
    random_state=1024,
	verbose=True,
	valid_set=(Z, y_test)
)

log.info('Starting fit.')
nn.fit(Train_Input, Train_Labels)

log.info('Starting Predict.')
Z = np.zeros((len(ZZ),32,32))
for i in range(len(ZZ)):
	Z[i] = datagen.standardize(ZZ[i])[0]

print Z.shape
result =  nn.predict(Z)

g = 0
for i in range(len(result)):
	if result[i][0]==labels[i+N_TRAIN+1]:
		g += 1
		print N_TRAIN+i,',',result[i],' - X.'
	else:
		print N_TRAIN+i,',',result[i],' = ',labels[i+N_TRAIN+1]

log.info('Final Results:%d correct from %d. %d %%',g,N_TRAIN_TEST,g*100/float(N_TRAIN_TEST))
