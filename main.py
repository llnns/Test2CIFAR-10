import numpy as np

from sknn.mlp import Classifier,Convolution, Native, Layer
from lasagne import layers as lasagne, nonlinearities as nl
from PIL import Image
import sys
from array import array
labels = []
f = open("trainLabels.csv")
for line in f:
	labels.append( '"' + line.rstrip().split(',')[1] + '"' )
f.close()

MAX_LER = 2000
MAX_LER_TEST = 1000
from skimage import io, measure,feature
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt


arr = []
for i in range(0,32*32):
	for j in ['r','g','b']:
		arr.append('"px' + j + str(i) + '"')
arr.append('"class"')


					
rgb_all = []
for x in range(1, MAX_LER+1):

	#im = Image.open('train/' + str(x) + '.png').convert('LA')
	im = Image.open('train/' + str(x) + '.png')
	
	img = Image.open('train/' + str(x) + '.png')
	gray = img.convert('L')   # 'L' stands for 'luminosity'
	gray = np.asarray(gray)

	image = io.imread('train/' + str(x) + '.png')
	#gray = image.sum(-1)
#	gray = image.convert('L')
	edges = feature.canny(gray, sigma=1)
	#plt.imshow(edges)
	#plt.show()
	print '\r Carregando:',x,
	#print edges
	rgb_all.append(edges)
	
print '..ok\n'

rgb_all_test = []
for x in range(MAX_LER+1, MAX_LER+MAX_LER_TEST+1):

	img = Image.open('train/' + str(x) + '.png')
	gray = img.convert('L')   # 'L' stands for 'luminosity'
	gray = np.asarray(gray)

	#gray = image.sum(-1)
#	gray = image.convert('L')
	edges = feature.canny(gray, sigma=1)

	print '\r Carregando Test:',x,
	#print edges
	rgb_all_test.append(edges)
	
def store_stats(avg_train_error, **_):
	store_stats.counter+=1 
	print '\t',store_stats.counter,avg_train_error,

store_stats.counter=0

print '\nStart classifier...',

X = np.array(rgb_all)
Z = np.array(rgb_all_test)
y = np.array(labels[1:MAX_LER+1])

nn = Classifier(
    layers=[
	Layer("Tanh", units=10),
	Layer("Rectifier", units=10),
	Layer("Tanh", units=10),
	Layer("Rectifier", units=10),
	Layer("Tanh", units=10),
	Layer("Rectifier", units=10),
	Layer("Tanh", units=10),
	Layer("Rectifier", units=10),
	
	
    Layer("Softmax")],
    learning_rate=0.01,
    n_iter=100,
    random_state=1024,
	callback={'on_epoch_finish': store_stats}
)
print 'ok\nFit:'
nn.fit(X, y)
print 'a'
g = 0
print 'Fit..ok\nPredict...'
result =  nn.predict(Z)
for i in range(len(result)):
	if result[i][0]==labels[MAX_LER+i]:
		g += 1

for i in range(len(result)):
	print i+1,',',result[i]
#print y	
print g