import caffe
import numpy as np
# http://stackoverflow.com/questions/33828582/vgg-face-descriptor-in-python-with-caffe
img = caffe.io.load_image( "ak.png" )
img = img[:,:,::-1]*255.0 # convert RGB->BGR
avg = np.array([129.1863,104.7624,93.5940])
img = img - avg # subtract mean (numpy takes care of dimensions :)
img = img.transpose((2,0,1))
img = img[None,:] # add singleton dimension
net = caffe.Net("VGG_FACE_deploy.prototxt","VGG_FACE.caffemodel",  caffe.TEST)
out = net.forward_all( data = img )
caffe_fc8 = net.blobs['fc8'].data[0]
r2 = np.argmax(caffe_fc8)

names = open('names.txt')
names = names.readlines()
print names[r2]
