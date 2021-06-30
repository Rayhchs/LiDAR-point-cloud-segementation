import numpy as np
from numpy import genfromtxt
import keras
import os
import matplotlib.pyplot as plt
from main import Conv3

if __name__ == '__main__':
    b1 = genfromtxt('test/sb1.csv', delimiter=',')
    b2 = genfromtxt('test/sb2.csv', delimiter=',')
    nb1 = genfromtxt('test/snb1.csv', delimiter=',')
    nb2 = genfromtxt('test/snb2.csv', delimiter=',')
    b1 = np.delete(b1,0,axis=0)
    b2 = np.delete(b2,0,axis=0)
    nb1 = np.delete(nb1,0,axis=0)
    nb2 = np.delete(nb2,0,axis=0)

    b = np.vstack((b1, b2))
    nb = np.vstack((nb1, nb2))
    test = np.vstack((b,nb))
    y1 = np.ones(shape=(b.shape[0],1),dtype=np.float32)
    y2 = np.zeros(shape=(nb.shape[0],1),dtype=np.float32)
    label = np.vstack((y1, y2))
    del b, b1, b2, nb, nb1, nb2, y1, y2

    p_test = np.copy(test)
    p_test[:,2] = p_test[:,2]

    lx = min(p_test[:,0])
    hx = max(p_test[:,0])
    ly = min(p_test[:,1])
    hy = max(p_test[:,1])
    zmax = max(p_test[:,2])
    p_test[:,0] = p_test[:,0]-lx
    p_test[:,1] = p_test[:,1]-ly


    img_sz = 50
    height = 380
    count = -1
    test_all = []
    while True:
        count += 1
        for j in range(int((hy-ly)/img_sz)):
            x_test = []
            for i in range(test.shape[0]):
                if p_test[i,0] > + img_sz*count and p_test[i,0] <= img_sz+img_sz*count:
                    if p_test[i,1] > img_sz+img_sz*j:
                        pass
                    elif p_test[i,1] > 0+img_sz*j and p_test[i,1] <= img_sz+img_sz*j:
                        x = p_test[i,:]
                        x_test.append(x)
                else:
                    pass
            if len(x_test) < img_sz:
                pass
            elif len(x_test) > img_sz:
                test_all.append(x_test)
        if count == int((hx-lx)/img_sz):
            break
    test_all = np.array(test_all)
    space = np.zeros(shape=(test_all.shape[0],img_sz,img_sz,height,1),dtype=np.float32)

    for i in range(space.shape[0]):
        a = np.array(test_all[i])
        a[:,0] = a[:,0] % img_sz
        a[:,1] = a[:,1] % img_sz
        for j in range(a.shape[0]):
            space[i,int(a[j,0]),int(a[j,1]),int(a[j,2]),:] = a[j,2]


    length = img_sz
    width = img_sz
    n_classes = 3

    def get_model():
    	return Conv3(length,width,height,n_classes)

    from tensorflow.keras.models import *
    base_model = get_model()
    base_model.load_weights('model')
    model = Model(inputs=base_model.input, outputs=base_model.output)

    predict_all = []
    for i in range(space.shape[0]):
        test_sample = np.zeros(shape=(1,img_sz,img_sz,height,1),dtype=np.float32)
        test_sample[0,:,:,:,:] = space[i,:,:,:,:]
        predict = model.predict(test_sample)   
        predict_all.append(predict)
        
    predict_all = np.array(predict_all)
    predict_all = np.reshape(predict_all,(space.shape[0],img_sz,img_sz,n_classes))
    predict_f = np.zeros(shape=(space.shape[0],img_sz,img_sz),dtype=np.float32)
    for i in range(predict_all.shape[0]):
        for j in range(img_sz):
            for k in range(img_sz):
                if predict_all[i,j,k,2] > 0.5:
                    predict_f[i,j,k] = 1
                else:
                    predict_f[i,j,k] = 0

    pp = []

    for i in range(space.shape[0]):
        aa = np.array(test_all[i])
        aa[:,0] = aa[:,0] % img_sz
        aa[:,1] = aa[:,1] % img_sz
        a = np.zeros(shape=(aa.shape[0],1),dtype=np.float32)
        pred = predict_f[i,:,:]
        for j in range(img_sz):
            for k in range(img_sz):
                for l in range(a.shape[0]):
                    if aa[l,0] == j:
                        if aa[l,1] == k:
                            a[l,0] = pred[j,k]
                        else:
                            pass
        pp.append(a)
    pp = np.array(pp)
    predict_final = -1*(np.zeros(shape=(p_test.shape[0],1),dtype=np.float32))
    for i in range(space.shape[0]):
        aa = np.array(test_all[i])
        p = pp[i]
        for j in range(aa.shape[0]):
            a = np.where((p_test[:,0] == aa[j,0]) & (p_test[:,1] == aa[j,1]) & (p_test[:,2] == aa[j,2]))
            a = np.array(a)
            predict_final[a,0] = p[j,0]
                                
    final_output = np.zeros(shape=(p_test.shape[0],3),dtype=np.float32)
    final_output[:,0] = test[:,0]
    final_output[:,1] = test[:,1]
    final_output[:,2] = predict_final[:,0]
    np.savetxt("output.csv", final_output, delimiter=",")
        ##Evaluation
    confusion_matrix = np.zeros(shape=(n_classes,n_classes),dtype=np.float32)
    for i in range(predict_final.shape[0]):
        for j in range(n_classes):
            for k in range(n_classes):
                if predict_final[i,0] == j and label[i,0] == k:
                    confusion_matrix[k,j] += 1


    np.savetxt("confusion.csv", confusion_matrix, delimiter=",")
                            
            
        










