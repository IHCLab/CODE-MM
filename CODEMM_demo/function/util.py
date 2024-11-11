from keras.models import Model
from tensorflow.keras.layers import Input,Dense,Flatten,Conv3D,Lambda
from tensorflow.keras import backend as K
import numpy as np

def prepare_data(image, standardize = False): 
    
    img_tmp = image.copy()

    if standardize == "scale":
        
        for i in range(image.shape[2]):
            
            img_tmp[:,:,i] = image[:,:,i] / np.max(image[:,:,i])
            
    return img_tmp

def euclid_dis(vects):
    
    x,y = vects
    sum_square = K.sum(K.square(x-y), axis=1, keepdims=True)
    
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    
    shape1, shape2 = shapes
    
    return (shape1[0], 1)

def createImageCubes(X, windowSize):

    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.empty((X.shape[0]* X.shape[1]* X.shape[2], windowSize, windowSize, X.shape[3]), dtype=np.float32)
    patchIndex = 0
  
    for r in range(margin, zeroPaddedX.shape[1] - margin):
        
        for c in range(margin, zeroPaddedX.shape[2] - margin):        
        
            patchesData[patchIndex, :, :, :] = zeroPaddedX[0,r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchIndex = patchIndex + 1

    return patchesData

def padWithZeros(X, margin):
    
    newX = np.zeros((X.shape[0],X.shape[1] + 2 * margin, X.shape[2] + 2 * margin, X.shape[3]))
    x_offset = margin
    y_offset = margin
    
    for i in range(0,X.shape[0]):
        
        newX[i,x_offset:X.shape[1] + x_offset, y_offset:X.shape[2] + y_offset, :] = X[i,:,:,:]    
    
    return newX

def create_base_net(input_shape,case = False):  
    
    input = Input(shape = input_shape)

    if case == "sentinel":
        
        x = Conv3D(6, 3, activation = 'relu')(input)
        x = Conv3D(8, 3, activation = 'relu')(x)
        x = Conv3D(10, 5, activation = 'relu')(x)
        x = Flatten()(x)
        x = Dense(64,activation = 'relu')(x)
        x = Dense(16,activation = 'relu')(x)   
        x = Dense(6,activation = 'relu')(x) 
    
    elif case == "hyperion":
                
        x = Conv3D(3, 3, activation = 'relu')(input)
        x = Flatten()(x)
        x = Dense(2500,activation = 'relu')(x)  
        x = Dense(1200,activation = 'relu')(x)  
        x = Dense(600,activation = 'relu')(x) 
        x = Dense(300,activation = 'relu')(x) 
        x = Dense(60,activation = 'relu')(x) 
        x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
             
    model = Model(input, x)

    return model

def SEMVI_imageCube(Data,cutoff, windowSize,case = False):
    
    if case == "sentinel":
        
        SEMVI=(Data[:,:,7]-Data[:,:,2])/(Data[:,:,10]+Data[:,:,11]-2*Data[:,:,2])   
        Data_nor=prepare_data(Data, standardize = 'scale')
    
    elif case == "hyperion": 
        
        SEMVI=(Data[:,:,41]-Data[:,:,13])/(Data[:,:,113]+Data[:,:,157]-2*Data[:,:,13]+1e-6)         
        Data_nor=prepare_data(Data, standardize = 'false')

    margin = int((windowSize - 1) / 2)
    data_pad=padWithZeros(np.expand_dims(Data_nor,axis=0), margin)
    
    Map=(SEMVI>cutoff)
    index_num=np.where(Map==1)
    
    patch=np.zeros((index_num[0].shape[0],windowSize, windowSize, Data.shape[2]), dtype=np.float32)

    new_index=[index_num[0]+margin,index_num[1]+margin]
    
    for i in range(0,len(index_num[0])):
        
        patch[i]=data_pad[0, new_index[0][i]-margin:new_index[0][i]+margin+1, new_index[1][i]-margin:new_index[1][i]+margin+1, :]                         
    
    return patch