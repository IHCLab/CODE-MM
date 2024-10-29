import scipy.io
import numpy as np
import time
import os
from util import prepare_data,Input,create_base_net,Lambda,euclid_dis,eucl_dist_output_shape,Model,createImageCubes,SEMVI_imageCube

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#%% Load data

File_path = "./Data/Sentinel2_TestData.mat"
Data=(scipy.io.loadmat(File_path))['Data']
GT=(scipy.io.loadmat(File_path))['GT']
Data_nor=prepare_data(Data, standardize = "scale")
Data_nor=Data_nor.astype(np.double)
GT=GT.astype(np.double)
Data=Data.astype(np.double)

#%% Load model 

alpha=6
windowSize=9
input_shape=(windowSize,windowSize,12,1)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
base_network = create_base_net(input_shape,case="sentinel")
warmup= base_network(np.expand_dims(np.zeros(input_shape), axis=0))
processed_a = base_network(input_a)
processed_b = base_network(input_b)
distance = Lambda(euclid_dis,output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model_siamese = Model([input_a,input_b], distance)
model_siamese.load_weights('./ckpt/Siamese_model_Sentinel2.h5')

#%% Test   

siamese_start = time.time()

T_zi=createImageCubes(np.expand_dims(Data_nor, axis=0), windowSize=windowSize)
h_T_zi= base_network(T_zi)
h_T_zi_arr=h_T_zi.numpy()
SEMVI_patch=SEMVI_imageCube(Data,alpha, windowSize,case = "sentinel")
T_fZ=np.mean(SEMVI_patch, axis=0) 
h_T_fZ= base_network(T_fZ[None,:])
h_T_fZ_arr=h_T_fZ.numpy()
h_T_fZ_arr=np.repeat(h_T_fZ_arr,Data.shape[0]*Data.shape[1],axis=0)
dist_map=np.sqrt(np.sum(np.square(h_T_fZ_arr-h_T_zi_arr),axis=1, keepdims=True))
clas_map=(dist_map<0.4).astype(np.double)
preds_Siamese = np.array(clas_map).reshape((Data.shape[0],Data.shape[1]))

siamese_end = time.time()
print("Siamese: %f seconds" % (siamese_end - siamese_start))

#%% Save  data

File_save ="./results/Sentinel2_TestData_SiameseOutput.mat"
scipy.io.savemat(File_save,{'XDL':preds_Siamese,'time_xdl':siamese_end-siamese_start})