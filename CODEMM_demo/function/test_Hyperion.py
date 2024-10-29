import numpy as np
import time
import scipy.io
import os
from util import prepare_data,create_base_net,Input,Lambda,euclid_dis,eucl_dist_output_shape,Model,createImageCubes,SEMVI_imageCube

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#%% Load data

File_path = "./Data/Hyperion_TestData.mat"
Data_temp=(scipy.io.loadmat(File_path))['Data_HP']
GT_temp=(scipy.io.loadmat(File_path))['GT_HP']
Data3d=prepare_data(Data_temp, standardize = "false") 
Data3d=Data3d.astype(np.double)
GT_temp=GT_temp.astype(np.double)
Data_temp=Data_temp.astype(np.double)

#%% Load model

alpha=10
windowSize=5
input_shape=(windowSize,windowSize,175,1)
base_network = create_base_net(input_shape,case = "hyperion")
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
warmup= base_network(np.expand_dims(np.zeros(input_shape), axis=0))
processed_a = base_network(input_a)
processed_b = base_network(input_b)
distance = Lambda(euclid_dis,output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = Model([input_a, input_b], distance)
model.load_weights("./ckpt/Siamese_model_Hyperion.h5")

#%% Test

start = time.time()

T_zi=createImageCubes(np.expand_dims(Data3d, axis=0), windowSize=windowSize)
h_T_zi= base_network(T_zi)
h_T_zi_arr=h_T_zi.numpy()
SEMVI_patch=SEMVI_imageCube(Data_temp,alpha, windowSize,case = "hyperion")
T_fZ=np.mean(SEMVI_patch, axis=0)
h_T_fZ= base_network(T_fZ[None,:])
h_T_fZ_arr=h_T_fZ.numpy()
h_T_fZ_arr=np.repeat(h_T_fZ_arr,Data_temp.shape[0]*Data_temp.shape[1],axis=0)
dist_map=np.sqrt(np.sum(np.square(h_T_fZ_arr-h_T_zi_arr),axis=1, keepdims=True))
clas_map=(dist_map<0.5).astype(np.double)
preds_Siamese = np.array(clas_map).reshape((Data_temp.shape[0],Data_temp.shape[1]))

end = time.time()
print("Siamese: %f seconds" % (end - start))

#%% Save  data

File_save ="./results/Hyperion_TestData_SiameseOutput.mat"
scipy.io.savemat(File_save,{'XDL_HP': preds_Siamese,'time_xdl':end - start})