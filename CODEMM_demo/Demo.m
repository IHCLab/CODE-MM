clc;clear;close all;
addpath(genpath('.\ckpt'));addpath(genpath('.\Data'));
addpath(genpath('.\function'));addpath(genpath('.\results'));
%% Load Data
load Madagascar.mat
%% Test multispectral case
% Compute XDE
save('.\Data\Sentinel2_TestData.mat','Data','GT')
system(['conda activate env & python function/test_Sentinel.py'])
load  Sentinel2_TestData_SiameseOutput.mat;
% Algorithm
[binary_map,time_co]=CODEMM(Data,XDL,'Mode=0');
% Plot
plot_result(Data,GT,binary_map,'Mode=0',time_xdl,time_co);
%% Test hyperspectral case
% Compute XDE
save('.\Data\Hyperion_TestData.mat','Data_HP','GT_HP')
system(['conda activate env & python function/test_Hyperion.py'])
load  Hyperion_TestData_SiameseOutput.mat;
% Algorithm
[binary_map,time_co]=CODEMM(Data_HP,XDL_HP,'Mode=1');
% Plot
plot_result(Data_HP,GT_HP,binary_map,'Mode=1',time_xdl,time_co);