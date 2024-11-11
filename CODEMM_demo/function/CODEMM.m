%================================================================================
% Programmer: Po-Wei Tang
% E-mail: q38091526@gs.ncku.edu.tw
% Date: Oct. 11, 2023
% -------------------------------------------------------------------------------
% Reference:
% ``CODE-MM: Convex Deep Mangrove Mapping Algorithm Based On Optical Satellite
% Images," accepted by IEEE Transactions on Geoscience and Remote Sensing, 2023.
%================================================================================
% [binary_map,time] = CODEMM(Data,XDL,Mode)
%================================================================================
% Input
% Data is the observed image for analysis, whose dimension is row*col*band.
% XDL is the classification map from a weak Adam optimizer-based Siamese network.
% Mode is set to compute the result under multispectral (i.e., Mode=0) or
% hyperspectral (i.e., Mode=1) case.
%--------------------------------------------------------------------------------
% Output
% binary_map is the classification result, whose dimension is row*col.
% time is the computation time (in seconds).
%================================================================================
function [binary_map,time]=CODEMM(Data,XDL,Mode)
tic
[row,col,band]=size(Data);
L=row*col;
%% Compute f(Z)
data=normalization(Data);
Z= reshape(data,L,band)';
switch Mode
    case "Mode=0"
        alpha=6;
        SEMVI_b=[3 8 11 12];
    case "Mode=1"
        alpha=10;
        SEMVI_b=[14 42 114 158]; %after band removal
end
SEMVI=(Data(:,:,SEMVI_b(2))-Data(:,:,SEMVI_b(1)))./(Data(:,:,SEMVI_b(3))+Data(:,:,SEMVI_b(4))-2*Data(:,:,SEMVI_b(1)));
SEMVI=SEMVI>alpha;
fZ=mean(Z(:,SEMVI==1),2);
%% Parameters
c=1e-3;            % ADMM penalty parameter
lambda=5e-1;       % regularization parameter
%% ADMM iter
for iter=0:1000
    if iter==0
        x=zeros(L,1);
        xdl=reshape(XDL,[],1);
        x_denominator=(2*(vecnorm(Z-fZ).^2)+c).';
        d=zeros(L,1);
    end
    %update z
    z=lambda/(lambda+c)*xdl+c/(lambda+c)*(x-d);
    z(z>1)=1;
    z(z<0)=0;
    %update x
    x=c*(z+d)./x_denominator;
    %update d
    d=d-(x-z);
end
x_star=reshape(x,row,col);
binary_map=zeros(row,col);
switch Mode
    case "Mode=0"
        binary_map(x_star>0.5)=1;
    case "Mode=1"
        x_rmoutliers = rmoutliers(x,"percentiles",[0,75]);
        binary_map(x_star>max(x_rmoutliers)/2)=1;
end
toc
time=toc;
end