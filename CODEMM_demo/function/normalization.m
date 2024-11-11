function [data]=normalization(ImageData)

for i = 1: size(ImageData,3)

    if  sum(sum(ImageData(:,:,i)))==0
        data(:,:,i) =ImageData(:,:,i);
        
    else
        data(:,:,i) =ImageData(:,:,i) ./ max(max(ImageData(:,:,i)));
    end
end

end