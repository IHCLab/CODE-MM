function plot_result(Data,GT,binary_map,mode,time_xdl,time_co)
%% Assessment
ADMM_ass=assessment(GT(:),binary_map(:),'class');
%% Plot
switch mode
    case "Mode=0"
        figure('Name','MSI demo');
        set(gcf, 'Position', [300, 100, 600, 400]); % [left, bottom, width, height]
        [RGB]=plot_falsecolor(Data,4,3,2,2.5,1,1);
        sgtitle(sprintf('Qualitative study of Sentinel-2 MSI from ESA over Madagascar.'));      
    case "Mode=1"
        figure('Name','HSI demo');
        set(gcf, 'Position', [1000, 100, 600, 400]); % [left, bottom, width, height]
        [RGB]=plot_falsecolor(Data,24,14,11,3,1,1);
        sgtitle(sprintf('Qualitative study of Hyperion HSI from NASA over Madagascar.'));
end
subplot(1,3,1)
imshow(RGB);title('RGB')
xlabel(["OA (%)","AA (%)","\kappa",'Time (sec.)']);
subplot(1,3,2)
imshow(GT);title('GT')
subplot(1,3,3)
imshow(binary_map);title('CODE-MM');
xlabel([round(ADMM_ass.OA,2),round(ADMM_ass.AA,2),round(ADMM_ass.Kappa,3),round((time_xdl+time_co),4)]);
end
function [RGB1]=plot_falsecolor(Xim,r,g,b,brightness,saturation,tone)
R = Xim(:,:,r);
R= (R-min(R(:)))/(max(R(:))-min(R(:)));
G = Xim(:,:,g);
G= (G-min(G(:)))/(max(G(:))-min(G(:)));
B = Xim(:,:,b);
B= (B-min(B(:)))/(max(B(:))-min(B(:)));
RGB(:,:,1)=R;
RGB(:,:,2)=G;
RGB(:,:,3)=B;
[m,n,~] = size(RGB);
hsv = rgb2hsv(RGB);
for ii = 1:m
    for j = 1: n
        hsv(ii,j,3) =brightness* hsv(ii,j,3);
        hsv(ii,j,2) =saturation* hsv(ii,j,2);
        hsv(ii,j,1) =tone* hsv(ii,j,1);
    end
end
RGB1 = hsv2rgb(hsv);
end