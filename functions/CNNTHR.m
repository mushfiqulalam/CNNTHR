function [contrast_detection_threshold, QP] = CNNTHR( img )
% %------------------------------------------------------------------------
% % function CNNTHR (Contrast Detection Threshold via Convolutional-
% % Neural-Network). Generates the distortion-visibility
% % (contrast-detection-thresholds) of a distortion
% % when overlayed on the input image
% % 
% % Input: img: RGB/Grayscale 8 bit image
% %        
% % Output:
% %        contrast_detection_threshold: contrast detection thresholds (dB)
% %        QP: HEVC Quantization Parameter 
% %  
% % Questions?Bugs?
% % Please contact by: Mushfiqul Alam
% %                    mdma@okstate.edu
% %
% % If you use the codes, cite the following works:
% %     (1) Alam, M. M., Nguyen, T., and Chandler, D. M., 
% %     "A perceptual strategy for HEVC based on a convolutional neural 
% %     network trained on natural videos," SPIE Applications of Digital 
% %     Image Processing XXXVIII, August 2015. Doi: 10.1117/12.2188913.  
% %     (2) Alam, M. M., Patil, P., Hagan, M. T., and Chandler, D. M., 
% %     "A computational model for predicting local distortion visibility 
% %      via convolutional neural network trainedon natural scenes," IEEE 
% %      International Conference on Image Processing, 2015. pp. 3967-3971.
% %------------------------------------------------------------------------

%---------------------------------------------------
% If no input found; return 
%---------------------------------------------------
if (nargin == 0)
    disp('Input image required. Ending program.');
    return;
end

%---------------------------------------------------
% Check if the input image is valid
%---------------------------------------------------
if (ndims(img) == 1)
    disp('Valid image input required. Ending program.');
end

%---------------------------------------------------
% Block size is default and must be [64 64]
%---------------------------------------------------
% default block size
block_size(1) = 64;
block_size(2) = 64;

%---------------------------------------------------
% Normalizing and rgb2gray conversion if needed
%---------------------------------------------------
if (ndims(img) == 3)
    img = rgb2gray(img);
end
img = double( img );
img = 2 * img ./ 255 - 1; % -1 to +1

%---------------------------------------------------
% If the image size is not integer multiple of
% block_size, crop the image so that the cropped
% region is integer multiple of block_size
%---------------------------------------------------
if ( ( rem(size(img, 1), block_size(1)) ~= 0 ) || ( rem(size(img, 1), block_size(2)) ~= 0 ) )
    disp('Warning! Image size is not interger multiple of block size');
    disp('Proceeding with CROPPED image, making image size integer multiple of block size.');
    img = img( 1 : floor(size(img, 1)/block_size(1))*block_size(1), ...
               1 : floor(size(img, 2)/block_size(2))*block_size(2) );
end

blocks = blocking_function( img, block_size, 0 );
[n_height_blocks, n_width_blocks] = size(blocks);
no_of_blocks = n_width_blocks*n_height_blocks;
blocks = blocks(:); % vectorizing the blocks

%---------------------------------------------------
% Show the MASKING MAP/DISTORTION VISIBILITY MAP
%---------------------------------------------------
load( 'ct_trained_nets.mat' );
no_of_trained_nets = length(nets);
temp = zeros(no_of_blocks, no_of_trained_nets);
for net_idx = 1 : no_of_trained_nets
    temp(:, net_idx) = ( cell2mat( testcnn_m( blocks, nets{net_idx}) ) );    
end
temp = ( ( info.ct_max - info.ct_min ) * ( ( temp + 1 ) ./ 2 ) ) + info.ct_min;
contrast_detection_threshold = mean( temp, 2 );
contrast_detection_threshold = reshape( contrast_detection_threshold, [n_height_blocks n_width_blocks] );

%---------------------------------------------------
% Show the MASKING MAP/DISTORTION VISIBILITY MAP
%---------------------------------------------------
distortion_visibility_map = imresize(contrast_detection_threshold, [size(img, 1) size(img, 2)], 'nearest');
figure('Name', 'CRMS Threshold Map (dB)');
imshow(distortion_visibility_map, []);
title('CRMS Threshold Map (dB)');
colormap jet;colorbar;


%---------------------------------------------------
% CONTRAST DETECTION THRESHOLD TO QUANTIZATION
% PARAMETER
%---------------------------------------------------

% Load the trained network for predicting Quantization Thresholds
% from Contrast Detection Thresholds
load( 'ct_to_qp.mat' );

feat1 = zeros(no_of_blocks, 1);
feat2 = zeros(no_of_blocks, 1);
feat3 = zeros(no_of_blocks, 1);
feat4 = zeros(no_of_blocks, 1);

for block_idx = 1 : length(blocks)
    
    blk = 255 * ( blocks{block_idx} + 1 ) / 2;
    blk_lum = ( display.b + display.k * blk ) .^ display.gamma; % converting to luminance

    % Michelson Contrast
    feat1(block_idx)  = ( max(blk_lum(:)) - min(blk_lum(:)) ) ./ ( max(blk_lum(:)) + min(blk_lum(:)) );
    % log10(RMS contrast)
    feat2(block_idx)    = log10((max(blk_lum(:)) - min(blk_lum(:)))./mean(blk_lum(:)));
    % std
    feat3(block_idx)    = std(blk(:));
    % slope of magnitude spctra 
    temp = blk_amp_spec_slope_eo_toy( blk );
    feat4(block_idx)   = temp(2);
    
end



x = [feat1 feat2 feat3 feat4]';

for net_idx = 1 : length(net_a)
    
    net = net_a{net_idx};
    temp = net(x);
    alpha(:, net_idx) = temp(:);
    
    net = net_b{net_idx};
    temp = net(x);
    beta(:, net_idx)  = temp(:);
    
    net = net_c{net_idx};
    temp = net(x);
    gamma(:, net_idx) = temp(:);
    
end

alpha = reshape( mean(alpha, 2), [n_height_blocks n_width_blocks] );
beta  = reshape( mean(beta, 2), [n_height_blocks n_width_blocks] );
gamma = reshape( mean(gamma, 2), [n_height_blocks n_width_blocks] );

log_Q_step = alpha .* contrast_detection_threshold.^2 +...
             beta  .* contrast_detection_threshold +...
             gamma;
         
% Quantization Parameter (QP, range: 0~51)         
QP = round( log_Q_step / log((2^(1/6))) + 4 );
QP(QP < 0)  = 0;
QP(QP > 51) = 51;

%---------------------------------------------------
% Show the Quantization Parameter
%---------------------------------------------------
QP_map = imresize(QP, [size(img, 1) size(img, 2)], 'nearest');
figure('Name', 'HEVC - Quantization Parameter Map');
imshow(QP_map, []);
title('HEVC - Quantization Parameter Map');
colormap jet;colorbar;
