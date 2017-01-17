clc;
clear;
close all;

addpath( genpath ('functions/') );

%---------------------------------------------------
% Load an image
% (Must be an 8 bit color or grayscale image)
% Recommended: Image size integer multiple of block_size [64 64]
%              (However, program will run 
%               irrespective of Image size, by 
%               cropping the image region which is
%               multiple of block_size)
%---------------------------------------------------
img = imread('data/monarch.png');

%---------------------------------------------------
% Calculating the contrast detection thresholds
%---------------------------------------------------
[contrast_detection_threshold, QP] = CNNTHR( img );