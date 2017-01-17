function a=imagetansig(n,ba,i,net,dA,varargin)

%IMAGETANSIG Hyperbolic tangent sigmoid transfer function.
%
% Syntax
%
%   a=imagetansig(n,ba,i,net,dA)
%
% Description
%
%   IMAGETANSIG is a neural transfer function.  Transfer functions
%   calculate a layer's output from its net input.
%
%	a=imagetansig(n,ba,i,net,dA) takes these inputs,
%
%                               n is input on which operation is performed.
%                               ba is a flag i.e. when ba=0 i.e. forward calculation
%                                                 when ba=1 i.e. backward calcultion
%                               i is layer number
%                               net is a network given with parameters
%                               dA is zero for forward calculation but for
%                               backward calculation, it is gradient value
%	  and returns,
%                       a - output of transfer function
%
%
%	  IMAGETANSIG returns useful information about transfer function
%	  operation
%
% Examples
%
%	  Here is a problem consisting of inputs p and targets t that we would
%	  like to solve with a network.
%
%     [p,t,p1,t1,p2,t2]=datag;
%
%	  Here a LeNet-1 network is created for 2D image data.  The network's
%	  input is two dimensional image data.  The TRAINRP network training
%     function is to be used.
%
%	    % Create and forward calculation with ba=0
%	    net = newcnn;
%        [net1,tr]=imtrain(net,p,t,p1,t1,p2,t2);
%      a(3,:) = imagetansig(n(3,:),0,0,3,net,0)
%
%   Here we assign this transfer function to layer i of a network.
%
%     net.tf{i} = 'imagetansig';
%
% Algorithm
%
%       Here ba is used to indicate wheather derivative is calculated or not
%       ba=0
%     a = imagetansig(n,d) = 2/(1+exp(-2*n))-1
%
%       Its derivative is given as i.e. ba=1
%
%     imagelogsig(n,d) = n .* (1-n)
%
%   This is mathematically equivalent to TANH(N).  It differs
%   in that it runs faster than the MATLAB implementation of TANH,
%   but the results can have very small numerical differences.  This
%   function is a good trade off for neural networks, where speed is
%   important and the exact shape of the transfer function is not.
%   It calculates element wise matrix multiplication for dV calculation in
%   backpropagation algorithm. Also when full connection is not layer then
%   vector is divided into FMs which is given as input to next layer.This
%   is done in forward calculation. So while propagating back, FMs are
%   again converted into vector and this is done in this function.
%
%   This code is part of Flexible Image Recognition Software
%   Toolbox, by Pranita Patil, and Martin T. Hagan, 
%   https://shareok.org/bitstream/handle/11244/11177/Patil_okstate_0664M_12914.pdf?sequence=1

% for forward calculation
if (nargin==4)
    dA=0;
end
a=num2cell(zeros(1,size(net.w,2)));

if(ba==1)
    % to calculate derivative
    if(i==net.numLayers && strcmp(net.wf{i},'fullconnection')==1)
        % if full connectiona layer then vector
        a1{1}=1-n{1}.^2;
        a{1}=a1{1}.*dA{1};
        
        % for converting FMs into vector if full connection layer is not last layer
        % for backward calculation
    elseif(i~=net.numLayers && strcmp(net.wf{i},'fullconnection')==1)
        for h=1:net.featureMap(i+1)
            a1{h}=1-n{h}.^2;
            a{h}=a1{h}.*dA{h};
        end
        k=0;
        no=(size(a{1},1)*size(a{1},2));
        for jj=1:net.featureMap(i+1)
            t=1;
            for ii=1+k:no+k
                a{1}(ii)=a{jj}(t);
                k=k+1;
                t=t+1;
            end
        end
        a{1}=a{1}';
    else
        for h=1:net.featureMap(i)
            a1{h}=1-n{h}.^2;
            a{h}=a1{h}.*dA{h};
        end
    end
elseif(ba==0)
    % to calculate transfer function output
    if(i==net.numLayers && strcmp(net.wf{i},'fullconnection')==1)
        % if full connectiona layer then vector
        a{1}=2./(1+exp(-2*n{1}))-1;
    else
        for h=1:net.featureMap(i)
            a{h}=2./(1+exp(-2*n{h}))-1;
        end
    end
    
    % if it is not last layer and it is full connection that means trasnfer
    % function should be divided into FMs of next layer. Depending on
    % number of FMs in next layer and number of elements in vector of
    % present layer, no i.e. number of elements in one FM can be
    % calculated.
    if(i~=net.numLayers && strcmp(net.wf{i},'fullconnection')==1 )
        no=size(a{1},1)/net.featureMap(i+1);
        k=0;
        cc=cell(1,size(net.w,2));
        % place elements of vector into FMs
        for jj=1:net.featureMap(i+1)
            t=1;
            for ii=1+k:no+k
                cc{jj}(t)=a{1}(ii);
                t=t+1;
                k=k+1;
            end
        end
        a=cell(1,size(net.w,2));
        % reshape elements of FMs into matrix according to value
        % calculated by no variable
        for e=1:net.featureMap(i+1)
            a{e}=reshape(cc{e},ceil(no/2),ceil(no/2))';
        end
    end
    
end

end