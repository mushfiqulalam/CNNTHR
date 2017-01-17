function [dd,dd1]=convol(a,b,ba,i,net)

%CONVOL N-dimensional convolution
%
%	Syntax
%
%	  [dd,dd1]=convol(a,b,ba,i,net,q)
%
%	Description
%
%              performs the N-dimensional convolution of matrices
%
%	[dd,dd1]=convol(a,b,ba,i,net,q) takes these inputs,
%
%                       a and b are input on which convolution is performed.
%                       ba is a flag i.e. when ba=0 i.e. forward calculation
%                                         when ba=1 i.e. backward calcultion(weight update)
%                                         when ba=2 i.e. sensitivity calcution i.e. dV calculation
%                       i is layer number
%                       net is a network given with parameters
%
%	  and returns,
%                       dd - output of convolution operation in forwaard
%                       calculation or weight update value in backward
%                       calculation
%
%                       dd1 - used to calculate dV in backward calculation
%
%
%	  CONVOL returns useful information about convolution operation
%
%
%	Examples
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
%	    % Create and Train a Network
%	    net = newcnn;
%       [net1,tr]=imtrain(net,p,t,p1,t1,p2,t2);
%       z(2,:)=convol(net.w(2,:,:),p(1,:),0,1,net);
%
%   Here we assign this transfer function to layer i of a network.
%
%     net.wf{i} = 'convol';
%
%
%	Algorithm
%
%   It basically uses convn for performing convolution. It is used during
%   forward caculation and for updating weights by using backpropagation
%   algorithm
%   CONVN  N-dimensional convolution.
%   C = CONVN(A, B) performs the N-dimensional convolution of
%   matrices A and B. If nak = size(A,k) and nbk = size(B,k), then
%   size(C,k) = max([nak+nbk-1,nak,nbk]);
%
%   C = CONVN(A, B, 'shape') controls the size of the answer C:
%     'full'   - (default) returns the full N-D convolution
%     'same'   - returns the central part of the convolution that
%                is the same size as A.
%     'valid'  - returns only the part of the result that can be
%                computed without assuming zero-padded arrays.
%                size(C,k) = max([nak-max(0,nbk-1)],0).
%
%   Class support for inputs A,B:
%      float: double, single
%
%   This code is part of Flexible Image Recognition Software
%   Toolbox, by Pranita Patil, and Martin T. Hagan, 
%   https://shareok.org/bitstream/handle/11244/11177/Patil_okstate_0664M_12914.pdf?sequence=1


if(ba==0)
    % forward calcuation i.e. to calculate weight function output
    dd=num2cell(zeros(1,size(net.w,2)));
%     tic
%     i
    for h=1:net.featureMap(i)
        % output dimensions after valid convolution operation
        Sr=size(b{1},1)-size(a{1,1,1},1)+1;
        Sc=size(b{1},2)-size(a{1,1,1},2)+1;
        c=zeros(Sr,Sc);
        for ii=1:net.featureMap(i-1)
            if(net.ConnectionMatrix{i}(ii,h)==1)
                %convn rotates w (i.e. a in this case) before the
                %convolution, which causes another rotation in the 
                %forward calculation so in thesis, while writing 
                %equation this rotation is not mentioned.
                aa=convn(b{1,ii},rot90(a{1,h,ii},2),'valid');
                c=c+aa;
            end
        end
        dd{1,h}=c;
    end
%     toc
else
    % backward calcultion to update weights of convolution layer
    dd=net.w(i,:,:);
    for h=1:net.featureMap(i)
        for ii=1:net.featureMap(i-1)
            if(net.ConnectionMatrix{i}(ii,h)==1)
                %convn rotates dZ (i.e. a in this case) before the 
                %convolution, which causes another rotation in the 
                %backpropagation calculation so in thesis, while writing 
                %equation this rotation is not mentioned.
                aa=convn(b{1,ii},rot90(a{1,h},2),'valid');
                dd{1,h,ii}=aa;
            end
        end
    end
    % dV calculation
    dd1=num2cell(zeros(1,size(net.w,2)));
    for h=1:net.featureMap(i-1)
        % output dimensions after normal convolution operation
        Sr=size(a{1},1)+size(net.w{i,1,1},1)-1;
        Sc=size(a{1},2)+size(net.w{i,1,1},2)-1;
        c=zeros(Sr,Sc);
        for ii=1:net.featureMap(i)
            if(net.ConnectionMatrix{i}(h,ii)==1)
                aa1=convn(a{1,ii},net.w{i,ii,h},'full');
                c=c+aa1;
            end
        end
        dd1{1,h}=c;
    end
end



end


