function [c,c1]=fullconnection(a,b,ba,i,net)

%FULLCONNECTION dot product operation
%
%	Syntax
%
%	  [c,c1]=fullconnection(a,b,ba,i,net)
%
%	Description
%
%              This function performs dot product operation
%
%	[c,c1]=fullconnection(a,b,ba,i,net,q) takes these inputs,
%
%                       a and b are input on which dot product is performed.
%                       ba is a flag i.e. when ba=0 i.e. forward calculation
%                                         when ba=1 i.e. backward calcultion(weight update)
%                                         when ba=2 i.e. sensitivity calcution i.e. dV calculation
%                       i is layer number
%                       net is a network given with parameters
%                       
%	  and returns,
%                       
%                       c - output of dot product operation in forwaard
%                       calculation or weight update value in backward
%                       calculation
%
%                       c1- used to calculate dV in backward calculation
%
%
%	  FULLCONNECTION returns useful information about subsampling operation
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
%	    % Create and to perform full connection operation
%	    net = newcnn;
%      [net1,tr]=imtrain(net,p,t,p1,t1,p2,t2);
%      z(6,:)=fullconnection(net.w(6,:,:),v(5,:),0,1,net);
%
%   Here we assign this transfer function to layer i of a network.
%
%     net.wf{i} = 'fullconnection';
%
%
%	Algorithm
%
%   It is basically used to perform dot product operation i.e. full connection
%   layer. It can be multilayer neural network which is considered as full
%   connection layer.
%
%   This code is part of Flexible Image Recognition Software
%   Toolbox, by Pranita Patil, and Martin T. Hagan, 
%   https://shareok.org/bitstream/handle/11244/11177/Patil_okstate_0664M_12914.pdf?sequence=1

%initialization of variable
z1=zeros(size(net.b{i},1),1);


if(ba==1)
    % backward calculation to update weights
    c=net.w(i,:,:);
    for h=1:net.featureMap(i)
        b1{h}=reshape(b{h}',[1 size(b{h},1)*size(b{h},2)]);
        cc=a{1}*b1{h};
        c{1,1,h}=cc;
    end
    
        % dV calculation to calculate sensitivity
    c1=num2cell(zeros(1,size(net.w,2)));
    if(i==net.numLayers)
        % if last layer is full connection then form of output is
        % vector.
        for h=1:net.featureMap(i)
            c1{h}=net.w{i,1,h}'*a{1};
        end
    else
        % if layer is full connection other than last layer then
        % vector output is divided into FMs of next layer
        for h=1:net.featureMap(i+1)
            c1{h}=net.w{i,1,h}'*a{1};
        end
    end
elseif(ba==0)
    % forward calculation to calculate weight function output
    c=num2cell(zeros(1,size(net.w,2)));
    for h=1:net.featureMap(i)
        % different weights for previous FMs are added in this layer to
        % form single vector output.
        %--mushfiq debug----%
%         size(a{1,1,h}) 
%         size(b{h})
        %-----------%
        cc=a{1,1,h}*b{h};
        z1=z1+cc;
        c{1}=z1;
    end
    c1=0;
end


end