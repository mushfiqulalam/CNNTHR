function [n,dZ]=mplus(b,z,ba,i,net)

%MPLUS to sum excitations of layers.
%
% Syntax
%
%   [n,dZ]=mplus(b,z,ba,i,net)
%
% Description
%
%   MPLUS is a function used to perform plus operation. It calculates net
%   input function output.
%
%	[n,dZ]=mplus(b,z,ba,i,net,q) takes these inputs,
%                           b is bias in forward whereas dN in backward
%                           z is output of weight function in forward
%                           whereas 0 in backward
%                           ba is a flag (future use)
%                           i is layer number
%                           net is a network given with parameters
%
%	  and returns,
%
%                       n - output of net input function in forwaard
%                       calculation or bias update value in backward
%                       calculation
%
%                       dZ - used to calculate gradient in backward calculation
%
%
%	  MPLUS returns useful information about net input function
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
%       [net1,tr]=imtrain(net,p,t,p1,t1,p2,t2);
%      n(2,:)=mplus(net.b(2,:),z(2,:),0,2,net);
%
%   Here we assign this transfer function to layer i of a network.
%
%     net.nf{i} = 'mplus';
%
% Algorithm
%
%     n = mplus(b,z) = b+z
%
%   where z is weight function output.i.e. z=wp
%
%   This code is part of Flexible Image Recognition Software
%   Toolbox, by Pranita Patil, and Martin T. Hagan, 
%   https://shareok.org/bitstream/handle/11244/11177/Patil_okstate_0664M_12914.pdf?sequence=1

n=num2cell(zeros(1,size(net.w,2)));
if(ba==0)
    if(strcmp(net.wf{i},'fullconnection')==1)
        % for full connection layer, bias is a vector
        n{1}=b{1}+z{1};
        
    else
        for h=1:net.featureMap(i)
            
            n{h}=b{h}+z{h};
            
        end
    end
else
    n=num2cell(zeros(1,size(net.b,2)));
    for h=1:net.featureMap(i)
        if(i==net.numLayers)
            if(h==1)
                n{h}=b{1};
            else
                n{h}=[];
                
            end
            
        else
            del=cell2mat(b(1,h));
            n{h}=sum(sum(del));
            
        end
    end
    dZ=b;
end

end