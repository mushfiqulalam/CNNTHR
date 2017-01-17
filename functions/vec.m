function v=vec(a,ba,i,net,q,varargin)

%VEC for conversion - transformation function
%
% Syntax
%
%   v=vec(a,ba,i,net,q)
%
% Description
%
%           vec is a function in which marix to vector
%           conversions takes place
%
%	v=transform(a,ba,i,net,q) takes these inputs,
%                                a -  input on which coversion takes place.
%                                ba is a flag i.e. when ba=0 i.e. forward calculation
%                                                  when ba=1 i.e. backward
%                                                  calcultion
%                               i is layer number
%                               net is a network given with parameters
%                               q is a output of network which is required
%                               to calculate dimensions of matrix which is
%                               converted into vector during forward
%                               calculation, so mtarix dimensions are used 
%                               in backpropagation in order to convert
%                               vector back to matrix. Whereas for one
%                               dimensional, this transformation is not 
%                               necessary for full connection when ba=1,
%                               so it will remain as in the same form.
%	  and returns,
%                       v - output of transform function
%
%
%	  VEC returns useful information about transform function
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
%       v(5,:)=vec(a(5,:),0,5,net);
%
%   Here we assign this transform function to layer i of a network.
%
%     net.vf{i} = 'vec';
%
% Algorithm
%
%   This function uses reshape function to convert matrix into vector.
%   For backward calculation, this function is used to take
%   transpose of vector.
%
%   This code is part of Flexible Image Recognition Software
%   Toolbox, by Pranita Patil, and Martin T. Hagan, 
%   https://shareok.org/bitstream/handle/11244/11177/Patil_okstate_0664M_12914.pdf?sequence=1

% for forward calculation
if (nargin==4)
    q=0;
end

% row and column dimensions
q1=size(q,1);
q2=size(q,2);

% intialization of variables
v=num2cell(zeros(1,size(net.w,2)));

if(strcmp(net.wf{i},'fullconnection')==1)
    % this transformation is only for full connection layer as full
    % connection requires only vector input.
    if(ba==0)
        % vector transformation for forward calculation
        for h=1:net.featureMap(i)
            v{h}=reshape(a{h}',[size(a{h},1)*size(a{h},2) 1]);
        end
    elseif(ba==1)
        % transformation for backward calculation
        for h=1:net.featureMap(i)
            if(q1~=1 && q2~=1)
                % matrix transformation for dA calculation for 2D
                v{h}=reshape(a{h},[q1 q1])';
            else
                % trasformation for dA calculation for 1D
                v{h}=a{h};
            end
        end
    end
else
    % layers other than full connection, no conversion will take place.
    for h=1:net.featureMap(i-1)
        v{h}=a{h};
    end
end

end