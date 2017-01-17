function [c,c1]=subsampling(b,a,ba,i,net)

%SUBSAMPLING subsampling operation
%
%	Syntax
%
%	  [c,c1]=subsampling(b,a,ba,i,net)
%
%	Description
%
%              This function performs reduction operation on matrix
%
%	[c,c1]=subsampling(b,a,ba,i,net) takes these inputs,
%
%                          a and b are input on which operation is performed.
%                          ba is a flag i.e. when ba=0 i.e. forward calculation
%                                            when ba=1 i.e. backward calcultion(weight update)
%                                            when ba=2 i.e. sensitivity calcution i.e. dV calculation
%                          i is layer number
%                          net is a network given with parameters
%
%	  and returns,
%
%                       c - output of subsampling operation in forwaard
%                       calculation or weight update value in backward
%                       calculation
%
%                       c1 - used to calculate dV in backward calculation
%
%
%	  SUBSAMPLING returns useful information about subsampling operation
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
%      z(3,:)=subsampling(net.w(3,:,:),v(2,:),0,1,net);
%
%   Here we assign this transfer function to layer i of a network.
%
%     net.wf{i} = 'subsampling';
%
%
%	Algorithm
%
%   It basically used to reduce dimensions of matrix to reduce resolution.
%   This subsampling factor for row and column is given as net.subfactor_r
%   and subfactor_c respectively. For forward calculation reduction is
%   performed so while going backward i.e. during backpropagation,
%   expansion is performed.
%
%   This code is part of Flexible Image Recognition Software
%   Toolbox, by Pranita Patil, and Martin T. Hagan, 
%   https://shareok.org/bitstream/handle/11244/11177/Patil_okstate_0664M_12914.pdf?sequence=1


if (ba==1)
    c=net.w(i,:,:);
    
    for h=1:net.featureMap(i)
        
%         size(a{1,h},1)/net.subfactor_r(i)
%         size(a{1,h},2)/net.subfactor_c(i)
%         z3=zeros(size(a{1,h},1)/net.subfactor_r(i),size(a{1,h},2)/net.subfactor_c(i)); % changed by mushfiq
        z3=zeros(size(a{1,h},1)/net.subfactor_r(i),size(a{1,h},2)/net.subfactor_c(i));
        
        aa=a{1,h};
        for k=1:net.subfactor_r(i)
            for l=1:net.subfactor_c(i)
                % reduction in matrix by factor
                z3 = z3+aa(1+(k-1):net.subfactor_r(i):size(aa,1),1+(l-1):net.subfactor_c(i):size(aa,2));
            end
        end
        % as subsampling weights are 1 by 1 and while propagating back
        % to update weights, matrices are converted to vectors before
        % calculating dw
        b1{1,h}=reshape(b{1,h}',[1 size(b{1,h},1)*size(b{1,h},2)]);
        
        z3=reshape(z3',[size(z3,1)*size(z3,2) 1]);
        
        % weights are updating
        for ii=1:net.featureMap(i-1)
            if(net.ConnectionMatrix{i}(ii,h)==1)
                cc=b1{1,h}*z3;
                c{1,h,ii}=cc;
            end
        end
    end
    
    c1=num2cell(zeros(1,size(net.w,2)));
    for h=1:net.featureMap(i-1)
        for k=1:net.subfactor_r(i)
            for l=1:net.subfactor_c(i)
                % while propagating backwards through subsampling layer expansion is done
                uu(1+(k-1):net.subfactor_r(i):size(b{1,h},1)*net.subfactor_r(i),1+(l-1):net.subfactor_c(i):size(b{1,h},2)*net.subfactor_c(i)) =b{1,h} ;
                
            end
            
        end
        for ii=1:net.featureMap(i)
            if(net.ConnectionMatrix{i}(h,ii)==1)
                c1{h}=net.w{i,ii,h}*uu;
            end
        end
    end
    
    
else
    c=num2cell(zeros(1,size(net.w,2)));
    % forward calculation to obtain weight function output
    for h=1:net.featureMap(i)
%         size(a{1,h},1)/net.subfactor_r(i) % debug mushfiq
%         size(a{1,h},2)/net.subfactor_c(i) % debug mushfiq
%         z3=zeros(ceil(size(a{1,h},1)/net.subfactor_r(i)), ceil(size(a{1,h},2)/net.subfactor_c(i))); % Changed by mushfiq
        z3=zeros(size(a{1,h},1)/net.subfactor_r(i),size(a{1,h},2)/net.subfactor_c(i)); % original
        
        aa=a{1,h};
        for k=1:net.subfactor_r(i)
            for l=1:net.subfactor_c(i)
                % matrix is reduced by factor red
%                 size(z3) % debug mushfiq
%                 size(aa(1+(k-1):net.subfactor_r(i):size(aa,1),1+(l-1):net.subfactor_c(i):size(aa,2))) % debug mushfiq
                z3 = z3+aa(1+(k-1):net.subfactor_r(i):size(aa,1),1+(l-1):net.subfactor_c(i):size(aa,2));
            end
        end
        for ii=1:net.featureMap(i-1)
            if(net.ConnectionMatrix{i}(ii,h)==1)
                % calculating weight function output
                cc=b{1,h,ii}*z3;
                c{1,h}=cc;
            end
        end
    end
    
    c1=0;
end


end