function [out]=testcnn_m(p1, net)
% function [confusionmatrix_test,accuracy_test]=testcnn(p1,t1,net,train_labels1,siz)
% TESTCNN to caculate confusionmatrix and accuracy for training
%
%	Syntax
%
%	  [confusionmatrix_test,accuracy_test]=testcnn(p1,t1,net)
%
%	Description
%
%               This testcnn function is used to calculate testing accuracy and confusion
%               matrix for testing data set.
%
%	[confusionmatrix_test,accuracy_test]=testcnn(p1,t1,net) takes these inputs,
%
%                                   net- network with updated values of weights and biases
%                                   p1 and t1- testing data  and target
%                                   values for testing data
%
%	  and returns,
%                                   confusionmatrix_test- confusion matrix
%                                   accuracy_test- accuracy calculated from confusion matrix
%
%
%	  TESTCNN returns useful information about calculation of confusion matrix
%	  and accuracy for testing.
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
%	    % Create and Calculate confusion matrix and accuracy
%	    net = newcnn;
%       [net1,tr]=imtrain(net,p,t,p1,t1,p2,t2);
%       [confusionmatrix_test,accuracy_test]=testcnn(p1,t1,net)
%
%
%
%
%	Algorithm
%
%   It calculates confusion matrix and accuracy for testing data. It
%   performs forward calculation to calculate result for tesing data by
%   using updated trainable parameters. For manually derived target
%   values,train_labels1 is used which contains original form of target
%   values. By performing forward calculation, it calculates
%   output from last layer and compared it with target value and if it is
%   true it is stored in diagonal element of confusion matrix depending on
%   type of pattern and if it is not true then it is stored in non-diagonal
%   element of confusion matrix depending on original pattern and estimated
%   pattern.
%
%   This code is part of Flexible Image Recognition Software
%   Toolbox, by Pranita Patil, and Martin T. Hagan, 
%   https://shareok.org/bitstream/handle/11244/11177/Patil_okstate_0664M_12914.pdf?sequence=1


testPatterns=size(p1,1);
% confusionmatrix_test=zeros(net.numpatterns,net.numpatterns);

% if(size(t1{1,1},1)==size(train_labels1(:,1),1)||train_labels1==0)
% if target values are not chosen manually and it is using 1 of c
% technique
out = cell(testPatterns, 1);
for q=1:testPatterns
    z=num2cell(zeros(1,size(net.w,2)));
    n=num2cell(zeros(1,size(net.w,2)));
    a=num2cell(zeros(1,size(net.w,2)));
    v=num2cell(zeros(1,size(net.w,2)));
    % calculate output by using updated weight values and compared with
    % target in order to calculate confusion matrix
    for i=2:net.numLayers
        
        ba=0;
        if(i==2)
            % transform function output calculation for 2nd layer i.e. layer after
            % input data is given
            v(i,:)=feval(net.vf{i},p1(q,:),ba,i,net);
            
        else
            % transform function output calculated for layers other than 2nd
            % calculated
            v(i,:)=feval(net.vf{i},a(i-1,:),ba,i,net);
            
        end
        % weight function output calculated
        z(i,:)=feval(net.wf{i},net.w(i,:,:),v(i,:),ba,i,net);
        % calculated net input function output
        n(i,:)=feval(net.nf{i},net.b(i,:),z(i,:),ba,i,net);
        % calculated transfer function output
        a(i,:)=feval(net.tf{i},n(i,:),ba,i,net);
        
    end
    
    out{q} = a{net.numLayers};
    
%     if(strcmp(net.tf{net.numLayers},'imagetansig')==1)
%         com=hardlims(a{net.numLayers,1});
%         com1=ceil(a{net.numLayers,1});
%     elseif(strcmp(net.tf{net.numLayers},'imagelogsig')==1)
%         com=hardlim(a{net.numLayers,1});
%         com1=round(a{net.numLayers,1});
%     end
%     if((1/0.76)*t1{:,q}==com)
%         % calculating confusion matrix
%         k=find((1/0.76)*t1{:,q}==1);
%         confusionmatrix_test(k,k)=confusionmatrix_test(k,k)+1;
%     else
%         if(size(find(com1==1),1)==1)
%             k1=find((1/0.76)*t1{:,q}==1);
%             k=find(com1==1);
%             confusionmatrix_test(k1,k)=confusionmatrix_test(k1,k)+1;
%         end
%     end
end
% accuracy calculation for testing data
% accuracy_test=trace(confusionmatrix_test)/size(p1,1)*100;
% else
%     % if target values are chosen manually
%     for i=1:testPatterns
%         train_labels1(:,i)=train_labels1(:,siz+i);
%     end
%     for q=1:testPatterns
%         z=num2cell(zeros(1,size(net.w,2)));
%         n=num2cell(zeros(1,size(net.w,2)));
%         a=num2cell(zeros(1,size(net.w,2)));
%         v=num2cell(zeros(1,size(net.w,2)));
%         for i=2:net.numLayers
%
%         ba=0;
%         if(i==2)
%             % transform function output calculation for 2nd layer i.e. layer after
%             % input data is given
%             v(i,:)=feval(net.vf{i},p1(q,:),ba,i,net);
%
%         else
%             % transform function output calculated for layers other than 2nd
%             % calculated
%             v(i,:)=feval(net.vf{i},a(i-1,:),ba,i,net);
%
%         end
%         % weight function output calculated
%         z(i,:)=feval(net.wf{i},net.w(i,:,:),v(i,:),ba,i,net);
%         % calculated net input function output
%         n(i,:)=feval(net.nf{i},net.b(i,:),z(i,:),ba,i,net);
%         % calculated transfer function output
%         a(i,:)=feval(net.tf{i},n(i,:),ba,i,net);
%
%         end
%         if((1/0.76)*t1{:,q}==hardlims(a{net.numLayers,1}))
%             % comparing output with target values
%             k=find((1/0.76)*train_labels1(:,q)==1);
%             confusionmatrix_test(k,k)=confusionmatrix_test(k,k)+1;
%         else
%             for q1=1:testPatterns
%                 if((1/0.76)*t1{:,q1}==hardlims(a{net.numLayers,1}))
%                     % train_label1 is according to 1 of c technique so it
%                     % is used to find exact pattern identity
%                     k2=q1;
%                     k=find((1/0.76)*train_labels1(:,k2)==1);
%                     k1=find((1/0.76)*train_labels1(:,q)==1);
%                     confusionmatrix_test(k1,k)=confusionmatrix_test(k1,k)+1;
%
%                 end
%             end
%
%         end
%
%     end
%     % accuracy calculation for testing data
%     accuracy_test=trace(confusionmatrix_test)/size(p1,1)*100;
end