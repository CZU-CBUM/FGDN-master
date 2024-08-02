function [BrainNetSet]=PC(BOLD,lambda)
%% Basic parameter 
nTime=size(BOLD{1},1);
nSubj=length(BOLD);  
RegionNum=size(BOLD{1},2);
Total_Data=zeros(nSubj,nTime,RegionNum);
for SubjectID=1:nSubj 
    tmp=BOLD{SubjectID};
    subject=tmp(:,1:RegionNum); 
    subject=subject-repmat(mean(subject),nTime,1); 
    subject=subject./(repmat(std(subject),nTime,1)); 
    Total_Data(SubjectID,:,:)=subject; 
end
%% Network construction
BrainNetSet=cell(length(lambda),1);
for SubjectID=1:nSubj
    for l1=1:size(lambda,2)
        param=lambda(l1);
        BrainNet=zeros(RegionNum,RegionNum);
        tmp=zeros(size(Total_Data,2),size(Total_Data,3));
        tmp(:,:)=Total_Data(SubjectID,:,:);
        currentNet=corrcoef(tmp); 
        currentNet=currentNet-diag(diag(currentNet));
        threhold=prctile(abs(currentNet(:)),param); 
        currentNet(abs(currentNet)<=threhold)=0;
        BrainNet(:,:)=currentNet;
        
        BrainNetSet{l1,1}(SubjectID,:,:)=BrainNet;
        fprintf('Done the %d subject networks with lamda1 equal to %d!\n',SubjectID,l1);
    end
end
save('data\BrainNetSet_HC_SZ_PC.mat','BrainNetSet');

%在数据预处理步骤中，代码首先从变量 BOLD 中读取 fMRI 数据。nTime 代表时间点数目，nSubj 代表受试者数目，RegionNum 代表脑区域数目。
%Total_Data 是一个三维数组，它保存了每个受试者的 fMRI 数据，其中第一维表示受试者编号，第二维表示时间点编号，第三维表示脑区域编号。
%在构建网络步骤中，代码首先遍历所有的受试者和所有的 $\lambda$ 值，对于每个受试者和每个 $\lambda$ 值，它会进行以下操作：
%从 Total_Data 中获取当前受试者的 fMRI 数据，并进行标准化处理，即减去均值并除以标准差。
%计算脑区域之间的相关系数，得到一个 RegionNum × RegionNum 的矩阵 currentNet。
%将 currentNet 中的绝对值小于当前 $\lambda$ 值的元素置为零，得到一个阈值化的相关系数矩阵 BrainNet。
%将 BrainNet 保存到 BrainNetSet 中，其中第一维表示 $\lambda$ 值编号，第二维表示受试者编号，第三维表示脑区域之间的连接。
%最后，代码将 BrainNetSet 保存到 data\BrainNetSet_HC_SZ_PC.mat 文件中。

%代码中的 corrcoef() 函数计算的是皮尔森相关系数。该函数可以计算矩阵中任意两个行之间的皮尔森相关系数。
%这里将 BOLD 信号矩阵作为输入，计算出每个脑区之间的相关性系数，得到一个相关性矩阵。
%通过对相关性矩阵进行阈值处理，可以得到一个稀疏的脑网络，即 BrainNet。