   clc;
   clear;
   dataset_names = ["Yale" ];
   ds=1;
    dataName = dataset_names{ds};   
    fprintf('\n Dataset:%s \n',dataName);
    data = dataName;
    load("datasets/"+data)
    %% set random observed ratio
    MR=0.1;
    V=length(X);
    
                 for v=1:V
            X{v}=NormalizeFea(X{v},0);
             [Dv,N]=size(X{v});
                 end
             
                 
           Omega=zeros(N,1);
            enList =[];
            % n_p=0.45;
            for v = 1:V-1
                ind_folds(:,v)=ones(N,1);
                rng('default');
                 rng('shuffle');
                ind = randsample(N,floor(N*MR)); 
            %     ind=find(rand(N,1)< MR(m));
                ind_folds(ind,v)=0;
                Omega=Omega|ind_folds(:,v);
            end
            indv=find(Omega);
            if length(indv)> floor(N*MR)
            % ind=find(rand(length(indv),1)< MR(m));
              ind = randsample(length(indv),floor(N*MR));
            else
              ind = randsample(length(indv),length(indv));
            end

            ind_folds(:,V)=ones(N,1);
            ind_folds(indv(ind),V)=0;
            clear Omega
            for v=1:V
          %  X{v}=NormalizeFea(X{v},0);
            [Dv,N]=size(X{v});
             Omega{v}=ones(Dv,N);
             ind_0 = find(ind_folds(:,v) == 0);
              Omega{v}(:,ind_0) = 0;    % 缺失视角补0
            end
         i=0;
      %% Missing Cases(MC)         
            i=i+1;
            Time(i)=0;
            for v=1:V
                X_in{v}=X{v}.*Omega{v};     
            end
             X_rec{i}=X_in;
            [RMSE(i),TCS(i)]=EvaluationMetrics(X, X_rec{i},Omega);
            disp([ ' RMSE ' num2str(RMSE(i)), ' .'])
            disp([ ' TCS ' num2str(TCS(i)), ' .'])     
            Result{i} = K_means_c(X_rec{i},double(gt)); 
            disp([ ' ACC ' num2str(Result{i}.ACC), ' .'])
           disp([ ' NMI ' num2str(Result{i}.NMI), ' .'])
           disp([ ' Purity ' num2str(Result{i}.Purity), ' .'])

         %% Ground Truth  (GT) 
           i=i+1;
     Time(i)=0;

           X_rec{i}=X;
           [RMSE(i),TCS(i)]=EvaluationMetrics(X, X_rec{i},Omega);
           disp([ ' RMSE ' num2str(RMSE(i)), ' .'])
           disp([ ' TCS ' num2str(TCS(i)), ' .'])   
           Result{i} = K_means_c(X_rec{i},double(gt)); 
        disp([ ' ACC ' num2str(Result{i}.ACC), ' .'])
        disp([ ' NMI ' num2str(Result{i}.NMI), ' .'])
        disp([ ' Purity ' num2str(Result{i}.Purity), ' .'])
            
            
           i=i+1;    
    %% ITC_MVR
        R=20;%% rank, needs to set with different MRs
        tic;
        [X_rec{i},W]=ITC_MVR(X,Omega,R);
        Time(i)=toc;
        [RMSE(i),TCS(i)]=EvaluationMetrics(X, X_rec{i},Omega);
        disp([ ' done in ' num2str(Time(i)), ' s.'])
        disp([ ' RMSE ' num2str(RMSE(i)), ' .'])
        disp([ ' TCS ' num2str(TCS(i)), ' .'])
            
        Result{i} = feature_Kmeans(W,double(gt)); 
        disp([ ' ACC ' num2str(Result{i}.ACC), ' .'])
        disp([ ' NMI ' num2str(Result{i}.NMI), ' .'])
        disp([ ' Purity ' num2str(Result{i}.Purity), ' .'])
       i=i+1;
    %% SITC_MVR
        R=20;%% rank, needs to set with different MRs
        gamma=1;%% smoothness trade-off parameters
        tic;
        [X_rec{i},W]=SITC_MVR(X_rec{3},Omega,R,gamma,length(unique(gt)));
        Time(i)=toc;
        [RMSE(i),TCS(i)]=EvaluationMetrics(X, X_rec{i},Omega);
        disp([ ' done in ' num2str(Time(i)), ' s.'])
        disp([ ' RMSE ' num2str(RMSE(i)), ' .'])
        disp([ ' TCS ' num2str(TCS(i)), ' .'])
            
        Result{i} = feature_Kmeans(W,double(gt)); 
        disp([ ' ACC ' num2str(Result{i}.ACC), ' .'])
        disp([ ' NMI ' num2str(Result{i}.NMI), ' .'])
        disp([ ' Purity ' num2str(Result{i}.Purity), ' .'])

        methodname ={'MC','GT','ITC-MVR','SITC-MVR'};
        fprintf('\n');
        fprintf('================== Result =====================\n');
        fprintf(' %8.8s \t   %5.4s \t  %5.4s \t  %5.4s \t  %5.4s \t  %5.6s  \t   %5.6s   \n','method','Time','RMSE','TCS','ACC', 'NMI','Purity' );
        for i = 1:4
        fprintf(' %8.8s \t  %5.4f \t %5.4f \t %5.4f \t %6.3f \t   %6.3f \t    %6.3f    \n',...
        methodname{i},Time(i),RMSE(i),TCS(i),Result{i}.ACC, Result{i}.NMI,Result{i}.Purity);
        end
    
    
    