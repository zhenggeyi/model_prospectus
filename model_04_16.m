clear
%%% Model Parameters
sigma_1=1.8; % in the utility function the risk aversion rate the higher the rate the smoother it shifts close to zero
sigma_2=1.5;
gamma_l=0.25; % gamma for the low productivity production function
gamma_h=0.5;% gamma for the high productivity production function
E = 2; % fixed costs invloved for the high productivity production function
K = 40;   % largest value for capital asset

%%% Discretize the state space
% Note: this doesn't necessarily mean we will model the state space as discrete (more below)
nx   = 200;                              % number of nodes for the initial capital asset level in the grid
Xvec = linspace(0.001,K+20,nx);          % state space vector.

%%% hours space vector
% avec=linspace(min(aupd),max(aupd),na);
hvec=linspace(0,24,nx);         % hours space vector
lvec=24-hvec;
nh = length(hvec);


%%% theta space
thetavec=(0.00001:0.005:1);

%%%draw the functions to see if they make sense
%high technology
f_h=@(k,h) k.^gamma_h.*(h/24).^(1-gamma_h)-E;
%low technology
f_l=@(k,h) k.^gamma_l.*(h/24).^(1-gamma_l);


%plot(lvec,f_h(Xvec,hvec),lvec,f_l(Xvec,hvec))


R = @(c) (c.^(1-sigma_1)-1)./(1-sigma_1);
V = @(L) (L.^(1-sigma_2)-1)./(1-sigma_2);
%%% Randomly draw ni persons with initial k and theta
ni = 300;
rng(1);
thetaupd = normrnd(0.5,0.29,[1,ni]);
for i=1:length(thetaupd)
    % If it is negative, replace it with 0
    if thetaupd(i)<0
        thetaupd(i)=0;
    end
    
end
kupd = 0.1+8*rand(1,ni);
indvupd=[thetaupd;kupd];


%give theta=0.5087 k_0=4.0772
theta=0.2;
%theta=[0.2 0.3 0.4 0.5 0.6 0.7 0.8];
k_0=mean(kupd)-1;
%% characterize the maximization problem
%plot the two parts of the utility functions

% %consumption = @(h) (k_0+f_h(k_0,h));
% R = @(c) ((c).^(1-sigma_1)-1)./(1-sigma_1);
% %plot(hvec,R(hvec))
% %hold on
% V = @(L) (L.^(1-sigma_2)-1)./(1-sigma_2);
% %plot(hvec,V(24-hvec))
% %plot(hvec,R(hvec)+V(24-hvec))
% U = @ (c,h,theta) R(c)+theta*V(24-h);
% %fix theta and initial k
% %draw two budget constraints for non-BOMA
% %plot utility function
% %meshgrid and contour
% [l1,c1]=meshgrid(hvec,Xvec);
% figure
% G1=U(c1,24-l1,theta);
% [C,L]=contour(l1,c1,G1,[1 1.2 1.3 1.4 1.5],'ShowText','on');
% hold on
% cvec=linspace(0.001,K,nx);
% c_h = @(k,h) k+f_h(k,h);
% c_l = @(k,h) k+f_l(k,h);
% c_1=c_h(cvec,hvec);
% c_2=c_l(cvec,hvec);
% plot(lvec,c_1,lvec,c_2)
% xlim([0 24]);ylim([0 K]);
%% build a 10 people village and simulate a network structure
% initialize a 10 * 2 matrix of initial wealth and initial theta
nv=10;
kvec = linspace(0.001,K,nv);
tvec = datasample(thetavec,nv);
% 30% BOMA treated
vindi = [kvec;tvec]; %10 individuals draw from the k space and theta space

%select BOMA women
t=0.3;
%number of BOMA women
nb=nv*t;

%selected three BOMA women treated in the village
[bindi,idxb]=datasample(vindi,nb,2,'Replace',false);

%% given the consumption levels of the BOMA women, let's calculate the
%initialize the network structure for the entire village
%initialize the nof count
nof=NaN*zeros(1,nv);
friends=zeros(nv,nv);

for i=1:nv-1
   %generate random total number of friends
   nof(i)=randi([0 nv-i],1);
   %determine who are the friends
   friends_idx=datasample(i+1:nv,nof(i),'Replace',false);
   friends(i,friends_idx)=1;
end

%correct for the BOMA women as I assume that BOMA women know each other
idxbs=sort(idxb);
for j=1:length(idxb)-1
    for k=j+1:length(idxb)
        friends(idxbs(j),idxbs(k))=1;
    end
end

%symmetric social network matrix
friends=friends+friends';
%probability of meeting one other person in the village
ita=0.1;
%generate the meeting probability matrix %expected number of times that these people meet during one period taking
MV=multi(100,ita,friends);

%change diagonal to 0
for i=1:nv
    MV(i,i)=0;
end

%%
%subtract the BOMA women interaction probability matrix
M1=MV(idxb,idxb);

M2=zeros(nb,nb);
%M would be the network structure among the treated BOMA women
%BOMA treatment status
T=ones(1,nb);





%% BOMA WOMEN
%%Iteration of hvec and cvec
%%% Tolerence Setup
tol_iter = [1e-5, 1e-2, 5*1e-2];
max_iter = [200 400 600 1000];

%initialize the consumptions for the nb
cvec_i = NaN*zeros(1,nb);
%with no network effect - everybody consumes her initial wealth level and
%chooses the optimal level of labor based on the current parameters
%initialize

%initial value of cvec_j is the intial wealth level
cvec_j= bindi(1,:);

%U = @(k,h,theta) R(k+f_h(k,h))+(theta-M2(i,:)*T').*V(1-h);
%switch between with and with/o network effects
%with network effects
M=M1;
%with no network effects
% M=M2;
%person i's utility function
% U = @(k,h,theta,cvec_j) R(k+f_h(k,h)-M(i,:)*(k+f_h(k,h)-cvec_j'))+(theta-M(i,:)*T').*V(24-h);

tol = tol_iter(1);
maxAbsDiff = 2*tol; % initiate convergence check metric (to any level > tol).
t = 1;     % initiate iteration counter

while maxAbsDiff>tol
    tic;
    %t   % iteration counter
    
    hvec = NaN*zeros(1,nb); % Initialize place holder for updated consumption levels.
    expU = NaN*zeros(1,nb); % Initialize place holder for updated total utility levels.
    %R1 = NaN*zeros(1,nb);
    %R2 = NaN*zeros(1,nb);
    %V1 = NaN*zeros(1,nb);
    %calculate one period hvec
    
    for i=1:length(cvec_i)
        k=bindi(1,i);
        theta=bindi(2,i);
        % f = @(h) -U(k,h,theta,cvec_j);
        f = @(h) -(R(k+f_h(k,h)-M(i,:)*(k+f_h(k,h)-cvec_j'))+(theta-M(i,:)*T').*V(24-h));
        [hvec(i),expU(i)]=fminbnd(f,0,24);
    end
    
    %calculate cvec_j using hvec(i)
    cvec_i=f_h(bindi(1,:),hvec)+bindi(1,:);
    %to check the behavior of the functions
%     for i=1:length(cvec_i)
%         R1(i)=cvec_i(i)-M(i,:)*(cvec_i(i)-cvec_j');
%         R2(i)=R(R1(i));
%         V1(i)=V(24-hvec(i));
%     end
    
    maxAbsDiff = max(max(abs(cvec_i-cvec_j)));
    %message control
    disp(strcat('_Interation_', num2str(t), ': maxAbsDiff=', num2str(maxAbsDiff), ', tol=', num2str(tol)));
    
    timespend=toc;
    disp(strcat('Time Spent: ', num2str(timespend)));
    
    cvec_j = cvec_i; % Update others' consumptions with updated consumptions
    t = t+1;
    
    tol_idx = find(max_iter<t,1,'first');
    if tol_idx>0
        tol = tol_iter(tol_idx+1);
    end
    
end

%% village consumption j vector
%non-BOMA women's consumption levels
%index of non-BOMA women in the village
idxb_b=setdiff(1:nv,idxb);
%initialize the c_j for every individual in the village
cindj=NaN*zeros(1,nv);
%replace the BOMA treated ones consumption levels with the ones calculated before
cindj(idxb)=cvec_j;
%replace the rest of the non-treated women's initial c_j with their initial
%wealth level
cindj(idxb_b)=vindi(1,idxb_b);
%BOMA treatment status
TV=zeros(1,nv);
TV(idxb)=1;
MV2=zeros(nv,nv);

%% non-BOMA WOMEN in the village
%%Iteration of hvec and cvec
%%% Tolerence Setup
tol_iter = [1e-5, 1e-2, 5*1e-2];
max_iter = [200 400 600 1000];

%initialize the consumptions for the nb
cvec_iv = NaN*zeros(1,nv);
cvec_iv(idxb)=cvec_i;
%with no network effect - everybody consumes her initial wealth level and
%chooses the optimal level of labor based on the current parameters
%initialize

%initial value of cvec_j is the intial wealth level
cvec_jv= cindj;

%U = @(k,h,theta) R(k+f_h(k,h))+(theta-M2(i,:)*T').*V(1-h);
%switch between with and with/o network effects
%with network effects
%M=MV;
%with no network effects
M=MV2;
%person i's utility function
% U = @(k,h,theta,cvec_j) R(k+f_h(k,h)-M(i,:)*(k+f_h(k,h)-cvec_j'))+(theta-M(i,:)*T').*V(24-h);

tol = tol_iter(1);
maxAbsDiff = 2*tol; % initiate convergence check metric (to any level > tol).
t = 1;     % initiate iteration counter

while maxAbsDiff>tol
    tic;
    %t   % iteration counter
    
    hvec_nb = NaN*zeros(3,nv-nb); % Initialize place holder for updated consumption levels.
    expU_nb = NaN*zeros(2,nv-nb); % Initialize place holder for updated total utility levels.
    %R1 = NaN*zeros(1,nb);
    %R2 = NaN*zeros(1,nb);
    %V1 = NaN*zeros(1,nb);
    %calculate one period hvec
    
    for i=1:nv-nb
        k=vindi(1,idxb_b(i));
        theta=vindi(2,idxb_b(i));
        % f = @(h) -U(k,h,theta,cvec_j);
        %compare the high return technology and the low return technology
        fh = @(h) -(R(k+f_h(k,h)-M(idxb_b(i),:)*(k+f_h(k,h)-cvec_jv'))+(theta-M(idxb_b(i),:)*TV').*V(24-h));
        [hvec_nb(1,i),expU_nb(1,i)]=fminbnd(fh,0,24);
        fl = @(h) -(R(k+f_l(k,h)-M(idxb_b(i),:)*(k+f_l(k,h)-cvec_jv'))+(theta-M(idxb_b(i),:)*TV').*V(24-h));
        [hvec_nb(2,i),expU_nb(2,i)]=fminbnd(fl,0,24);
            if expU_nb(1,i)~= expU_nb(2,i)
                if expU_nb(1,i)<expU_nb(2,i)
                hvec_nb(3,i)= hvec_nb(1,i);  
                cvec_iv(idxb_b(i))=f_h(k,hvec_nb(3,i))+k;
                else
                hvec_nb(3,i)= hvec_nb(2,i);
                cvec_iv(idxb_b(i))=f_l(k,hvec_nb(3,i))+k;
                end
            else
                hvec_nb(3,i)= hvec_nb(2,i);
                cvec_iv(idxb_b(i))=f_l(k,hvec_nb(3,i))+k;
            end
        %fl = @(h) -(R(k+f_l(k,h)-M(idxb_b(i),:)*(k+f_l(k,h)-cvec_jv'))+(theta-M(idxb_b(i),:)*TV').*V(24-h));
        %[hvec(i),expU(i)]=fminbnd(fh,0,24);
    end
    
    %calculate cvec_j using hvec(i)
    %cvec_iv(idxb_b)=f_h(vindi(1,idxb_b),hvec_nb(3,:))+vindi(1,idxb_b);
    %to check the behavior of the functions
%     for i=1:length(cvec_i)
%         R1(i)=cvec_i(i)-M(i,:)*(cvec_i(i)-cvec_j');
%         R2(i)=R(R1(i));
%         V1(i)=V(24-hvec(i));
%     end
    
    maxAbsDiff = max(max(abs(cvec_iv-cvec_jv)));
    %message control
    disp(strcat('_Interation_', num2str(t), ': maxAbsDiff=', num2str(maxAbsDiff), ', tol=', num2str(tol)));
    
    timespend=toc;
    disp(strcat('Time Spent: ', num2str(timespend)));
    
    cvec_jv = cvec_iv; % Update others' consumptions with updated consumptions
    t = t+1;
    
    tol_idx = find(max_iter<t,1,'first');
    if tol_idx>0
        tol = tol_iter(tol_idx+1);
    end
    
end