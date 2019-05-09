%% function Test_l1_problems
%% Readme:
% 
%   (1) Primal problem: 
%       \min_x c'x
%        s.t.  Ax = b
%              x >= 0
% 
%   (2) Dual problem:
%       \min_y -b'y
%        s.t.  A'y+s = c
%              s >= 0

%% 0. Data

        clear
        clc

% 0.1 Random data

     n = 100;
     m = 20;
     A = rand(m, n);
     xs = full(abs(sprandn(n, 1, m / n)));
     b = A * xs;
     y = randn(m, 1);
     s = rand(n, 1) .* (xs == 0);
     c = A'*y + s;
     x0 = abs(randn(n, 1));
    
% save test data x0,A,b,c

%    save('data_sampleN.mat','x0','A','b','c','xs','y','s') 

% load test data x0,A,b,c 

%    load('data_sample1.mat')
%    load('data_sample2.mat')
%    load('data_sample3.mat')
    
% 0.2 Netlib

%     load .\Netlib\sc50bpre.mat;
%     load .\Netlib\sc105pre.mat;
    
%    out = reformulation(Model);
%    c = out.c;
%    A = out.A;
%    b = out.b;
%    [m, n] = size(A);
%    x0 = abs(randn(n, 1));



%% 0. mosek and gurobi

  % 0.1. Call mosek and gurobi through CVX.
    % 0.1.1 cvx calling mosek

        opts1 = []; 
        tic; 
            [x1, out1] = lp_0_01_cvx_mosek(c, A, b, opts1, x0);
        t1 = toc;

    % 0.1.2 cvx calling gurobi

        opts2 = []; 
        tic; 
            [x2, out2] = lp_0_02_cvx_gurobi(c, A, b, opts2, x0);
        t2 = toc;
    
%   % 0.2. Call mosek and gurobi directly.
%     % 0.2.1 call mosek directly
% 
%         opts3 = []; 
%         tic; 
%             [x3, out3] = lp_0_01_mosek(c, A, b, opts3, x0);
%         t3 = toc;
% 
%     % 0.2.2 call gurobi directly
% 
%         opts4 = []; 
%         tic; 
%             [x4, out4] = lp_0_02_gurobi(c, A, b, opts4, x0);
%         t4 = toc;

%% 1. Augmented Lagrangian method for the dual problem 

  % 1.1 Gradient method

    opts5 = [];
    tic; 
        [x5, out5] = lp_1_01_dual_ALM_grad(c, A, b, opts5, x0);
    t5 = toc;

  % 1.2 Semi-smooth Newton method

    opts6 = [];
    tic; 
        [x6, out6] = lp_1_02_dual_ALM_SSN(c, A, b, opts5, x0);
    t6 = toc;
    
%% 2. Semi-smooth Newton method based on solving a fixed-point equation.

  % 2.1 DRS

    opts7 = [];
    tic; 
        [x7, out7] = lp_2_01_DRS(c, A, b, opts7, x0);
    t7 = toc;

  % 2.2 ADMM

    opts8 = [];
    tic; 
        [x8, out8] = lp_2_02_dual_ADMM(c, A, b, opts8, x0);
    t8 = toc;   
    
  % 2.3 RSSN

    opts9 = [];
    tic; 
        [x9, out9] = lp_2_03_RSSN(c, A, b, opts9, x0);
    t9 = toc;     
    
    
%% print comparison results with xs for random data
% 
% % 0.1 errfun 
%     errfun = @(x) norm(xs-x)/(1+norm(xs));
% %     errfun = @(x1, x2) norm(x1-x2)/(1+norm(x1));
% 
% 
% fprintf('cvx_call_mosek  	:       cpu: %5.2f,   err-to-xs: %3.2e   optval: %5.8f\n', t1, errfun(x1), out1.optval);
% fprintf('cvx-call_gurobi 	:       cpu: %5.2f,   err-to-xs: %3.2e   optval: %5.8f\n', t2, errfun(x2), out2.optval);
% fprintf('call_mosek     	:       cpu: %5.2f,   err-to-xs: %3.2e   optval: %5.8f\n', t3, errfun(x3), out3.optval);
% fprintf('call_gurobi    	:       cpu: %5.2f,   err-to-xs: %3.2e   optval: %5.8f\n', t4, errfun(x4), out4.optval);
% fprintf('ALM_grad_dual     	:       cpu: %5.2f,   err-to-xs: %3.2e   optval: %5.8f     itr: %d\n', t5, errfun(x5), out5.optval, out5.itr);
% fprintf('ALM_SSN_dual      	:       cpu: %5.2f,   err-to-xs: %3.2e   optval: %5.8f     itr: %d\n', t6, errfun(x6), out6.optval, out6.itr);
% fprintf('DRS                :       cpu: %5.2f,   err-to-xs: %3.2e   optval: %5.8f     itr: %d\n', t7, errfun(x7), out7.optval, out7.itr);
% fprintf('ADMM_dual      	:       cpu: %5.2f,   err-to-xs: %3.2e   optval: %5.8f     itr: %d\n', t8, errfun(x8), out8.optval, out8.itr);
% fprintf('RSSN               :       cpu: %5.2f,   err-to-xs: %3.2e   optval: %5.8f     itr: %d\n', t9, errfun(x9), out9.optval, out9.itr);


%% print comparison results with xs for Netlib data

    errfun = @(x) norm(x1(c~=0)-x(c~=0))/(1+norm(x1(c~=0)));
    
fprintf('cvx_call_mosek  	:       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f\n', t1, errfun(x1), out1.optval);
fprintf('cvx-call_gurobi 	:       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f\n', t2, errfun(x2), out2.optval);
% fprintf('call_mosek     	:       cpu: %5.2f,   err-to-xs: %3.2e   optval: %5.8f\n', t3, errfun(x3), out3.optval);
% fprintf('call_gurobi    	:       cpu: %5.2f,   err-to-xs: %3.2e   optval: %5.8f\n', t4, errfun(x4), out4.optval);
fprintf('ALM_grad_dual     	:       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t5, errfun(x5), out5.optval, out5.itr);
fprintf('ALM_SSN_dual      	:       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t6, errfun(x6), out6.optval, out6.itr);
fprintf('DRS                :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t7, errfun(x7), out7.optval, out7.itr);
fprintf('ADMM_dual      	:       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t8, errfun(x8), out8.optval, out8.itr);
fprintf('RSSN               :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t9, errfun(x9), out9.optval, out9.itr);
