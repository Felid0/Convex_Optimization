%% function Test_l1_regularized_problems

%% Readme:
% 
%   Problem: 
%       \min_x 0.5 ||Ax - b||_2^2 + mu ||x||_1
% 
%   Formulation:
%       out = \min_x 0.5 ||Ax - b||_2^2 + mu ||x||_1
%         x = \arg\min_x 0.5 ||Ax - b||_2^2 + mu ||x||_1

%% 0. Data

clear
clc

% % 0.1 generate data

    n = 1024;
    m = 512;
    
    x0 = rand(n,1);
    A = randn(m,n);
    u = sprandn(n,1,0.1);

% % 0.2 save x0,A,u 
%       save('data_sampleN',x0,A,u) 

% % 0.3 load x0,A,u 

%    load('data_sample0.mat')
%    load('data_sample1.mat')
%    load('data_sample2.mat')
    
    b = A*u;
    mu = 1e-3;

%% 1. Solve (1.1) using CVX by calling different solvers mosek and gurobi.

% 1.1 cvx calling mosek

    opts1 = []; 
    tic; 
        [x1, out1] = l1_1_01_cvx_mosek(x0, A, b, mu, opts1);
    t1 = toc;

% 1.2 cvx calling gurobi

    opts2 = []; 
    tic; 
        [x2, out2] = l1_1_02_cvx_gurobi(x0, A, b, mu, opts2);
    t2 = toc;
    

%% 2. First write down an equivalent model of (1.1) which can be solved by calling mosek and gurobi directly, then implement the codes.

% 2.1 call mosek directly

    opts3 = []; 
    tic; 
        [x3, out3] = l1_2_01_mosek(x0, A, b, mu, opts3);
    t3 = toc;

% 2.2 call gurobi directly

    opts4 = []; 
    tic; 
        [x4, out4] = l1_2_02_gurobi(x0, A, b, mu, opts4);
    t4 = toc;
    
    
%% 3. First write down, then implement the following algorithms
%% 3.(a) projection gradient (quadratic program with box constraints)

    opts5 = []; 
    %     options:
    %           ss:         initial step size   (5e-4)
    %           minss:      min step size       (1e-10)
    %           maxss:      max step size       (1e10)
    %           maxsteps:   maximun step        (30)
    %           tol:        tolerance           (1e-5)
    tic; 
        [x5, out5] = l1_3_01_proj_grad(x0, A, b, mu, opts5);
    t5 = toc;

%% 3.(b) subgradient method

    opts6 = []; 
    %     options:
    %           ss:         initial step size   (2.5e-4)
    %         	maxsteps:   maximun step        (100)
    %       	tol:        tolerance           (0.5e-7)
    tic; 
        [x6, out6] = l1_3_02_subgrad(x0, A, b, mu, opts6);
    t6 = toc;


%% 3.(c) gradient method for the smoothed primal problem

    opts7 = []; 
    %     options:
    %        	ss:         initial step size   (5e-4)
    %        	maxsteps:   maximun step        (80)
    %       	tol:        tolerance           (1e-7)
    %         	t:          smooth parameter    (1e-6)
    tic; 
        [x7, out7] = l1_3_03_smooth_grad(x0, A, b, mu, opts7);
    t7 = toc;

%% 3.(d) fast gradient method for the smoothed primal problem
    
% FISTA
    opts8 = [];  
    %     options:
    %        	ss:         initial step size   (5e-4)
    %        	maxsteps:   maximun step        (20)
    %       	tol:        tolerance           (1e-7)
    %           t:          smooth parameter    (1e-6)
    tic; 
        [x8, out8] = l1_3_04_smooth_fgrad_FISTA(x0, A, b, mu, opts8);
    t8 = toc;

%% 3.(e) proximal gradient method for the primal problem

    opts9 = [];  
    %     options:
    %        	ss:         initial step size   (5e-4)
    %        	maxsteps:   maximun step        (50)
    %       	tol:        tolerance           (1e-7)
    tic; 
        [x9, out9] = l1_3_05_prox_grad(x0, A, b, mu, opts9);
    t9 = toc;

%% 3.(f) fast proximal gradient method for the primal problem

    opts10 = [];
    %     options:
    %        	ss:         initial step size   (5e-4)
    %        	maxsteps:   maximun step        (20)
    %       	tol:        tolerance           (1e-7)
    tic; 
        [x10, out10] = l1_3_06_prox_fgrad_FISTA(x0, A, b, mu, opts10);
    t10 = toc;
    
%% 3.(g) Augmented Lagrangian method for the dual problem

    opts11 = [];
    %     options:
    %           rho:            augmented parameter                 (190)
    %        	ss:             initial step size                   (4.5e-6)
    %        	maxsteps:       maximun step                        (30)
    %         	grad_maxsteps:  maximun step of gradient descent 	(20)
    %       	tol:            tolerance                           (1e-7)
    tic; 
        [x11, out11] = l1_3_07_dual_Augmented_Lagrangian(x0, A, b, mu, opts11);
    t11 = toc;

%% 3.(h) Alternating direction method of multipliers for the dual problem

    opts12 = [];
    %     options:
    %           rho:            augmented parameter                 (30)
    %        	maxsteps:       maximun step                        (200)
    %       	tol:            tolerance                           (1e-9)
    tic; 
        [x12, out12] = l1_3_08_dual_ADMM(x0, A, b, mu, opts12);
    t12 = toc;
    
%% 3.(i) Alternating direction method of multipliers with linearization for the primal problem
    
    opts13_1 = [];
    %     options:
    %           rho:            augmented parameter                 (1e-1)
    %           maxsteps:       maximun step                        (200)
    %           tol:            tolerance                           (1e-7)
    tic; 
        [x13_1, out13_1] = l1_3_09_ADMM(x0, A, b, mu, opts13_1);
    t13_1 = toc;
    
    opts13 = [];
    %     options:
    %           rho:            augmented parameter                 (1e-1)
    %           alpha:          linearization parameter on x        (1e-1+2.5e-3)
    %           beta:           linearization parameter on y        (1e-1)
    %           maxsteps:       maximun step                        (300)
    %           tol:            tolerance                           (1e-7)
    tic; 
        [x13, out13] = l1_3_09_ADMM_with_linearization(x0, A, b, mu, opts13);
    t13 = toc;
 
    
%% 4. Write down and implement the deterministic version of AdaGrad, Adam, RMSProp, Momentum 
%% 4.(a) AdaGrad 

    opts18_1 = [];
    opts18_1.mode = "smooth";
    tic; 
        [x18_1, out18_1] = l1_4_01_AdaGrad(x0, A, b, mu, opts18_1);
    t18_1 = toc;
    
    opts18_2 = [];
    opts18_2.mode = "prox";
    tic; 
        [x18_2, out18_2] = l1_4_01_AdaGrad(x0, A, b, mu, opts18_2);
    t18_2 = toc;
    
    opts18 = [];
    opts18.mode = "FISTA";
    tic; 
        [x18, out18] = l1_4_01_AdaGrad(x0, A, b, mu, opts18);
    t18 = toc;

%% 4.(b) Adam

    opts19_1 = [];
    opts19_1.mode = "smooth";
    tic; 
        [x19_1, out19_1] = l1_4_02_Adam(x0, A, b, mu, opts19_1);
    t19_1 = toc;
    
    opts19_2 = [];
    opts19_2.mode = "prox";
    tic; 
        [x19_2, out19_2] = l1_4_02_Adam(x0, A, b, mu, opts19_2);
    t19_2 = toc;
    
    opts19 = [];
    opts19.mode = "FISTA";
    tic; 
        [x19, out19] = l1_4_02_Adam(x0, A, b, mu, opts19);
    t19 = toc;

%% 4.(c) RMSProp

    opts20_1 = [];
    opts20_1.mode = "smooth";
    tic; 
        [x20_1, out20_1] = l1_4_03_RMSProp(x0, A, b, mu, opts20_1);
    t20_1 = toc;
    
    opts20_2 = [];
    opts20_2.mode = "prox";
    tic; 
        [x20_2, out20_2] = l1_4_03_RMSProp(x0, A, b, mu, opts20_2);
    t20_2 = toc;
    
    opts20 = [];
    opts20.mode = "FISTA";
    tic; 
        [x20, out20] = l1_4_03_RMSProp(x0, A, b, mu, opts20);
    t20 = toc;

%% 4.(d) Momentum 

    opts21_1 = [];
    opts21_1.mode = "smooth";
    tic; 
        [x21_1, out21_1] = l1_4_04_Momentum(x0, A, b, mu, opts21_1);
    t21_1 = toc;
    
    opts21_2 = [];
    opts21_2.mode = "prox";
    tic; 
        [x21_2, out21_2] = l1_4_04_Momentum(x0, A, b, mu, opts21_2);
    t21_2 = toc;
    
    opts21 = [];
    opts21.mode = "FISTA";
    tic; 
        [x21, out21] = l1_4_04_Momentum(x0, A, b, mu, opts21);
    t21 = toc;


%% print comparison results with cvx-call-mosek
    
    errfun = @(x1, x2) norm(x1-x2)/(1+norm(x1));

fprintf('cvx_call_mosek                     :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f\n', t1, errfun(x1, x1), out1.optval);
fprintf('cvx-call_gurobi                    :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f\n', t2, errfun(x1, x2), out2.optval);
fprintf('call_mosek                         :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f\n', t3, errfun(x1, x3), out3.optval);
fprintf('call_gurobi                        :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f\n', t4, errfun(x1, x4), out4.optval);
fprintf('projection_gradient                :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t5, errfun(x1, x5), out5.optval, out5.itr);
fprintf('subgradient                        :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t6, errfun(x1, x6), out6.optval, out6.itr);
fprintf('smooth_gradient                    :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t7, errfun(x1, x7), out7.optval, out7.itr);
fprintf('fast_smooth-gradient_FISTA         :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t8, errfun(x1, x8), out8.optval, out8.itr);
fprintf('proximal_gradient                  :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t9, errfun(x1, x9), out9.optval, out9.itr);
fprintf('fast_proximal_gradient_FISTA       :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t10, errfun(x1, x10), out10.optval, out10.itr);
fprintf('ALM_dual                           :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t11, errfun(x1, x11), out11.optval, out11.itr);
fprintf('ADMM_dual                          :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t12, errfun(x1, x12), out12.optval, out12.itr);
fprintf('ADMM                               :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t13_1, errfun(x1, x13_1), out13_1.optval, out13_1.itr);
fprintf('ADMM_with_linearization            :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t13, errfun(x1, x13), out13.optval, out13.itr);
fprintf('AdaGrad_smooth                     :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t18_1, errfun(x1, x18), out18_1.optval, out18_1.itr);
fprintf('AdaGrad_prox                       :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t18_2, errfun(x1, x18), out18_2.optval, out18_2.itr);
fprintf('AdaGrad_FISTA                      :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t18, errfun(x1, x18), out18.optval, out18.itr);
fprintf('Adam_smooth                        :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t19_1, errfun(x1, x19_1), out19_1.optval, out19_1.itr);
fprintf('Adam_prox                          :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t19_2, errfun(x1, x19_2), out19_2.optval, out19_2.itr);
fprintf('Adam_FISTA                         :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t19, errfun(x1, x19), out19.optval, out19.itr);
fprintf('RMSProp_smooth                     :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t20_1, errfun(x1, x20_1), out20_1.optval, out20_1.itr);
fprintf('RMSProp_prox                       :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t20_2, errfun(x1, x20_2), out20_2.optval, out20_2.itr);
fprintf('RMSProp_FISTA                      :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t20, errfun(x1, x20), out20.optval, out20.itr);
fprintf('Momentum_smooth                    :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t21_1, errfun(x1, x21_1), out21_1.optval, out21_1.itr);
fprintf('Momentum_prox                      :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t21_2, errfun(x1, x21_2), out21_2.optval, out21_2.itr);
fprintf('Momentum_FISTA                     :       cpu: %5.2f,   err-to-cvx-mosek: %3.2e   optval: %5.8f     itr: %d\n', t21, errfun(x1, x21), out21.optval, out21.itr);

%% plot 1

    figure(1) 
    semilogy(out5.objval_path, 'LineWidth', 1.0); hold on;
    semilogy(out6.objval_path, 'LineWidth', 1.0);
    semilogy(out7.objval_path, 'LineWidth', 1.0);
    semilogy(out8.objval_path, 'LineWidth', 2.5);
    semilogy(out9.objval_path, 'LineWidth', 1.0); 
    semilogy(out10.objval_path, 'LineWidth', 1.0);
    
    semilogy(out11.objval_path, 'LineWidth', 2.0);
    semilogy(out12.objval_path, 'LineWidth', 1.0);
    % semilogy(out13_1.objval_path, 'LineWidth', 2.0);
    semilogy(out13.objval_path, 'LineWidth', 1.0);

    title('Solution path comparision')
    xlabel('Step')
    ylabel('Objective Value')

    legend( 'projection gradient',...
            'subgradient',...
            'smooth-gradient',...
            'fast smooth gradient-FISTA',...
            'proximal gradient',...
            'fast proximal gradient-FISTA',...
            'ALM-dual',...
            'ADMM-dual',...
            'ADMM with linearization')

    %                 'ADMM',...
    xlim([1, 1000]) 

    
%% plot 2 smooth methods

    figure(2)
    semilogy(out7.objval_path, 'LineWidth', 1.0);hold on;
    % semilogy(out8.objval_path, 'LineWidth', 1.0);   
    semilogy(out18_1.objval_path, 'LineWidth', 1.0);
    semilogy(out19_1.objval_path, 'LineWidth', 1.5);
    semilogy(out20_1.objval_path, 'LineWidth', 1);
    semilogy(out21_1.objval_path, 'LineWidth', 1);
    % To Be Continued...
    % semilogy(out20.objval_path, 'LineWidth', 1.0);

    title('Solution paths of smooth methods')
    xlabel('Step')
    ylabel('Objective Value')

    legend( 'smooth-gradient',...
            'AdaGrad-smooth',...
            'Adam-smooth',...
            'RMSProp-smooth',...
            'Momentum-smooth')
            
    %         'fast smooth gradient-FISTA',...    
    xlim([1, 1000]) 
    
%% plot 3 prox methods

    figure(3)
    semilogy(out9.objval_path, 'LineWidth', 1.0); hold on;
    semilogy(out18_2.objval_path, 'LineWidth', 1.0);
    semilogy(out19_2.objval_path, 'LineWidth', 1.5);
    semilogy(out20_2.objval_path, 'LineWidth', 1);
    semilogy(out21_2.objval_path, 'LineWidth', 1);
    % To Be Continued...
    % semilogy(out20.objval_path, 'LineWidth', 1.0);

    title('Solution paths of proximal methods')
    xlabel('Step')
    ylabel('Objective Value')

    legend( 'proximal gradient',...
            'AdaGrad-prox',...
            'Adam-prox',...
            'RMSProp-prox',...
            'Momentum-prox')
            
    %         'fast smooth gradient-FISTA',...    
    xlim([1, 1000]) 
    
%% plot 4 FISTA methods

    figure(4)
    semilogy(out10.objval_path, 'LineWidth', 1.0);hold on;
    semilogy(out18.objval_path, 'LineWidth', 1.0);
    semilogy(out19.objval_path, 'LineWidth', 1.5);
    semilogy(out20.objval_path, 'LineWidth', 1);
    semilogy(out21.objval_path, 'LineWidth', 1);
    % semilogy(out20.objval_path, 'LineWidth', 1.0);

    title('Solution paths of FISTA-methods')
    xlabel('Step')
    ylabel('Objective Value')

    legend( 'fast proximal gradient-FISTA',...
            'AdaGrad-FISTA',...
            'Adam-FISTA',...
            'RMSProp-FISTA',...
            'Momentum-FISTA')
            
    %         'fast smooth gradient-FISTA',...    
    xlim([1, 1000]) 