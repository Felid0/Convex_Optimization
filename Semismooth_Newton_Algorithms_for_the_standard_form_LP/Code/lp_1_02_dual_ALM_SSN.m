function [x,out]= lp_1_02_dual_ALM_SSN(c, A, b, opts, x0)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by Augmented Lagrangian Method through solving the dual problem.
% 
%       \min_y -b'y
%        s.t.  A'y+s = c
%              s >= 0
% 
%   Parameters:
%       x0:     m by 1 matrix
%       A:      m by n matrix
%       b:      m by 1 matrix
%       c:      m by 1 matrix
%       opts:   (options)
% 
%   Returns:
%       x:      m by 1 matrix  i.e. the optimal point
%       out:    optimal value of the objective

%% Step 0: Set default options 

  	if isfield(opts, 'rho')             
        rho = opts.rho;
    else
        rho = 1500;              
    end
    
    if isfield(opts, 'ss0')
        ss0 = opts.ss0;
    else
        ss0 = 1;
    end
    
    if isfield(opts, 'tau')
        tau = opts.tau;
    else
        tau = 1e-4;
    end
    
    if isfield(opts, 'tau1')
        tau1 = opts.tau1;
    else
        tau1 = 1e-4;
    end
    
    if isfield(opts, 'tau2')
        tau2 = opts.tau2;
    else
        tau2 = 1e-2;
    end
    
    if isfield(opts, 'eta0')
        eta0 = opts.eta0;
    else
        eta0 = 1e-6;
    end    
    
    if isfield(opts, 'maxsteps')   
        maxsteps = opts.maxsteps;
    else
        maxsteps = 1000;
    end
    
    if isfield(opts, 'grad_maxsteps')  
        grad_maxsteps = opts.grad_maxsteps;
    else
        grad_maxsteps = 50;
    end
    
    if isfield(opts, 'tol')
        tol = opts.tol;
    else
        tol = 1e-6;
    end

    if isfield(opts, 'mu')
        mu = opts.mu;
    else
        mu = 0.1;
    end
    
    if isfield(opts, 'delta')
        delta = opts.delta;
    else
        delta = 0.8;
    end

%% Step 1: Initialization

    x = x0;
    y = 0*b;
    i = 1;
    j = 1;
    primal_objval_path = [];
    dual_objval_path = [];
    primal_objval_path(i) = c'*x;
    dual_objval_path(i) = b'*y;
    
    lag = @(y, x) -b' * y + rho / 2* norm(max(0,A' * y - c + x/rho))^2;

%% Step 2: ALM
    for i = 1:maxsteps     

        for k = 1:grad_maxsteps 
            j = j + 1;
            
          % update gradient and Hermitian
            thre = max(0, A' * y - c + x/rho);
            grad_y = -b + rho * (A * thre);
            indicator = (thre ~= 0);
            A_1 = A(:,indicator);
            H = rho*(A_1*A_1');
            
          % Solve (H + epsilon * I) d = g
            grad_norm = norm(grad_y);
            epsilon = tau1 * min(tau2, grad_norm);  
            [m, ~] = size(A);
            h = sparse(H + epsilon * eye(m)); 
            
            % Cholesky decomposition
            L = ichol(h, struct('type', 'ict', 'droptol', 1e-10, 'diagcomp', 0.0001));
            
            % CG
            N = 100;
            eta = min(eta0, grad_norm ^ (1 + tau));
            [d, ~] = pcg(h, grad_y, eta, N, L, L'); 
            
          % Backtracking
            ss = ss0;
            while lag(y - ss*d, x) > lag(y, x) - mu*ss*(grad_y'*d)
                ss = ss * delta;
            end
            
          % update y
            y = y - ss * d;
            
            if grad_norm < tol
                break
            end
            
        end
        
      % update x
        x_pre = rho * (A' * y - c) + x;
        x = max(0, x_pre);

        primal_objval_path(i) = c'*x;
        dual_objval_path(i) = b'*y;
        
      % when to stop
        pobj = primal_objval_path(i);
        dobj = dual_objval_path(i);        
        R_P = norm(A * x - b) / (1 + norm(b));
        R_D = norm(A' * y + (x - x_pre) / rho - c) / (1 + norm(c)); 
        gap = abs(pobj - dobj) /(1 + abs(pobj) + abs(dobj));
        
        if max([R_P, R_D, gap]) < tol
            break
        end
        
    end

%% Step 3: Return optimal
  
    out.optval = primal_objval_path(i);
    out.dual_optval = dual_objval_path(i);
    out.objval_path = primal_objval_path;
    out.dual_path = dual_objval_path;
    out.itr = j;
    out.status = 'Solved';     
    
end