function [x,out]= l1_3_09_ADMM_with_linearization(x0, A, b, mu, opts)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by ADMM with linearization
% 
%       out = \min_x 0.5 ||Ax - b||_2^2 + mu ||x||_1
%         x = \arg\min_x 0.5 ||Ax - b||_2^2 + mu ||x||_1
% 
%   Parameters:
%       x0:     m by 1 matrix
%       A:      m by n matrix
%       b:      m by 1 matrix
%       mu:     scalar
%       opts:   
%           rho:            augmented parameter
%           alpha:          linearization parameter on x 
%           beta:           linearization parameter on y 
%           maxsteps:       maximun step
%           tol:            tolerance
% 
%   Returns:
%       x: m by 1 matrix  i.e. the optimal point
%       out: optimal value of objective

%% Step 0: Set default options 
    
    % set rho
    if isfield(opts, 'rho')          
        rho = opts.rho;
    else
        rho = 1e-1;       
    end
    
    % set linearization parameter - alpha
    if isfield(opts, 'alpha')          
        alpha = opts.alpha;
    else
        alpha = rho + 2.5e-3;              
    end
    
    % set linearization parameter - beta
    if isfield(opts, 'beta')          
        beta = opts.beta;
    else
        beta = rho;              
    end
    
    % maximun steps
    if isfield(opts, 'maxsteps')    
        maxsteps = opts.maxstep;              
    else
        maxsteps = 200;
    end
    
    % set tolerance
    if isfield(opts, 'tol')          
        tol = opts.tol;              
    else
        tol = 1e-7; 
    end
    
%% Step 1: Initialization

    [~, n] = size(A);
    x = x0;
    y = 0*x0;
    v = 0*x0;
    i = 1;                      % set iteration counter
    objval_path = [];           % set path of objection value 
    objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 

    % precalculation for reducing flops
    ATA = (A'*A);
    ATb = A'*b;
    Q = ATA + alpha * eye(n);
    [L, U] = lu(Q);
    
    % set soft threshold operator w.r.t. mu ||x||_1
    soft_thre = @(x,mu) sign(x).*max(0, abs(x) - mu); 
    
  %% Step 2: ADLPMM

    % linearization on x & y 
    for step = 1:maxsteps

        x = U \ (L \ (ATb + (alpha - rho)*x + rho*y - v));
        y = soft_thre(y + (rho *(x - y) + v)/beta, mu/beta);
        v = v + rho * 1.618 * (x - y);
        
        i = i + 1;
        objval_path(i) = 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1);

        if abs(objval_path(i)-objval_path(i-1)) / objval_path(i-1) <= tol
            break;
        end

    end
    
%% Step 3: Return optimal

%     x;
    out.optval = objval_path(i);
    out.itr = i;
    out.objval_path = objval_path;
    out.status = 'Solved';   
    
end