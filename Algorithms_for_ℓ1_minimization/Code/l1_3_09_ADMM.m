function [x,out]= l1_3_09_ADMM(x0, A, b, mu, opts)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by ADMM.
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

%% Step 1: Initialization & reformulation
    
    [~, n] = size(A);
    x = x0;
    y = 0*x0;
    v = 0*x0;
    i = 1;                      % set iteration counter
    objval_path = [];           % set path of objection value 
    objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 

    % LU decompostion for reducing flops
    ATA = (A'*A);
    ATb = A'*b;
    [L, U] = lu(ATA + rho * eye(n));
    
    % set soft threshold operator w.r.t. mu ||x||_1
    soft_thre = @(x,mu) sign(x).*max(0, abs(x) - mu); 
    
%% Step 2: ADMM

    for step = 1:maxsteps
        
        x = U \ (L \ (ATb + rho * y - v));
        y = soft_thre(x + v/rho,mu/rho);
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