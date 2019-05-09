function [x,out]= l1_3_08_dual_ADMM(x0, A, b, mu, opts)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by ADMM solving the dual problem.
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
% 
%   ===================== reformulation ========================
%   p  <=dual=>	\min_(v,z)	0.5 v'v + b'v + I(||z||_inf <= mu)
%            	 s.t.       A'v - z = 0
%      where x is the multiplier of the dual

%% Step 0: Set default options 

    
    % set rho
    if isfield(opts, 'rho')          
        rho = opts.rho;
    else
        rho = 30;              
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
        tol = 1e-9; 
    end
    
%% Step 1: Initialization

    [m, ~] = size(A);
    v0 = 0 * b;
    x = x0;
    v = v0;
    i = 1;                      % set iteration counter
    objval_path = [];           % set path of objection value 
    objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 

    % LU decompostion for reducing flops
    [L, U] = lu(eye(m) + rho * (A*A'));  

%% Step 2: ADMM

    for step = 1:maxsteps
        
        z = min(max(-mu, A' * v - x / rho), mu);
        v = U \ (L \ (rho * A * (z + x / rho) - b));
        x = x + rho * 1.618 * (z - A' * v);
        
        i = i + 1;
        objval_path(i) = 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1);
        
        if abs(objval_path(i)-objval_path(i-1)) / objval_path(i-1) <= tol
%         if abs(abs(x) - abs(x_backward))<= tol          
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