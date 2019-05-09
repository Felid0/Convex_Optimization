function [x,out]= l1_3_07_dual_Augmented_Lagrangian(x0, A, b, mu, opts)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by Augmented Lagrangian Method through solving the dual problem.
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
%           ss:             initial step size
%           maxsteps:       maximun step
%           grad_maxsteps:	maximun step of gradient descent
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
        rho = 190;              
    end
    
    % set initial step size
    if isfield(opts, 'ss')          
        ss = opts.ss;
    else
        ss = 4.5e-6;   
    end
    
    % maximun steps
    if isfield(opts, 'maxsteps')    
        maxsteps = opts.maxstep;              
    else
        maxsteps = 30;
    end
    
    % maximum steps of gradient descent
    if isfield(opts, 'grad_maxsteps')    
        grad_maxsteps = opts.grad_maxsteps;              
    else
        grad_maxsteps = 20;
    end
    
    % set tolerance
    if isfield(opts, 'tol')          
        tol = opts.tol;              
    else
        tol = 1e-7; 
    end
    
%% Step 1: Initialization

    % initialization
    v0 = 0 * b;
    x = x0;
    v = v0;
    i = 1;                      % set path counter
    j = 1;                      % set iteration counter
    objval_path = [];           % set path of objection value 
    objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 

    % set soft threshold operator w.r.t. mu ||x||_1
    soft_thre = @(x,mu) sign(x).*max(0, abs(x) - mu);  

%% Step 2: Augmented Lagrangian Method

    for step=1:maxsteps
        
        % 2.1 gradient descent part & update v and z
        for m = 1:grad_maxsteps
            
            grad_v = v + b + rho * A * soft_thre(A' * v - x / rho, mu);
            v = v - ss * grad_v;
            j = j + 1;
            
        end
        z = min(max(-mu, A' * v - x / rho), mu);

        % 2.2 update/recover x & objective value path
        x = x + rho * 1.618 * (z - A' * v);
        i = i + 1;
        objval_path(i) = 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1);
        
        if abs(objval_path(i)-objval_path(i-1)) / objval_path(i-1) <= tol
            break;
        end
        
    end

%% Step 3: Return optimal

%     x;
    out.optval = objval_path(i);
    out.itr = j;
    out.objval_path = objval_path;
    out.status = 'Solved';   
    
end