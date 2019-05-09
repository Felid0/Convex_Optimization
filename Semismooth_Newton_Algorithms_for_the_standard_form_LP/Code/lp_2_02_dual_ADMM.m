function [x,out]= lp_2_02_dual_ADMM(c, A, b, opts, x0)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by ADMM through solving the dual problem.
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
        rho = 20;              
    end
    
    if isfield(opts, 'maxsteps')   
        maxsteps = opts.maxsteps;
    else
        maxsteps = 2e4;
    end
    
    if isfield(opts, 'tol')
        tol = opts.err1;
    else
        tol = 1e-10;
    end
  
%% Step 1: Initialization
    
    i = 1;
    x = x0;
    s = 0*x0;
    y = 0*b;
    L = chol((A*A'), 'lower');
    ATy = A'*y;
    
    primal_objval_path = [];
    primal_objval_path(i) = c'*x;

%% Step 2: ADMM

    for i = 1:maxsteps 

        s = max(0, -ATy + c - x/rho);
        y = L'\(L\ ((A * (c - s - x/rho)) + b/rho));
        ATy = A'*y;
        
        delta = (ATy + s - c);
        x = x + 1.618 * delta;
        
        primal_objval_path(i) = c'*x;
        
        if norm(delta) < tol && norm(A*x-b,inf) < tol
            break;
        end
    end
    
%% Step 3: Return optimal
  
    out.optval = primal_objval_path(i);
    out.objval_path = primal_objval_path;
    out.itr = i;
    out.status = 'Solved';      
    
end