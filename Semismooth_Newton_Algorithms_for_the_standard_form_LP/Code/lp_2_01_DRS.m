function [x,out]= lp_2_01_DRS(c, A, b, opts, x0)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by DRS through solving the primal problem.
% 
%       \min_x c'x
%        s.t.  Ax = b
%              x >= 0
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

    if isfield(opts, 'lambda')   
        lambda = opts.lambda;
    else
        lambda = 20;              
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
  
%% Initialization

    [~, n] = size(A);
    
    i = 1;
    x = x0;
    v = 0*x0;
    Mat2 = eye(n) - A'*((A*A') \ A);
    Mat1 = A'*((A*A') \ b);
    
    primal_objval_path = [];
    primal_objval_path(i) = c'*x;

%% Step 2: DRS

    for i = 1:maxsteps  
        
        u = max(0, x - v - lambda * c);   
        x = Mat2*(u + v) + Mat1;
        v = v + (u - x);
        primal_objval_path(i) = c'*x;
        
        if mod(i, 1000) == 0
            if norm(u-x) < tol && norm(A*x-b,inf) < tol
                break;
            end
        end
    end
    
%% Step 3: Return optimal
  
    out.optval = primal_objval_path(i);
    out.objval_path = primal_objval_path;
    out.itr = i;
    out.status = 'Solved'; 
    
end