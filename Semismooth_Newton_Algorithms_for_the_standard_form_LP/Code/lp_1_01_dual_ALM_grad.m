function [x,out]= lp_1_01_dual_ALM_grad(c, A, b, opts, x0)
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

    % set rho
    if isfield(opts, 'rho')          
        rho = opts.rho;
    else
        rho = 2;              
    end
    
    % set initial step size
    if isfield(opts, 'ss')          
        ss0 = opts.ss;
    else
        ss0 = 5e-3;   
    end
    
    % set range of step size
    if isfield(opts, 'maxss')          
        maxss = opts.maxss;
    else
        maxss = 5e-1;   
    end
    
    if isfield(opts, 'minss')
        minss = opts.minss;
    else
        minss = 1e-5;   
    end
    
    % maximun steps
    if isfield(opts, 'maxsteps')    
        maxsteps = opts.maxstep;              
    else
        maxsteps = 1000;
    end
    
    % maximum steps of gradient descent
    if isfield(opts, 'grad_maxsteps')    
        grad_maxsteps = opts.grad_maxsteps;              
    else
        grad_maxsteps = 1000;
    end
    
    % set tolerance
    if isfield(opts, 'tol')          
        tol = opts.tol;              
    else
        tol = 1e-9; 
    end
    
%% Step 1: Initialization

    % initialization
    x = x0;
    y = 0*b;
    i = 1;
    j = 1;
    primal_objval_path = [];
    dual_objval_path = [];
    primal_objval_path(i) = c'*x;
    dual_objval_path(i) = b'*y;

%% Step 2: ALM

    for i = 1:maxsteps
        ss = ss0;
  
        for k = 1:grad_maxsteps
            j = j + 1;
            
            % update y
            y_backward = y;
            grad_y = -b + rho * (A * max(0, A'*y + x/rho - c));    
            y = y - ss*grad_y;
            
            % update s
            grad_y_forward = -b + rho * (A * max(0, A'*y + x/rho - c));         
            dis_g = grad_y - grad_y_forward;
            ss = min(max(minss,ss * (grad_y'*grad_y)/(grad_y'*dis_g)),maxss);
            
            % when to stop 
            if norm(y - y_backward,inf) < tol
                break;
            end
        end
        
        % update x
        x = max(0,rho*(A'*y-c)+x);
        primal_objval_path(i) = c'*x;
        dual_objval_path(i) = b'*y;
    end
    
%% Step 3: Return optimal

%     x;
    out.optval = primal_objval_path(i);
    out.dual_optival = dual_objval_path(i);
    out.objval_path = primal_objval_path;
    out.dual_objval_path = dual_objval_path;
    out.itr = j;
    out.status = 'Solved';   
    
end