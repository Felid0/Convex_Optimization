function [x,out]= l1_3_05_prox_grad(x0, A, b, mu, opts)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by proximal gradient descent method
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
%           ss:         initial step size
%           maxsteps:    maximun step
%           tol:        tolerance
% 
%   Returns:
%       x: m by 1 matrix  i.e. the optimal point
%       out: optimal value of objective
% 
%   ===================== reformulation ========================
%   p  <=>  \min_x g(x) + h(x)
%           where   g(x) = 0.5 x'(A'A)x + (-A'b)'x + 0.5 (b'b)
%                   h(x) = mu ||x||_1

%% Step 0: Set default options 

    % set initial step size
    if isfield(opts, 'ss')          
        ss = opts.ss;
    else
        ss = 5e-4;              
    end
    
    % maximun steps
    if isfield(opts, 'maxsteps')    
        maxsteps = opts.maxsteps;              
    else
        maxsteps = 50;
    end
    
    % set tolerance
    if isfield(opts, 'tol')          
        tol = opts.tol;              
    else
        tol = 1e-7; 
    end
    
%% Step 1: Initialization & reformulation

    [~, n] = size(A);
    
    % reformulation
    Q = (A'*A);                 % set Q = (A'A)
    c = -A'*b;                  % set c = -A'b
    con = 0.5 * (b'* b) ;       % set constent part of objective, 0.5 * b^T b
    
    % initialization
    x = x0;
    i = 1;                      % set iteration counter
    mu_finder = 100;            % initial mu
    objval_path = [];           % set path of objection value 
    objval_path(i) = 0.5 * x'* Q *x + c'*x + con + mu * norm(x, 1);     
    
    % set proximal operator of mu ||x||_1
    prox_l1 = @(x,mu) sign(x).*max(0, abs(x) - mu); 

%% Step 2: Proximal-Gradient-Descent

    while mu_finder > mu 
        mu_finder =  (mu_finder + mu) * 0.5;
        
        for step = 1:maxsteps
 
            % step2.1: gradient decesent part and update variable x
            g_grad_x = Q * x + c ;
            y = x - ss * g_grad_x;
            x = prox_l1(y,ss * mu_finder);     % Proximal mapping with line search
            
            % step2.2: update objective value path 
            i = i + 1;
%             objval_path(i) = 0.5 * x'* Q *x + c'*x + con + mu * norm(x, 1);              
            objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1);   % save time
 
            % loop criterion
            if abs(objval_path(i)-objval_path(i-1)) / objval_path(i-1) <= tol 
                break;
            end
        end
    end

%% Step 3: Return optimal

%     x;
    out.optval = objval_path(i);
    out.itr = i;
    out.objval_path = objval_path;
    out.status = 'Solved';   
    
end