function [x,out]= l1_3_04_smooth_fgrad_FISTA(x0, A, b, mu, opts)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by FISTA for the smoothed primal problem.
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
%           s:          smooth parameter
% 
%   Returns:
%       x: m by 1 matrix  i.e. the optimal point
%       out: optimal value of objective
% 
%   ================== Smoothing and reformulation ========================
%   p --->  \min_x 0.5 x'(A'A)x + (-A'b)'x + 0.5 (b'b) + mu Huber(x)
%           where Huber(x) is the huber penalty on x
% 
%      <=>  \min_x g(x) + h(x)
%           where   g(x) = 0.5 x'(A'A)x + (-A'b)'x + 0.5 (b'b)
%                   h(x) = mu Huber(x)

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
        maxsteps = 20;
    end
    
    % set tolerance
    if isfield(opts, 'tol')          
        tol = opts.tol;              
    else
        tol = 1e-7; 
    end
    
    % set smooth parameter
    if isfield(opts, 's')
        s = opts.s;              
    else
        s = 1e-6;
    end
    
%% Step 1: Initialization & reformulation
    
    % reformulation
    Q = (A'*A);                 % set Q = (A'A)
    c = -A'*b;                  % set c = -A'b
    con = 0.5 * (b'* b) ;       % set constent part of objective, 0.5 * b^T b
    
    % initialization
    x = x0;
    x_backward = x0;
    i = 1;                      % set iteration counter
    mu_finder = 100;            % initial mu
    objval_path = [];           % set path of objection value 
    objval_path(i) = 0.5 * x'* Q *x + c'*x + con + mu * norm(x, 1);    
    
    % set proximal operator w.r.t. h(x) = mu Huber(x)
    prox_Huber = @(x,mu,s) (1-mu./max(abs(x), mu+s)).*x; 

%% Step 2: Smooth-gradient-Descent

    while mu_finder > mu 
        mu_finder =  (mu_finder + mu) * 0.5;
        
        for step = 1:maxsteps
            
            % step2.1: FISTA part and update variable y and x^(-1)
            y = x + (step - 2) / (step + 1.0) * (x - x_backward);
            x_backward = x;
            
            % step2.2: gradient descent part and update variable x
            g_grad_y = Q * y + c;
            z = y - ss * g_grad_y;
            x = prox_Huber(z,ss * mu_finder,s);     % Proximal mapping with line search
             
            % step2.3: update objective value path 
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