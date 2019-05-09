function [x,out]= l1_3_01_proj_grad(x0, A, b, mu, opts)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by projection gradient method.
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
%           minss:      min step size
%           maxss:      max step size
%           maxsteps:   maximun step
%           tol:        tolerance
% 
%   Returns:
%       x: m by 1 matrix  i.e. the optimal point
%       out: optimal value of objective
% 
%   ================== Reformulation =============================
%   p <=> \min_x 0.5 x'(A'A)x + (-A'b)'x + 0.5 (b'b) + mu ||x||_1
%     
%     <=> \min_(x_+;x_-) 0.5 (x_+;x_-)'(A'A,-A'A;-A'A,A'A)(x_+;x_-) 
%                        + (mu 1-A'b;mu 1+A'b)'(x_+;x_-) 
%          s.t.  0 <= (x_+;x_-) < \infty
% 
%     <=> \min_y 0.5 y'Q y + c'y
%          s.t.  0 <= y < \infty 

%% set initial step size

    if isfield(opts, 'ss')          
        ss = opts.ss;
    else
        ss = 5e-4;              
    end
    
    % set min step size
    if isfield(opts, 'minss')
        minss = opts.minss;      
    else
        minss = 1e-10;
    end
    
    % set max step size
    if isfield(opts, 'maxss')
        maxss = opts.maxss;      
    else
        maxss = 1e10;
    end
    
    % maximun steps
    if isfield(opts, 'maxsteps')    
        maxsteps = opts.maxsteps;              
    else
        maxsteps = 30;
    end
    
    % set tolerance
    if isfield(opts, 'tol')          
        tol = opts.tol;              
    else
        tol = 1e-5; 
    end
    

%% Step 1: Initialization & reformulation
    
    [~, n] = size(A);
    
    % reformulation
    A_square = (A'*A);
    Q = [A_square, -A_square; -A_square, A_square];     % set Q = (A'A,-A'A;-A'A,A'A)
    ATb = A'*b;
    c = @(var) var * ones(2 * n,1) + [-ATb;ATb];        % set c = (mu 1-A'b;mu 1+A'b)
    x = x0;
    x_p = x.*(x >= 0);
    x_n = -x.*(x < 0);
    y = [x_p; x_n];                                     % set y = (x_+; x_-)
    con = 0.5 * (b'* b) ;                               % set constent part of objective, 0.5 * b'b
    
    % initialization
    i = 1;                                              % set iteration counter
    mu_finder = 100;                                    % initial mu
    objval_path = [];                                   % set path of objection value 
    objval_path(i) = 0.5 * y'*Q*y + c(mu)'*y + con; 


%% Step 2: Projection Gradient Descent

    while mu_finder > mu 
        mu_finder =  (mu_finder + mu) * 0.5;
        
        for step = 1:maxsteps 
 
            % step2.1: gradient decesent part
            grad_y = Q * y + c(mu_finder);
            y_dec = y - ss * grad_y;
            
            % step2.2: projection part, project y_dec onto [0,\infty)
            proj_y = max(y_dec, 0);
 
            % step2.3: Barzilai-Borwein: update stepsize
            dis_y = y - proj_y;   
            dis_x = dis_y(1:n) - dis_y(n+1:end);
            dAsd = dis_x' * A_square * dis_x;
            dd = dis_y' * dis_y;
            ss = min(maxss, max(minss, dd / dAsd));            
            
            % step2.4: update variable x and y
            y = proj_y;                                 
            x = y(1:n) - y(n+1:end); 
            
            % step2.5: update objective value path 
            i = i + 1;
%             objval_path(i)= 0.5 * y'*Q*y + mu*y + con ;
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
