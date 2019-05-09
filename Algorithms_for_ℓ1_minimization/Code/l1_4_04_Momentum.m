function [x,out]= l1_4_04_Momentum(x0, A, b, mu, opts)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by Momentum method.
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
%           mode:       "FISTA","smooth","prox"(default)
%           alpha:      momentum parameter 
% 
%   Returns:
%       x: m by 1 matrix  i.e. the optimal point
%       out: optimal value of objective

%% Step 0: Set mode & Initialization

  % set mode
    if isfield(opts, 'mode')          
        mode = opts.mode;       
    else 
        mode = "prox"; 
    end
    
  % momentum parameter 
    if isfield(opts, 'alpha') 
        alpha = opts.alpha;              
    else
        alpha = 6e-1;
    end
    
  % reformulation
    Q = (A'*A);                 % set Q = (A'A)
    c = -A'*b;                  % set c = -A'b

  % initialization
    x = x0;
    v = 0*x0;
    i = 1;                                              % set iteration counter
    mu_finder = 100;            % initial mu
    objval_path = [];           % set path of objection value 
    objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 

  % set proximal operator of mu ||x||_1
    prox_l1 = @(x,mu) sign(x).*max(0, abs(x) - mu);
    
    
%% Step 1: Momentum  
if mode == "FISTA"
    %% "prox_FISTA" 0: Set hyperparameters 
    
        ss = 5e-4; 
        maxsteps = 20;
        tol = 1e-7;       
        
    %% "prox_FISTA" 1: Momentum
    
        x_backward = x0;

        while mu_finder > mu 
            mu_finder =  (mu_finder + mu) * 0.5;

            for step = 1:maxsteps

                y = x + (step - 2) / (step + 1.0) * (x - x_backward);
                x_backward = x;

                grad_y = Q * y + c;
                v = alpha * v - ss * grad_y;            
                x = prox_l1(y + v, ss * mu_finder);
                
                i = i + 1;
                objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 
                if abs(objval_path(i)-objval_path(i-1)) / objval_path(i-1) <= tol 
                    break;
                end
            end
        end
    
elseif  mode == "smooth" 
    %% "smooth" 0: Set hyperparameters 

        ss = 5e-4; 
        maxsteps = 80;
        tol = 1e-7; 
        sms = 1e-6;     % smooth parameter

    %% "smooth" 1: Momentum

        while mu_finder > mu 
            mu_finder =  (mu_finder + mu) * 0.5;

            for step = 1:maxsteps   
%                 grad_x = Q * x + c + mu_finder * sign(x);
                grad_x = Q * x + c + mu_finder * min(max(-1, x/sms), 1);
                v = alpha * v - ss * grad_x;
                x = x + v;
                
                i = i + 1;
                objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 
                if abs(objval_path(i)-objval_path(i-1)) / objval_path(i-1) <= tol 
                    break;
                end
            end
        end
           
else
    %% "prox" 0: Set hyperparameters 

        ss = 5e-4; 
        maxsteps = 50;
        tol = 1e-7; 
        
    %% "prox" 1: Momentum

        while mu_finder > mu 
            mu_finder =  (mu_finder + mu) * 0.5;

            for step = 1:maxsteps
                
                grad_x = Q * x + c ;
                v = alpha * v - ss * grad_x;
                y = x + v;
                x = prox_l1(y, ss * mu_finder );

                i = i + 1;
                objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 
                if abs(objval_path(i)-objval_path(i-1)) / objval_path(i-1) <= tol 
                    break;
                end
            end
        end
    
end

%% Step 2: Return optimal

%     x;
    out.optval = objval_path(i);
    out.itr = i;
    out.objval_path = objval_path;
    out.status = 'Solved';   
    
end