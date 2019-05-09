function [x,out]= l1_4_02_Adam(x0, A, b, mu, opts)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by Adam.
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
      
  % set parameter for numerical stabilization
    if isfield(opts, 'del') 
        del = opts.del;              
    else
        del = 1e-8;
    end
    
  % set exponential decay rate for 1st moment estimate
    if isfield(opts, 'rho1')         
        rho1 = opts.rho1;              
    else
        rho1 = 1 - 1e-1;
    end
    
  % set exponential decay rate for 2nd moment estimate
    if isfield(opts, 'rho2') 
        rho2 = opts.rho2;              
    else
        rho2 = 1 - 1e-3;
    end
    
  % reformulation
    Q = (A'*A);                 % set Q = (A'A)
    c = -A'*b;                  % set c = -A'b

  % initialization
    x = x0;
    r = 0*x0;
    s = 0*x0;
    i = 1;                      % set iteration counter
    mu_finder = 100;            % initial mu
    objval_path = [];           % set path of objection value 
    objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 

  % set proximal operator of mu ||x||_1
    prox_l1 = @(x,mu) sign(x).*max(0, abs(x) - mu);

%% Step 1: Adam  
if mode == "FISTA"
    %% "prox_FISTA" 0: Set hyperparameters 
    
        ss = 5e-1; 
        maxsteps = 50;
        tol = 1e-6; 
        
    %% "prox_FISTA" 1: Adam
    
        x_backward = x0;
        while mu_finder > mu  + 1e-9
            mu_finder =  (mu_finder* 0.5 + mu* 0.5) ;

            for step = 1:maxsteps

              % FISTA part & update variable y and x^(-1)
                y = x + (step - 2) / (step + 1.0) * (x - x_backward);
                x_backward = x;

              % gradient descent part
                grad_y = Q * y + c;
                                
                s = rho1 * s + (1 - rho1) * grad_y; 
                r = rho2 * r + (1 - rho2) * grad_y .* grad_y;
                s_hat = s / (1 - rho1^i);               % 1st moment unbiased estimate
                r_hat = r / (1 - rho2^i);               % 2ns moment unbiased estimate
                
                delta_y = ss./(del + sqrt(r_hat)) .* s_hat;
                y = y - delta_y;             
                x = prox_l1(y, ss./(del + sqrt(r_hat)) * mu_finder);  
                
                i = i + 1;
                objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 
                if abs(objval_path(i)-objval_path(i-1)) / objval_path(i-1) <= tol 
                    break;
                end
            end
        end
    
elseif  mode == "smooth" 
    %% "smooth" 0: Set hyperparameters 

        ss = 3e-2; 
        maxsteps = 100;
        tol = 1e-7; 
        sms = 1e-6;

    %% "smooth" 1: Adam

        while mu_finder > mu 
            mu_finder =  (mu_finder + mu) * 0.5;

            for step = 1:maxsteps   
%                 grad_x = Q * x + c + mu_finder * sign(x);
                grad_x = Q * x + c + mu_finder * min(max(-1, x/sms), 1);
                s = rho1 * s + (1 - rho1) * grad_x; 
                r = rho2 * r + (1 - rho2) * grad_x .* grad_x;
                s_hat = s / (1 - rho1^i);               % 1st moment unbiased estimate
                r_hat = r / (1 - rho2^i);               % 2ns moment unbiased estimate
                
                delta_x = ss./(del + sqrt(r_hat)) .* s_hat;
                x = x - delta_x;      
                
                i = i + 1;
                objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 
                if abs(objval_path(i)-objval_path(i-1)) / objval_path(i-1) <= tol 
                    break;
                end
            end
        end
           
else
    %% "prox" 0: Set hyperparameters 

        ss = 3e-2; 
        maxsteps = 100;
        tol = 1e-7; 

    %% "prox" 1: Adam

        while mu_finder > mu 
            mu_finder =  (mu_finder + mu) * 0.5;

            for step = 1:maxsteps

                grad_x = Q * x + c ;

                s = rho1 * s + (1 - rho1) * grad_x; 
                r = rho2 * r + (1 - rho2) * grad_x .* grad_x;
                s_hat = s / (1 - rho1^i);               % 1st moment unbiased estimate
                r_hat = r / (1 - rho2^i);               % 2ns moment unbiased estimate
                
                delta_x = ss./(del + sqrt(r_hat)) .* s_hat;
                y = x - delta_x;
                y_prox = prox_l1(y, ss./(del + sqrt(r_hat)) * mu_finder);
                x = y_prox - delta_x;

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