function [x,out]= l1_4_03_RMSProp(x0, A, b, mu, opts)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by RMSProp.
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
%           del:        parameter for numerical stabilization
%           rho:        decay rates
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
        del = 1e-6;
    end
    
  % set decay rate
    if isfield(opts, 'rho')         
        rho = opts.rho;              
    else
        rho = 5e-1;
    end
    
  % reformulation
    Q = (A'*A);                 % set Q = (A'A)
    c = -A'*b;                  % set c = -A'b

  % initialization
    x = x0;
    r = 0*x0;
    i = 1;                                              % set iteration counter
    mu_finder = 100;            % initial mu
    objval_path = [];           % set path of objection value 
    objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 

  % set proximal operator of mu ||x||_1
    prox_l1 = @(x,mu) sign(x).*max(0, abs(x) - mu);
    
    
%% Step 1: RMSProp  
if mode == "FISTA"
    %% "prox_FISTA" 0: Set hyperparameters 
    
        ss = 5e-2; 
        maxsteps = 50;
        tol = 1e-7; 
        mu_finder = mu_finder * 10;
        x_backward =x0;
        
    %% "prox_FISTA" 1: RMSProp
     
        while mu_finder > mu 
            mu_finder =  mu_finder /10;

            for step = 1:maxsteps

              % FISTA part & update variable y and x^(-1)
                y = x + (step - 2) / (step + 1.0) * (x - x_backward);
                x_backward = x;

              % gradient descent part
                grad_y = Q * y + c;            
                r = rho * r + (1 - rho) * grad_y .* grad_y;
                
                delta_y = ss./sqrt(del + r) .* grad_y;
                y = y - delta_y;             
                x = prox_l1(y, ss./sqrt(del + r) * mu_finder);  
            
                i = i + 1;
                objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 
                if abs(objval_path(i)-objval_path(i-1)) / objval_path(i-1) <= tol 
                    break;
                end
            end
            ss = ss / 10;  
        end
    
elseif  mode == "smooth" 
    %% "smooth" 0: Set hyperparameters 

        ss = 5e-2; 
        maxsteps = 250;
        tol = 1e-8; 
        sms = 1e-6;   % smooth parameter
 
        mu_finder = mu_finder * 10;
        
    %% "smooth" 1: RMSProp
     
        while mu_finder > mu 
            mu_finder =  mu_finder /10;

            for step = 1:maxsteps               
%                 grad_x = Q * x + c + mu_finder * sign(x);
                grad_x = Q * x + c + mu_finder * min(max(-1, x/sms), 1);
                r = rho * r + (1 - rho) * grad_x .* grad_x;
                delta_x = ss./sqrt(del + r) .* grad_x;
                x = x - delta_x;

                i = i + 1;
                objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 
                if abs(objval_path(i)-objval_path(i-1)) / objval_path(i-1) <= tol 
                    break;
                end
            end
            ss = ss / 10;  
        end      
    
else
    %% "prox" 0: Set hyperparameters 

        ss = 1e-1; 
        maxsteps = 200;
        tol = 1e-7; 
        mu_finder = mu_finder * 10;
        
    %% "prox" 1: RMSProp
     
        while mu_finder > mu 
            mu_finder =  mu_finder /10;
            
            for step = 1:maxsteps
                grad_x = Q * x + c ;
                r = rho * r + (1 - rho) * grad_x .* grad_x;
                delta_x = ss./sqrt(del + r) .* grad_x;
                y = x - delta_x;
                x = prox_l1(y, ss./sqrt(del + r) * mu_finder );

                i = i + 1;
                objval_path(i)= 0.5 * norm(A * x - b, 2)^2 + mu * norm(x, 1); 
                if abs(objval_path(i)-objval_path(i-1)) / objval_path(i-1) <= tol 
                    break;
                end
            end
            ss = ss / 10;         
        end
    
end

%% Step 2: Return optimal

%     x;
    out.optval = objval_path(i);
    out.itr = i;
    out.objval_path = objval_path;
    out.status = 'Solved';   
    
end