function [x,out]= lp_2_03_RSSN(c, A, b, opts, x0)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by RSSN through solving the dual problem.
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

    if isfield(opts, 'lambda')          
        lambda = opts.lambda;
    else
        lambda = 5;              
    end
        
%     if isfield(opts, 'gamma0')
%         gamma0 = opts.gamma0;
%     else
%         gamma0 = 0.6;
%     end  

    if isfield(opts, 'gamma1')
        gamma1 = opts.gamma1;
    else
        gamma1 = 1.1;
    end
    
    if isfield(opts, 'gamma2')
        gamma2 = opts.gamma2;
    else
        gamma2 = 1.5;
    end
    
    if isfield(opts, 'a_1')
        a_1 = opts.a_1;
    else
        a_1 = 0.1;
    end
    
    if isfield(opts, 'a_2')
        a_2 = opts.a_2;
    else
        a_2 = 0.8;
    end
    
    if isfield(opts, 'v')
        v = opts.v;
    else
        v = 0.4;
    end
    
    if isfield(opts, 'maxsteps')   
        maxsteps = opts.maxsteps;
    else
        maxsteps = 3000;
    end
    
    if isfield(opts, 'err1')
        tol = opts.tol;
    else
        tol = 1e-8;
    end

    if isfield(opts, 'kappa_line')
        kappa_line = opts.kappa_line;
    else
        kappa_line = 1e-6;
    end
    
%     if isfield(opts, 'kappa_bar')
%         kappa_bar = opts.kappa_bar;
%     else
%         kappa_bar = 1e3;
%     end 
    
    if isfield(opts, 'kappa')
        kappa = opts.kappa;
    else
        kappa = 0.05;
    end
    
%% Step 1: Initialization

    [~, n] = size(A);

    AIA = A'*((A*A')\A);
    AIb = A'*((A*A')\b);
    P = 2*AIA - eye(n) ;
    Q = eye(n) - AIA;


    F = @(z) P*max(0, z-lambda*c) + Q*z - AIb;
    J = @(z) P*diag(z-lambda*c > 0) + Q;

    z = zeros(n, 1);
    u_bar = z;
    Fu_bar = F(u_bar);
    
    x = x0;
    i = 1;  
    objval_path = [];
    objval_path(i) = c'*x; 
    
%% Step 2: RSSN

    for i = 1:maxsteps

        Fz = F(z);
        mu = kappa * norm(Fz);
        d = (J(z) + mu * eye(n)) \ (-Fz);

        norm_d = norm(d);
        if norm_d < tol
            break;
        end

        u = z + d;
        Fu = F(u);
        Fud = -Fu'*d;
        rho = Fud / norm_d ^ 2;

        if rho >= a_1
            if norm(Fu) <= v * norm(Fu_bar)
                z = u; 
                Fu_bar = Fu;
            else
                z = z - Fud / norm(Fu)^2 * Fu;
            end
        end

        if rho >= a_2
%             kappa = max(kappa_line, gamma0*kappa);
              kappa = 0.5*(kappa_line + kappa) ;          
        elseif rho >= a_1
%             kappa = gamma1 * kappa;  
            kappa = 0.5*(1 + gamma1) * kappa;  
        else
%             kappa = min(kappa_bar, gamma2*kappa);
            kappa = 0.5*(gamma1 + gamma2) * kappa;
        end

        objval_path(i) = c'*max(0, z-lambda*c);
    end

%% Step 3: Return optimal

    x = max(z - lambda*c, 0); 
    out.optval = c'*x;
    out.objval_path = objval_path;
    out.itr = i;
    out.status = 'Solved';  
    
end