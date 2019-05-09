function [x, out] = l1_2_02_gurobi(x0, A, b, mu, opts)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by calling gurobi directly.
% 
%       out = \min_x 0.5 ||Ax - b||_2^2 + mu ||x||_1
%         x = \arg\min_x 0.5 ||Ax - b||_2^2 + mu ||x||_1
% 
%   Parameters:
%       x0:     m by 1 matrix
%       A:      m by n matrix
%       b:      m by 1 matrix
%       mu:     scalar
%       opts:   (options)
% 
%   Returns:
%       x: m by 1 matrix  i.e. the optimal point
%       out: optimal value of objective
%
%   ================== Reformulation =============================
%   p   <=> \min_x 0.5 x'(A'A)x + (-b'A)x + 0.5 (b'b) + mu ||x||_1
%
%       <=> \min_(x;t) 0.5 (x;t)'(A'A,O;O,O)(x;t) + (-A'b;mu 1)'(x;t) 
%            s.t.  (I,-I;-I,-I)(x;t) <= 0
%
%       <=>:\min_(x;t) (x;t)'(Q/2)(x;t) + c'(x;t) 
%            s.t. -inf < a(x;t) <= 0

%% Step 0: Initialization & reformulation model

    clear model;
    [~, n] = size(A);
    
    % reformulation
    q_zero = zeros(n,n);
    q = [A'*A, q_zero; q_zero, q_zero];     % set q = (A'A,O;O,O) 
    a_I = eye(n);    
    a = [a_I, -a_I; -a_I, -a_I];            % set a = (I,-I;-I,-I) 
    c = [-(A'*b); mu * ones(n,1)];          % set c = (-A'b;mu 1)
    con = (b' * b)/2;                       % set constent part of objective b'b/2
    
    % NOTATION! : gurobi model is (x;t)'(Q/2)(x;t) + c'(x;t)
    model.Q = sparse(q/2);
    model.A = sparse(a);    
    model.obj = c;
    model.objcon = con; 
    
    % set constraints -inf < a(x;t) <= 0
    model.rhs = zeros(2 * n, 1);            % RHS of a(x;t) <=  0
    model.sense = '<';                      % >= of a(x;t) <= 0  
    model.lb = - inf * ones(2 * n , 1);     % lower bound of  a(x;t). 
    
%% Step 1: Call gurobi

    result = gurobi(model);
    
%% Step 2: Return optimal

    x = result.x(1:n);
    out.optval = result.objval;
    
    if result.status == 'OPTIMAL'
        out.status = 'Solved';
    else
        out.status = 'Failed';
    end   
end