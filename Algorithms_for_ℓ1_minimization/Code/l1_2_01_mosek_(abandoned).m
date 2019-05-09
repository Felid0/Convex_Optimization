function [x, out] = l1_2_01_mosek(x0, A, b, mu, opts)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by calling mosek directly.
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
%       <=>:\min_(x;t) 0.5 (x;t)'Q(x;t) + c'(x;t) 
%            s.t. a(x;t) <= 0

%% Step 0: Initializztion & reformulation

    [~, n] = size(A);
    
    % set Q = (A'A,O;O,O) 
    q_zero = zeros(n,n);
    q = [(A'*A), q_zero; q_zero, q_zero]; 
    
    % set c = (-A'b;mu 1)
    c = [-A'*b; mu * ones(n,1)];    
    
    % set a = (I,-I;-I,-I)
    a_I = eye(n);
    a = [a_I, -a_I; -a_I, -a_I];  
    
    % set constraints: bound a(x;t) and (x;t)
    blc = [];               % no lower bounds for a(x;t)
    buc = zeros(2 * n, 1);  % upper bounds for a(x;t)
    blx = [];               % no lower bounds for (x;t)
    bux = [];               % no upper bounds for (x;t)
    
%% Step 1: Call Mosek

    [res] = mskqpopt(q,c,a,blc,buc,blx,bux);
    
%% Step 2: Return optimal

    x = res.sol.itr.xx(1:n);
    out.optval = res.sol.itr.pobjval + 0.5 * (b' * b);
    
    if res.sol.itr.solsta == 'OPTIMAL'
        out.status = 'Solved';
    else
        out.status = 'Failed';
    end
end