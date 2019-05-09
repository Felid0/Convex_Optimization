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
%   p   <=> \min_x 0.5 y'y + (mu 1;mu 1)'(x+;x_) 
%            s.t.	A(x+ - x_) - b = y
%                   (x+;x_) >= 0
%
%       <=> \min_(x+;x_;y)  0.5 (x+;x_;y)'(O,O;O,I)(x+;x_;y) + (mu 1;mu 1;O)'(x+;x_;y) 
%            s.t. 	(A,-A,-I)'(x+;x_;y) = b 
%                   (x+;x_;y) >= (O;O;-inf)

%% Step 0: Initializztion & reformulation
   
    [m, n] = size(A);

    q = [zeros(2*n,2*n), zeros(2*n,m); zeros(m,2*n), eye(m)];	% set Q = (O,O; O,I)
    c = [mu * ones(2*n,1); zeros(m,1)];                         % set c = (mu 1; mu 1; O)  
    a = [A,-A,-eye(m)];                                         % set a = (A,-A,-I)

    % set constraints: (A,-A,-I)'(x+;x_;y) = b and (O;O;-inf)
    blc = b;                                    % (A,-A,-I)'(x+;x_;y) <= b
    buc = b;                                    % (A,-A,-I)'(x+;x_;y) >= b
    blx = [zeros(2*n,1);-inf* ones(m,1)];       % (x+;x_;y) >= (O;O;-inf)
    bux = [];                                   % no upper bounds for (x+;x_;y) 
   
%% Step 1: Call Mosek

    [res] = mskqpopt(q,c,a,blc,buc,blx,bux);
    
%% Step 2: Return optimal

    x = res.sol.itr.xx(1:n) - res.sol.itr.xx(n+1:2*n);
    out.optval = res.sol.itr.pobjval; 
    
    if res.sol.itr.solsta == 'OPTIMAL'
        out.status = 'Solved';
    else
        out.status = 'Failed';
    end
    
end