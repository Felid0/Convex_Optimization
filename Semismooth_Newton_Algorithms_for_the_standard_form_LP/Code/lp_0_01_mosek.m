function [x, out] = lp_0_01_mosek(c, A, b, opts, x0)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by calling mosek directly.
% 
%       \min_x c'x
%        s.t.  Ax = b
%              x >= 0
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

%% Step 0: Initializztion 
   
    [m, n] = size(A);
    
    % set constraints: Ax = b and x >= O
    blc = b;              % Ax <= b
    buc = b;              % Ax >= b
    blx = zeros(n,1);     % x >= O
    bux = [];             % no upper bounds for x 
   
%% Step 1: Call Mosek

    [res] = msklpopt(c, A, blc, buc, blx, bux);
    
%% Step 2: Return optimal

    x = res.sol.itr.xx;
    out.optval = res.sol.itr.pobjval; 
    
    if res.sol.itr.solsta == 'OPTIMAL'
        out.status = 'Solved';
    else
        out.status = 'Failed';
    end
    
end