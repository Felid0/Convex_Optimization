function [x, out] = lp_0_02_cvx_gurobi(c, A, b, opts, x0)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by using cvx calling gurobi.
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

%% Step 0: Initialize

    [m, n] = size(A);

%% Step 1: cvx calling gurobi

    cvx_begin 
        cvx_solver gurobi
        variable x(n) 
        minimize(c'*x) 
        subject to
            A * x == b;
            x >= 0;
    cvx_end

%% Step 2: Return optimal

    x = x;
    out.optval = cvx_optval;
    
    if cvx_status == 'Solved'
        out.status = 'Solved';
    else
        out.status = 'Failed';
    end
end