function [x, out] = l1_1_01_cvx_mosek(x0, A, b, mu, opts)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by using cvx calling mosek.
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
%       x:      m by 1 matrix  i.e. the optimal point
%       out:    optimal value of objective

%% Step 0: Initialize

    [m, n] = size(A);

%% Step 1: cvx calling mosek

    cvx_begin
        cvx_solver mosek
        variable x(n)
        minimize(0.5 * (A * x - b)' * (A * x - b) + mu * norm(x,1))  
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