function [x, out] = lp_0_02_gurobi(c, A, b, opts, x0)
%% Readme:
%   This fucntion is defined to solve the problem below 
%   by calling gurobi directly.
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
% 
%   ============== Reformulation ==============
%   p   <=>     \min_x c'x
%                s.t.  (A;-A;I)'(x;x;x) >= (b;-b;0)

%% Step 0: Initialization & reformulation model
      
    clear model;
    [m, n] = size(A);
    
    model.modelsense = 'Min';
    model.A = sparse([A;-A;eye(n)]); 
    model.rhs = [b; -b; zeros(n,1)]';       % RHS of (A;-A;I)'(x;x;x) >= (b;-b;0)
    model.obj = c';
    model.sense = '>';                      % >= of (A;-A;I)'(x;x;x) >= (b;-b;0)
 
    
%% Step 1: Call gurobi

    result = gurobi(model);
    
%% Step 2: Return optimal

    x = result.x;
    out.optval = result.objval;
    
    if result.status == 'OPTIMAL'
        out.status = 'Solved';
    else
        out.status = 'Failed';
    end   
end