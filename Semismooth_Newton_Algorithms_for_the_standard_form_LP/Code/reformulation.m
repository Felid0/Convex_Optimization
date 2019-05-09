function out = reformulation(in)
%% Readme: 
% This fucntion is defined to reformulate the problem 
% 
%       min   c'*x    
%       st.   bl <= A*x <= bu 
%             tl <=  x  <= tu
% 
% to
%       min c'*xp-c'*xn
%       st. A(xp-x2)+su = bu
%           A(xp-x2)-sl = bl
%           xp-xn +au = tu
%           xp-xn -al = tl
%           xp,xn,su,sl,au,al >= 0

%% Reformulation

    [m,n] = size(in.A);
    
    b = [in.rhs ; in.lhs ; in.ub ; in.lb];
    
    A = [sparse(in.A),  -sparse(in.A),  sparse(eye(m)),     sparse(m, m + 2 * n);
         sparse(in.A),  -sparse(in.A),  sparse(m, m),       -sparse(eye(m)),      sparse(m, 2 * n);
         speye(n),      -speye(n),      sparse(n, 2 * m),                   sparse(eye(n)),       sparse(n, n);
         speye(n),      -speye(n),      sparse(n, 2 * m),   sparse(n, n),                   -sparse(eye(n))];
     
%% Return out

    out.A = A(abs(b) ~= Inf, :);
    out.b = b(abs(b) ~= Inf);
    out.c = [in.obj; -in.obj; sparse(2 * m + 2 * n, 1)];
    
end