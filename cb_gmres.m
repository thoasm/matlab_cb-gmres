% Change format to show more digits: `format long`
% Show current format: `fmt = format`
% `single(x)` cast x to single precision
% `double(x)` cast x to double precision
% both work the same when dealing with input vectors

function [x,flag,relres,iter] = cb_gmres(A, b, x_init, restart, tol, maxit, M)
%CB_GMRES Summary of this function goes here
%   Detailed explanation goes here
%   maxit counts each iteration, not just the number of restarts

if (nargin < 2)
    error("Not enough Input arguments!");
elseif (size(A,1) ~= size(A,2))
    error("Matrix needs to be square");
elseif (size(A,2) ~= size(b,1))
    error("b must have size(A, 2) rows!");
end

if (nargin < 3)
    x_init = b;
elseif (size(x_init) ~= size(b))
    error("x_init and b must have the same size!");
end
if (nargin < 4)
    restart = size(A, 1);
elseif (restart > size(A, 1))
    error("Restart must not be larger than the number of rows of A!");
end
if (nargin < 5)
    tol = 1e-6;
end
if (nargin < 6)
    maxit = restart;
end
if (nargin < 7)
    M = eye(size(A));
end

if (size(M) ~= size(A))
    error("A and M must have same size!");
end
if (size(x_init) ~= size(b))
    error("A and M must have same size!");
end
if (size(b, 2) ~= 1)
    error("Only a single right hand side is supported!");
end

vector_size = size(b);
x = x_init;
flag = False;
residual = A*x - b;
residual_norm = norm(residual);
b_norm = norm(b);
relres = residual_norm/b_norm;
iter = 0;

residual_norm_collection = zeros(restart+1, 1);
residual_norm_collection(1) = residual_norm;
krylov_bases = zeros(restart+1, vector_size(1));
y = zeros(restart, 1);
next_krylov_basis = zeros(size(b)); % Later turned to lower precision
hessenberg = zeros(restart + 1, restart);
givens_sin = zeros(restart, 1);
givens_cos = zeros(restart, 1);
arnoldi_norm = 0.0;

end
