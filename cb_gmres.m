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

n = size(A,1);

vector_size = size(b);
x = x_init;
flag = false;
residual = A*x - b;
residual_norm = norm(residual);
b_norm = norm(b);
relres = residual_norm/b_norm;

iter = 0; % global iteration count
local_iter = 0; % Iteration count since the last reset

[residual_norm, residual_norm_collection, krylov_bases, next_krylov_basis] = initialize_2(residual, restart);

hessenberg = zeros(restart + 1, restart);
givens_sin = zeros(restart, 1);
givens_cos = zeros(restart, 1);

perform_reset = true;
%perform_reset = false;

while (true)
    if (iter == restart || perform_reset)
        % View not necessary because bounds are manually controlled
        %hessenberg_view = hessenberg(:, 1:local_iter);
        before_precond = step_2(residual_norm_collection, krylov_bases, hessenberg, local_iter);
        x = x + M * before_precond;
        residual = b - A*x;
        [residual_norm, residual_norm_collection, krylov_bases, next_krylov_basis] = initialize_2(residual, restart);
        local_iter = 0;
    end
    precond_vec = M * next_krylov_basis;
    
    % finish_arnoldi_CGS
    [next_krylov_basis, krylov_bases, hessenberg] = finish_arnoldi_CGS(next_krylov_basis, krylov_bases, hessenberg, local_iter);
    % givens_rotation
    % calculate_next_residual_norm
    %break
    iter = iter + 1;
    local_iter = local_iter + 1;
end
end


function before_precond = step_2(residual_norm_collection, krylov_bases, hessenberg, local_iter)
%TODO write in simpler terms!
%solve_upper_triangular
y = zeros(local_iter, 1);
for i=local_iter:-1:1 % +1 since indices start at 1
    tmp = residual_norm_collection(i);
    for j=i+1:local_iter-1
        tmp = tmp - hessenberg(i, j) * y(j);
    end
    y(i) = tmp / hessenberg(i, i);
end

%calculate_qy
before_precond = krylov_bases(:, 1:local_iter) * y;
% before_precond = zeros(size(krylov_bases, 1), 1);
% for i=1:size(before_precond,1)
%     for j=1:local_iter
%         before_precond(i) = before_precond(i) + krylov_bases(i, j) * y(j);
%     end
% end
end

function [residual_norm, residual_norm_collection, krylov_bases, next_krylov_basis] = initialize_2(residual, restart)
n = size(residual, 1);
residual_norm = norm(residual);
residual_norm_collection = zeros(restart + 1, 1);
residual_norm_collection(1) = residual_norm;
krylov_bases = zeros(n, restart+1);  % Transposed from Ginkgo storage
next_krylov_basis = (1/residual_norm) * residual;
end

function [next_krylov_basis, krylov_bases, hessenberg] = finish_arnoldi_CGS(next_krylov_basis, krylov_bases, hessenberg, local_iter)
eta = 1/sqrt(2);
old_arnoldi_norm = eta * norm(next_krylov_basis);
hessenberg_iter = zeros(local_iter + 1, 1);
hessenberg_iter = transpose(krylov_bases(:, 1:local_iter+1)) * next_krylov_basis;

next_krylov_basis = next_krylov_basis - krylov_bases(:, 1:local_iter+1) * hessenberg_iter;
arnoldi_norm = norm(next_krylov_basis);

% TODO add re-orthogonalization

arnoldi_norm = norm(next_krylov_basis);
hessenberg(1:local_iter+1, local_iter + 1) = hessenberg_iter;
hessenberg(local_iter + 2, local_iter + 1) = arnoldi_norm;
next_krylov_basis = (1/arnoldi_norm) * next_krylov_basis;
krylov_bases(:, local_iter + 2) = next_krylov_basis;
end