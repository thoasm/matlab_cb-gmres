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
residual = b - A*x;
residual_norm = norm(residual);
b_norm = norm(b);

iter = 0; % global iteration count
local_iter = 0; % Iteration count since the last reset

[residual_norm, residual_norm_collection, krylov_bases, next_krylov_basis] = initialize_2(residual, restart);
relres = residual_norm/b_norm;

hessenberg = zeros(restart + 1, restart);
givens_sin = zeros(restart, 1);
givens_cos = zeros(restart, 1);

%perform_reset = true;
perform_reset = false;

while (true)
    if (relres < tol)
        flag = true;
        break;
    elseif (iter > maxit)
        flag = false;
        break;
    end

    if (local_iter == restart || perform_reset)
        % View not necessary because bounds are manually controlled
        %hessenberg_view = hessenberg(:, 1:local_iter);
        before_precond = ...
            step_2(residual_norm_collection, krylov_bases, hessenberg, local_iter);
        x = x + M * before_precond;
        residual = b - A*x;
        [residual_norm, residual_norm_collection, krylov_bases, ...
            next_krylov_basis] = initialize_2(residual, restart);
        local_iter = 0;
    end
    %precond_vec = M * next_krylov_basis;
    %next_krylov_basis = A * precond_vec;
    next_krylov_basis = A * M * next_krylov_basis;
    
    % finish_arnoldi_CGS
    [next_krylov_basis, krylov_bases, hessenberg] = ...
        finish_arnoldi_CGS(next_krylov_basis, krylov_bases, hessenberg, local_iter);
    % givens_rotation
    [givens_sin, givens_cos, hessenberg] = ...
        givens_rotation(givens_sin, givens_cos, hessenberg, local_iter);
    % calculate_next_residual_norm
    residual_norm_collection(local_iter+2) = ...
        -givens_sin(local_iter+1) * residual_norm_collection(local_iter+1);
    residual_norm_collection(local_iter+1) = ...
        givens_cos(local_iter+1) * residual_norm_collection(local_iter+1);
    residual_norm = abs(residual_norm_collection(local_iter+2));
    relres = residual_norm / b_norm;

%     print_matrix("Iteration:", local_iter)
%     print_matrix("Hessenberg:", hessenberg)
%     print_matrix("krylov_bases:", krylov_bases)
%     print_matrix("next_krylov_basis:", next_krylov_basis)
%     print_matrix("residual_norm:", residual_norm)
    %break
    iter = iter + 1;
    local_iter = local_iter + 1;
end

before_precond = ...
    step_2(residual_norm_collection, krylov_bases, hessenberg, local_iter);
x = x + M * before_precond;

end


function before_precond = step_2(residual_norm_collection, krylov_bases, hessenberg, local_iter)
%solve_upper_triangular
y = hessenberg(1:local_iter, 1:local_iter) \ residual_norm_collection(1:local_iter);

%calculate_qy
before_precond = krylov_bases(:, 1:local_iter) * y;
end


function [residual_norm, residual_norm_collection, krylov_bases, next_krylov_basis] = ...
    initialize_2(residual, restart)
n = size(residual, 1);
residual_norm = norm(residual);
residual_norm_collection = zeros(restart + 1, 1);
residual_norm_collection(1) = residual_norm;
krylov_bases = zeros(n, restart+1);  % Transposed from Ginkgo storage
krylov_bases(:, 1) = (1/residual_norm) *residual;
next_krylov_basis = (1/residual_norm) * residual;
end


function [next_krylov_basis, krylov_bases, hessenberg] = ...
    finish_arnoldi_CGS(next_krylov_basis, krylov_bases, hessenberg, local_iter)
eta = 1/sqrt(2);
old_arnoldi_norm = eta * norm(next_krylov_basis);
%hessenberg_iter = zeros(local_iter + 1, 1);
hessenberg_iter = transpose(krylov_bases(:, 1:local_iter+1)) * next_krylov_basis;

%print_matrix("next_krylov_basis before re-compute:", next_krylov_basis)
%print_matrix("krylov_bases before:", krylov_bases)
%print_matrix("hessenberg_iter before:", hessenberg_iter)

next_krylov_basis = next_krylov_basis - krylov_bases(:, 1:local_iter+1) * hessenberg_iter;
% for k = 1:local_iter+1
%     for j=1:size(next_krylov_basis,1)
%         next_krylov_basis(j) = next_krylov_basis(j) - hessenberg_iter(k) * krylov_bases(j, k);
%     end
% end

%print_matrix("next_krylov_basis after re-compute:", next_krylov_basis)
arnoldi_norm = norm(next_krylov_basis);

% Our re-orthogonalization in this loop
for l=1:3
    if (arnoldi_norm < old_arnoldi_norm)
        break;
    end
    %print_matrix("re-orthogonalization", l)
    old_arnoldi_norm = eta * arnoldi_norm;
    %buffer = zeros(local_iter+1, 1);
    buffer = transpose(krylov_bases(:, 1:local_iter+1)) * next_krylov_basis;
    next_krylov_basis = next_krylov_basis - krylov_bases(:, 1:local_iter+1) * buffer;
    hessenberg_iter = hessenberg_iter + buffer;
    arnoldi_norm = norm(next_krylov_basis);
end

hessenberg(1:local_iter+1, local_iter + 1) = hessenberg_iter;
hessenberg(local_iter + 2, local_iter + 1) = arnoldi_norm;
next_krylov_basis = (1/arnoldi_norm) * next_krylov_basis;
krylov_bases(:, local_iter + 2) = next_krylov_basis;
end


function [givens_sin, givens_cos, hessenberg] = ...
    givens_rotation(givens_sin, givens_cos, hessenberg, local_iter)
hessenberg_iter = hessenberg(1:local_iter+2, local_iter+1);

for j=1:local_iter
    temp = givens_cos(j) * hessenberg_iter(j) + givens_sin(j) * hessenberg_iter(j+1);
    hessenberg_iter(j+1) = -givens_sin(j) * hessenberg_iter(j) +...
                          givens_cos(j) * hessenberg_iter(j+1);
    hessenberg_iter(j) = temp;
end

%print_matrix("hessenberg mid givens_rotation:", hessenberg_iter)
%print_matrix("givens_sin mid givens_rotation:", givens_sin)
%print_matrix("givens_cos mid givens_rotation:", givens_cos)

% calculate_sin_and_cos
if (hessenberg_iter(local_iter+1) == 0)
    givens_cos(local_iter+1) = 0;
    givens_sin(local_iter+1) = 1;
else
    this_hess = hessenberg_iter(local_iter+1);
    next_hess = hessenberg_iter(local_iter+2);
    scale = abs(this_hess) + abs(next_hess);
    hypotenuse = scale * sqrt(abs(this_hess / scale) * abs(this_hess / scale) +...
        abs(next_hess / scale) * abs(next_hess / scale));
    givens_cos(local_iter+1) = this_hess / hypotenuse;
    givens_sin(local_iter+1) = next_hess / hypotenuse;
end

hessenberg_iter(local_iter+1) = ...
    givens_cos(local_iter+1) * hessenberg_iter(local_iter+1) +...
    givens_sin(local_iter+1) * hessenberg_iter(local_iter+2);
hessenberg_iter(local_iter+2) = 0;

%print_matrix("hessenberg end givens_rotation:", hessenberg_iter)
%print_matrix("givens_sin end givens_rotation:", givens_sin)
%print_matrix("givens_cos end givens_rotation:", givens_cos)

hessenberg(1:local_iter+2, local_iter+1) = hessenberg_iter;
end

function print_matrix(str, mtx)
    fprintf("%s\n", str)
    disp(mtx)
end