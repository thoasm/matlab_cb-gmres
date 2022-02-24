function [x,flag,relres,iter] = cb_gmres(A, b, x_init, restart, tol, maxit, reduce_precision, M)
%CB_GMRES   Compressed-Basis GMRES (Generalized Minimum Residual Method)
%   X = CB_GMRES(A,B) attempts to solve the system of linear equations
%   A*X=B for X. The N-by-N coefficient matrix A must be square and the
%   right hand side column vector B must have length N and a single right
%   hand side. This uses the unrestarted method with at most N total
%   iterations.
%
%   X = CB_GMRES(A,B,X_INIT) specifies the initial value of X. Default
%   value of X is B (the right hand side).
%
%   X = CB_GMRES(A,B,X_INIT,RESTART) restarts the method every RESTART
%   iterations. Default value is N.
%
%   X = CB_GMRES(A,B,X_INIT,RESTART,TOL) specifies the tolerance of the
%   method. Default value is 1e-7.
%
%   X = CB_GMRES(A,B,X_INIT,RESTART,TOL,MAXIT) specifies the maximum number
%   of total iterations (every iteration counts, including inner
%   iterations). Default value is RESTART.
%
%   X = CB_GMRES(A,B,X_INIT,RESTART,TOL,MAXIT,REDUCE_PRECISION) specifies
%   weather the Krylov bases should be stored in reduced precision. Default
%   is true.
%
%   X = CB_GMRES(A,B,X_INIT,RESTART,TOL,MAXIT,REDUCE_PRECISION, M)
%   specifies the preconditioner M used to effectively solve the system
%   inv(M)*A*x = inv(M)*B for X. Default value is eye(size(A)).
%
%   [X,FLAG] = CB_GMRES(A,B,...) also returns a boolean indicating if the
%   tolerance was achieved (true), or not (false)
%
%   [X,FLAG,RELRES] = CB_GMRES(A,B,...) also returns the relative residual
%   NORM(B-A*X)/NORM(B).
%
%   [X,FLAG,RELRES,ITER] = CB_GMRES(A,B,...) also returns the total number
%   of iterations the alrorithm was run.
%
%   This is a Matlab implementation of the CB-GMRES algorithm used in the
%   sparse linear algebra library Ginkgo. If performance is your priority,
%   run the code directly with Ginkgo on a CPU or even a GPU. For more
%   information, head to https://ginkgo-project.github.io/
%   A CB-GMRES example can be seen here:
%   https://github.com/ginkgo-project/ginkgo/blob/develop/examples/cb-gmres/cb-gmres.cpp


if (nargin < 2)
    error("Not enough Input arguments!");
elseif (size(A,1) ~= size(A,2))
    error("Matrix needs to be square");
elseif (size(A,2) ~= size(b,1))
    error("b must have size(A, 2) rows!");
end

if (nargin < 3 || isempty(x_init))
    x_init = b;
elseif (size(x_init) ~= size(b))
    error("x_init and b must have the same size!");
end
if (nargin < 4 || isempty(restart))
    restart = size(A, 1);
elseif (restart > size(A, 1))
    error("Restart must not be larger than the number of rows of A!");
end
if (nargin < 5 || isempty(tol))
    tol = 1e-6;
end
if (nargin < 6 || isempty(maxit))
    maxit = restart;
end
if (nargin < 7 || isempty(reduce_precision))
    reduce_precision = true;
elseif (~islogical(reduce_precision))
    error("reduce_precision must be a boolean!");
end
if (nargin < 8)
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

[residual_norm, residual_norm_collection, krylov_bases, next_krylov_basis] = ...
    initialize_2(residual, restart, reduce_precision);
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
        before_precond = ...
            step_2(residual_norm_collection, krylov_bases, hessenberg, local_iter);
        x = x + M * before_precond;
        residual = b - A*x;
        [residual_norm, residual_norm_collection, krylov_bases, ...
            next_krylov_basis] = initialize_2(residual, restart, reduce_precision);
        local_iter = 0;
    end
    next_krylov_basis = A * M * next_krylov_basis;
    
    % finish_arnoldi_CGS
    [next_krylov_basis, krylov_bases, hessenberg] = ...
        finish_arnoldi_CGS(next_krylov_basis, krylov_bases, hessenberg, local_iter, reduce_precision);
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
    initialize_2(residual, restart, reduce_precision)

    n = size(residual, 1);
    residual_norm = norm(residual);
    residual_norm_collection = zeros(restart + 1, 1);
    residual_norm_collection(1) = residual_norm;
    krylov_bases = zeros(n, restart+1);  % Transposed from Ginkgo storage
    if (reduce_precision)
        krylov_bases(:, 1) = single((1/residual_norm) *residual);
    else
        krylov_bases(:, 1) = (1/residual_norm) *residual;
    end
    next_krylov_basis = (1/residual_norm) * residual;
end


function [next_krylov_basis, krylov_bases, hessenberg] = ...
    finish_arnoldi_CGS(next_krylov_basis, krylov_bases, hessenberg, local_iter, reduce_precision)

    eta = 1/sqrt(2);
    old_arnoldi_norm = eta * norm(next_krylov_basis);
    hessenberg_iter = transpose(krylov_bases(:, 1:local_iter+1)) * next_krylov_basis;
    
    
    next_krylov_basis = next_krylov_basis - krylov_bases(:, 1:local_iter+1) * hessenberg_iter;
    arnoldi_norm = norm(next_krylov_basis);
    
    % Our re-orthogonalization in this loop
    for l=1:3
        if (arnoldi_norm < old_arnoldi_norm)
            break;
        end
        old_arnoldi_norm = eta * arnoldi_norm;
        buffer = transpose(krylov_bases(:, 1:local_iter+1)) * next_krylov_basis;
        next_krylov_basis = next_krylov_basis - krylov_bases(:, 1:local_iter+1) * buffer;
        hessenberg_iter = hessenberg_iter + buffer;
        arnoldi_norm = norm(next_krylov_basis);
    end
    
    hessenberg(1:local_iter+1, local_iter + 1) = hessenberg_iter;
    hessenberg(local_iter + 2, local_iter + 1) = arnoldi_norm;
    next_krylov_basis = (1/arnoldi_norm) * next_krylov_basis;
    if (reduce_precision)
        krylov_bases(:, local_iter + 2) = single(next_krylov_basis);
    else
        krylov_bases(:, local_iter + 2) = next_krylov_basis;
    end
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
    
    hessenberg(1:local_iter+2, local_iter+1) = hessenberg_iter;
end


function print_matrix(str, mtx)
    fprintf("%s\n", str)
    disp(mtx)
end

% Change format to show more digits: `format long`
% Show current format: `fmt = format`
% `single(x)` cast x to single precision
% `double(x)` cast x to double precision
% both work the same when dealing with input vectors

