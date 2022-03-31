function [kappa,relres,iters] = test_cb_gmres(N, restart, tol, maxit, iters_to_plot)
%TEST_CB_GMRES Summary of this function goes here
%   Detailed explanation goes here

output_folder = 'plots/';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

x_init = zeros(N, 1);
b = ones(N, 1);
norm_b = norm(b);

if (nargin < 2)
    restart=max(100, N);
end
if (nargin < 3)
    tol = 1e-12;
end
if (nargin < 4)
    maxit=2*N;
end

loop_iters=50;
kappa = zeros(loop_iters, 1);
relres = zeros(loop_iters, 2);
iters = zeros(loop_iters, 2);

for i = 1:loop_iters
    kappa(i) = 10^(i*8/loop_iters);
    A = gallery("randsvd", N, kappa(i), 1);
    [cx,cflag,crelres,citer,cresvec] = cb_gmres(A, b, x_init, restart, tol, maxit, true);
    gmres_mi = ceil(maxit / restart);
    if (N == restart)
        gmres_mi = maxit;
    end
    [gx,gflag,grelres,giter,gresvec] = gmres(A, b, restart, tol, gmres_mi, [], [], x_init);
    
    relres(i, 1) = abs(norm(b - A*cx) / norm_b);
    iters(i,1) = citer;
    relres(i, 2) = abs(norm(b - A*gx) / norm_b);
    iters(i,2) = (giter(1) - 1) * restart + giter(2);

    if ismember(i, iters_to_plot)
        [fig, ax] = plot_helper(cresvec./norm_b, gresvec./norm_b, ...
                        append("Iteration ",string(i)," with Kappa ",compose("%.2e", kappa(i))));
        print_figure(fig, append(output_folder, "relres_norm_r", string(restart), "_m", string(maxit), "_i", string(i)));
    end
end

[fig, ax] = ...
    plot_helper(relres(:, 1), relres(:, 2), ...
        append("Final relres norm for Rest: ", string(restart), ", Maxint: ", string(maxit)));
print_figure(fig, append(output_folder, "final_relres_norms_r", string(restart), "_m", string(maxit)));

[fig, ax] = plot_helper(kappa, [], append("Kappa value per iteration"));
print_figure(fig, append(output_folder, "kappa"));

end

function [fig, ax] = plot_helper(y1, y2, title_str)

myblue = [ 0       0.4470  0.7410];
mygreen = [0.4660  0.6740  0.1880];
fig = figure('Color', [1,1,1]);
set(fig, 'position', [0 0 2000 800]);
ax = axes('Parent', fig);

hold(ax, 'on');
plt1 = plot(y1, 's-', 'Color', myblue,'MarkerEdgeColor', myblue, 'MarkerFaceColor', myblue);
if ~isempty(y2)
    plt2 = plot(y2, '+-', 'Color', mygreen,'MarkerEdgeColor', mygreen, 'MarkerFaceColor', mygreen);
    legend(ax, ["CB-GMRES", "GMRES"], 'Location', 'SW')
end
set(ax, 'XGrid', 'on', 'YGrid', 'on');
set(ax, 'YScale', 'log');
set(ax, 'XScale', 'linear');
xlim([0, max(size(y1, 1), size(y2, 1))]);
title(ax, title_str);
hold(ax, 'off');
end

function [] = print_figure(fig, fig_name)
paper_scalar_w = 0.85;
paper_scalar_h = 0.9;

% Actually plot:
set(fig, 'Color', 'white'); % white bckgr

% Makes it the default behavior to always fill out the graph window
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3)*paper_scalar_w fig_pos(4)*paper_scalar_h];
print(fig, fig_name, '-dpdf')

end