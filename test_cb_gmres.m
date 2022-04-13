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
relres = zeros(loop_iters, 4);
back_relres = zeros(loop_iters, 4);
iters = zeros(loop_iters, 4);

for i = 1:loop_iters
    kappa(i) = 10^(i*8/loop_iters);
    A = gallery("randsvd", N, kappa(i), 1);
    [cx,cflag,crelres,citer,cresvec] = cb_gmres(A, b, x_init, restart, tol, maxit, true);
    [sx,sflag,srelres,siter,sresvec] = cb_gmres(A, b, x_init, restart, tol, maxit, [true, true]);
    [dx,dflag,drelres,diter,dresvec] = cb_gmres(A, b, x_init, restart, tol, maxit, false);
    gmres_mi = ceil(maxit / restart);
    if (N == restart)
        gmres_mi = maxit;
    end
    [gx,gflag,grelres,giter,gresvec] = gmres(A, b, restart, tol, gmres_mi, [], [], x_init);
    
    relres(i, 1) = abs(norm(b - A*cx) / norm_b);
    back_relres(i, 1) = abs(norm(b - A*cx) / (norm(A) * norm(cx)));
    iters(i, 1) = citer;
    relres(i, 2) = abs(norm(b - A*sx) / norm_b);
    back_relres(i, 2) = abs(norm(b - A*sx) / (norm(A) * norm(sx)));
    iters(i, 2) = siter;
    relres(i, 3) = abs(norm(b - A*dx) / norm_b);
    back_relres(i, 3) = abs(norm(b - A*dx) / (norm(A) * norm(dx)));
    iters(i, 3) = diter;
    relres(i, 4) = abs(norm(b - A*gx) / norm_b);
    back_relres(i, 4) = abs(norm(b - A*gx) / (norm(A) * norm(gx)));
    iters(i, 4) = (giter(1) - 1) * restart + giter(2);

    if ismember(i, iters_to_plot)
        [fig, ax] = plot_helper(cresvec./norm_b, sresvec./norm_b, dresvec./norm_b, gresvec./norm_b, ...
                        append("Iteration ",string(i)," with Kappa ",compose("%.2e", kappa(i))));
        print_figure(fig, append(output_folder, "relres_norm_r", string(restart), "_m", string(maxit), "_i", string(i)));
    end
end

[fig, ax] = ...
    plot_helper(relres(:, 1), relres(:, 2), relres(:, 3), relres(:, 4), ...
        append("Final relres norm for Rest: ", string(restart), ", Maxit: ", string(maxit)));
print_figure(fig, append(output_folder, "final_relres_norms_r", string(restart), "_m", string(maxit)));
[fig, ax] = ...
    plot_helper(back_relres(:, 1), back_relres(:, 2), back_relres(:, 3), back_relres(:, 4), ...
        append("Final backwards error for Rest: ", string(restart), ", Maxit: ", string(maxit)));
print_figure(fig, append(output_folder, "final_backward_error_r", string(restart), "_m", string(maxit)));

[fig, ax] = plot_helper(kappa, [], [], [], append("Kappa value per iteration"), false);
print_figure(fig, append(output_folder, "kappa"));
[fig, ax] = plot_helper(iters(:, 1), iters(:, 2), iters(:, 3), iters(:, 4), append("Number of iterations for each solve"), true, 'linear');
print_figure(fig, append(output_folder, "iterations_r", string(restart), "_m", string(maxit)));

end

function [fig, ax] = plot_helper(cb_gmres, scb_gmres, cgs_gmres, gmres, title_str, show_legend, y_scale)

if (nargin < 6)
    show_legend = true;
end
if (nargin < 7)
    y_scale = 'log';
end

myblue =   [0       0.4470  0.7410];
myorange = [1.0     0.4     0.1];
mygreen =  [0.4660  0.6740  0.1880];
myred =    [0.6350  0.0780  0.1840];

fig = figure('Color', [1,1,1]);
set(fig, 'position', [0 0 2000 800]);
ax = axes('Parent', fig);

legend_strs = string([]);

hold(ax, 'on');
if ~isempty(cb_gmres)
    plt1 = plot(cb_gmres, 'X-', 'Color', myblue,'MarkerEdgeColor', myblue, 'MarkerFaceColor', myblue);
    legend_strs(end+1) = "CB-GMRES";
end
if ~isempty(scb_gmres)
    plt2 = plot(scb_gmres, 'o-', 'Color', myorange,'MarkerEdgeColor', myorange);
    legend_strs(end+1) = "sp CB-GMRES";
end
if ~isempty(cgs_gmres)
    plt3 = plot(cgs_gmres, '+-', 'Color', mygreen,'MarkerEdgeColor', mygreen, 'MarkerFaceColor', mygreen);
    legend_strs(end+1) = "CGS-GMRES";
end
if ~isempty(gmres)
    plt4 = plot(gmres, 's-', 'Color', myred,'MarkerEdgeColor', myred, 'MarkerFaceColor', myred);
    legend_strs(end+1) = "GMRES";
end

legend(ax, legend_strs, 'Location', 'best');
if (~show_legend)
    legend(ax, 'hide')
end
set(ax, 'XGrid', 'on', 'YGrid', 'on');
set(ax, 'YScale', y_scale);
if (strcmp(y_scale, 'linear'))
    ylim(ax, [0 max([max(cb_gmres) max(scb_gmres) max(cgs_gmres) max(gmres)])]);
end
set(ax, 'XScale', 'linear');
xlim(ax, [0, max([size(cb_gmres, 1), size(scb_gmres, 1) size(cgs_gmres, 1), size(gmres, 1)])]);
title(ax, title_str);
hold(ax, 'off');
end

function [] = print_figure(fig, fig_name)
paper_scalar_w = 0.85;
paper_scalar_h = 0.9;

% Actually plot:
set(fig, 'Color', 'white'); % white bckgr

print(fig, fig_name, '-dsvg');

% Makes it the default behavior to always fill out the graph window
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3)*paper_scalar_w fig_pos(4)*paper_scalar_h];
print(fig, fig_name, '-dpdf');

end
