% Space-time matrices of the chain response, matching scipy_ref/plot_matrices.py.
% Run main_disorder.m first (it leaves t, velocity_x_lin, velocity_x_non in the workspace),
% then run this script. Time on the y axis, mass index on the x axis, colour = velocity.
%
% velocity_x_* has N+1 columns: column 1 = impactor, columns 2..N+1 = chain masses 1..N.

N = size(velocity_x_lin, 2) - 1;
V_lin = velocity_x_lin(:, 2:end);   % chain masses only (drop the impactor column)
V_non = velocity_x_non(:, 2:end);
vmax = max(abs([V_lin(:); V_non(:)]));   % one colour scale for both panels

% blue-white-red diverging colormap
n = 128;
up = linspace(0, 1, n)';
cmap = [[up; ones(n, 1)], [up; flipud(up)], [ones(n, 1); flipud(up)]];

figure('Position', [100 100 1100 700]);
titles = {'Linear chain', 'Nonlinear chain'};
data = {V_lin, V_non};
for k = 1:2
    subplot(1, 2, k);
    imagesc(1:N, t, data{k});
    set(gca, 'YDir', 'normal');           % time increases upward
    caxis([-vmax vmax]);
    colormap(cmap);
    hold on;
    xline([5.5 10.5 15.5], ':', 'Color', [0.5 0.5 0.5]);   % family boundaries A|B, B|C, C|D
    hold off;
    title(titles{k});
    xlabel('mass index (1 = impacted end, 20 = next to wall)');
    if k == 1, ylabel('time t'); end
end
cb = colorbar('Position', [0.93 0.15 0.015 0.7]);
cb.Label.String = 'velocity';
% ylim([0 150])   % uncomment to zoom in on the first pass of the pulse
