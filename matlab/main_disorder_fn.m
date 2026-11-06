function [max_ke_ratio, max_KE_non, max_KE_lin] = main_disorder_fn(c2, c3)
% Function version of main_disorder.m (identical model, solver settings and output grid).
% c2, c3: 4-element vectors of layer coefficients.

c1 = [1; 1; 1; 1];
c2 = c2(:);
c3 = c3(:);

C1 = repelem(c1, 5, 1);
C2 = repelem(c2, 5, 1);
C3 = repelem(c3, 5, 1);

zeta = 0.02;
N = 20;
m_layer = 1;
M0_impactor = N/2;
C_impact = 1.2e4;
M_impactor = 0.01 * M0_impactor;
V_impactor = 0.25;

options_ode45 = odeset('RelTol', 1e-12, 'AbsTol', 1e-12 .* ones(1, 2*(N+1)));

k_f = max([1; 1 + 2*c2 + 3*c3]);
cycles = 25;
outputpercycle = 100;
f0 = 1/(2*pi)*sqrt(1/M0_impactor);
dt_cyc0 = 1/f0;
f = 1/(2*pi)*sqrt(k_f/M0_impactor);
dt_cyc = 1/f;
dt = dt_cyc/outputpercycle;
T = dt_cyc0*cycles;

material_info = [m_layer M_impactor C_impact zeta];
initialvals = zeros(2*(N+1), 1);
initialvals(2) = V_impactor;

linear_spring_info = [C1 zeros(N,1) zeros(N,1)];
[~, X_lin] = ode45(@(t,x) impact_equation_of_motion_hetero(t, x, ...
    linear_spring_info, material_info, 0), 0:dt:T, initialvals, options_ode45);
velocity_x_lin = X_lin(:, 2:2:2*(N+1));

nonlinear_spring_info = [C1 C2 C3];
[~, X_non] = ode45(@(t,x) impact_equation_of_motion_hetero(t, x, ...
    nonlinear_spring_info, material_info, 0), 0:dt:T, initialvals, options_ode45);
velocity_x_non = X_non(:, 2:2:2*(N+1));

KE_non = 1/2 * velocity_x_non(:, N+1).^2;
KE_lin = 1/2 * velocity_x_lin(:, N+1).^2;
max_KE_non = max(KE_non);
max_KE_lin = max(KE_lin);
max_ke_ratio = max_KE_non / max_KE_lin;
end
