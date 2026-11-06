clear,clc, close all

%% Spring Coefficients

% 4 spring families [AAAAA - BBBBB - CCCCC - DDDDD - EEEEE];

c1 = [1; 1; 1; 1];
c2 = [-70; -100; -5; -35];
c3 = [1200; 2000; 1500; 3000];


% 20x1 coefficients
C1 = repelem(c1, 5,1);
C2 = repelem(c2, 5,1);
C3 = repelem(c3, 5,1);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize input parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Nondimensional viscous damping parameter
zeta = 0.02;       

% Number of layers
N = 20;           
m_layer = 1;

% Impactor conditions
M0_impactor = N/2;
C_impact = 1.2e4;

% Impactor mass
M_impactor = 0.01 * M0_impactor;

% Impactor Velocity
V_impactor = 0.25;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solver Setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options_ode45 = odeset('RelTol',1e-12,'AbsTol',1e-12.*ones(1,2*(N+1)));

% Max linear stiffness
k_f = max([1; 1 + 2*c2 + 3*c3]);

% Output cycles
cycles = 25;
outputpercycle = 100;


f0 = 1/(2*pi)*sqrt(1/M0_impactor);
dt_cyc0 = 1/f0; 

f = 1/(2*pi)*sqrt(k_f/M0_impactor);
dt_cyc = 1/f;

dt = dt_cyc/outputpercycle;
T = dt_cyc0*cycles;


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve the DEM systems
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Material info and initial conditions
material_info=[m_layer M_impactor C_impact zeta];
initialvals = zeros(2*(N+1),1);
initialvals(2) = V_impactor;

% Linear System Solve
linear_spring_info = [C1 zeros(N,1) zeros(N,1)];

[t, X_lin] = ode45(@(t,x) impact_equation_of_motion_hetero(t,x,...
    linear_spring_info,material_info,0),0:dt:T, initialvals, options_ode45);

velocity_x_lin = X_lin(:,2:2:2*(N+1));
displacement_x_lin = X_lin(:,3:2:2*(N+1));

% Nonlinear System Solve
nonlinear_spring_info = [C1 C2 C3];

[~, X_non] = ode45(@(t,x) impact_equation_of_motion_hetero(t,x, ...
    nonlinear_spring_info, material_info, 0), 0:dt:T, initialvals, options_ode45);

velocity_x_non=X_non(:,2:2:2*(N+1));
displacement_x_non=X_non(:,3:2:2*(N+1));

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Postprocess
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute KE at last cell
KE_non = 1/2 * velocity_x_non(:,N+1).^2; 
KE_lin = 1/2 * velocity_x_lin(:,N+1).^2;

% Max KE at last cell
max_KE_non = max(KE_non);
max_KE_lin = max(KE_lin);

% KE ratio
max_ke_ratio = max_KE_non / max_KE_lin;
fprintf('Maximum KE ratio at the last cell = %.6f\n', max_ke_ratio);

