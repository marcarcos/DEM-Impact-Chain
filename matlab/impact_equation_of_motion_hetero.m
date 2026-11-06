function [dx] = impact_equation_of_motion_hetero(t,x,nonlinear_spring_info,material_info,F)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Inputs   
%       x =  [x, x'] = [displacement, velocity] 
%       t = time
%       nonlinear_spring_info =  polynomial coeffficients
%       m = unit cell mass
%       m_impactor = impactor mass
%       zeta =  (viscous) damping parameter
%       C_impact =  contact spring info
%
%
%   Outputs  
%       dx = [x', x''] = [velocity, acceleration]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ode45 takes the form y' = f(t,y)
% For our second order ODE:
%       y      = velocity
%       y'     = acceleration
%       f(t,y) = equation of motion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Unit cell numbering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% num_mass = N + 1 (including the impactor as mass)
% i = 1         : impactor
% i = 2         : first mass (impacted)
% i = 3,..,N    : middle masses
% i = N+1       : last mass (attached to wall)

% c2,c3: spring coefficients (1 x N)
% at mass i: right spring has c(i-1), left spring has c(i-2)



% initialization of some parameters
num_mass = length(x)/2;             % number of masses
dx = zeros(length(x),1);            % vector of time derivative of displacement

c1=nonlinear_spring_info(:,1);
c2=nonlinear_spring_info(:,2);
c3=nonlinear_spring_info(:,3);

m=material_info(1);
m_impact=material_info(2);
C_impact=material_info(3);
zeta=material_info(4);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% First unit cell, "impactor"
i=1;                                  
dx(2*i-1) = x(2*i);   
dx(2*i)=(F-impact_force(C_impact,x(2*i-1),x(2*(i+1)-1)))/m_impact;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Actual 1st unit cell
% if num_mass>2
i=2;
dx(2*i-1) = x(2*i);   
if x(2*(i+1)-1)>=x(2*i-1)
    dx(2*i)=(c1(i-1)*(x(2*(i+1)-1)-x(2*i-1)) ...
        +impact_force(C_impact,x(2*(i-1)-1),x(2*i-1))-2*zeta*(-x(2*(i+1))+x(2*i)))/m;
else
    dx(2*i)=(c1(i-1)*(x(2*(i+1)-1)-x(2*i-1))-c2(i-1)*(x(2*(i+1)-1)-x(2*i-1))^2+c3(i-1)*(x(2*(i+1)-1)-x(2*i-1))^3+ ...
        +impact_force(C_impact,x(2*(i-1)-1),x(2*i-1))-2*zeta*(-x(2*(i+1))+x(2*i)))/m;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Middle cells
for i=3:num_mass-1
    dx(2*i-1) = x(2*i);
    % force from right spring
    if x(2*(i+1)-1)>=x(2*i-1)
        Force_spring1=c1(i-1)*(x(2*(i+1)-1)-x(2*i-1));
    else
        Force_spring1=c1(i-1)*(x(2*(i+1)-1)-x(2*i-1))-c2(i-1)*(x(2*(i+1)-1)-x(2*i-1))^2+c3(i-1)*(x(2*(i+1)-1)-x(2*i-1))^3;
    end
    % force from left spring
    if x(2*i-1)>=x(2*(i-1)-1)
        Force_spring2=-c1(i-2)*(x(2*i-1)-x(2*(i-1)-1));
    else
        Force_spring2=-c1(i-2)*(x(2*i-1)-x(2*(i-1)-1))+c2(i-2)*(x(2*i-1)-x(2*(i-1)-1))^2-c3(i-2)*(x(2*i-1)-x(2*(i-1)-1))^3;
    end
    dx(2*i)=(Force_spring1+Force_spring2-2*zeta*(-x(2*(i+1))+2*x(2*i)-x(2*(i-1))))/m;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Last cell
i=num_mass;
dx(2*i-1) = x(2*i);

% force from right spring
if 0>=x(2*i-1)
    Force_spring1=c1(i-1)*(0-x(2*i-1));
else
    Force_spring1=c1(i-1)*(0-x(2*i-1))-c2(i-1)*(0-x(2*i-1))^2+c3(i-1)*(0-x(2*i-1))^3;
end
% force from left spring
if x(2*i-1)>=x(2*(i-1)-1)
    Force_spring2=-c1(i-2)*(x(2*i-1)-x(2*(i-1)-1));
else
    Force_spring2=-c1(i-2)*(x(2*i-1)-x(2*(i-1)-1))+c2(i-2)*(x(2*i-1)-x(2*(i-1)-1))^2-c3(i-2)*(x(2*i-1)-x(2*(i-1)-1))^3;
end
dx(2*i)=(Force_spring1+Force_spring2-2*zeta*(x(2*i)-x(2*(i-1)))-2*zeta*(x(2*i)))/m;


