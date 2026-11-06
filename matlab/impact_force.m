function [f] = impact_force(C,x1,x2)

%{
    ==========================================================================
    Inputs
    
    - a: unit cell height
    - h: unit cell depth (= sample depth)
    - relative_disp: relative displacement between the adjacent unit cells
    - E_lin: linear modulus
    - N_asp: numbere of asperities per unit cell length
    
    Outputs
    
    - f: force value for the given relative displacement
    ==========================================================================
%}

% For the compressive case
if x1>x2
    
    % single spring force vs. relative_disp
    
    f =C*(x1-x2)^(3/2);
    
else
    
    f = 0;

end