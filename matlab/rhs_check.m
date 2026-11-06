% Evaluate the original MATLAB equations of motion on every row of rhs_inputs.csv
% (columns: c2_A..c2_D, c3_A..c3_D, then the 42-element state x) and write rhs_matlab.csv.
% Takes seconds. Works in MATLAB and GNU Octave.

A = dlmread('rhs_inputs.csv', ',');
n = size(A, 1);
N = 20;
material_info = [1, 0.01 * N/2, 1.2e4, 0.02];   % [m_layer M_impactor C_impact zeta]
out = zeros(n, 2*(N+1));
for k = 1:n
    C2 = repelem(A(k, 1:4)', 5, 1);
    C3 = repelem(A(k, 5:8)', 5, 1);
    info = [ones(N, 1) C2 C3];
    x = A(k, 9:end)';
    out(k, :) = impact_equation_of_motion_hetero(0, x, info, material_info, 0)';
end
fid = fopen('rhs_matlab.csv', 'w');
fmt = [repmat('%.17g,', 1, 2*(N+1)-1) '%.17g\n'];
fprintf(fid, fmt, out');
fclose(fid);
fprintf('wrote %d rows -> rhs_matlab.csv\n', n);
