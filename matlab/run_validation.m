% Run the original MATLAB model on every design in validation_designs.csv and write
% matlab_results.csv. Works in MATLAB and in GNU Octave.
%
% Put this file, main_disorder_fn.m, impact_equation_of_motion_hetero.m,
% impact_force.m and validation_designs.csv in one folder, then:
%     run_validation          (MATLAB command window)
%     octave run_validation.m (terminal)
%
% CSV columns: id, c2_A..c2_D, c3_A..c3_D, paper_ke  (first row is a header)

infile  = 'validation_designs.csv';
outfile = 'matlab_results.csv';
if exist('OCTAVE_VERSION', 'builtin'), engine = 'octave'; else, engine = 'matlab'; end

D = dlmread(infile, ',', 1, 0);
n = size(D, 1);
res = nan(n, 5);
fprintf('%d designs, engine: %s\n', n, engine);
for k = 1:n
    tic;
    [r, ke_non, ke_lin] = main_disorder_fn(D(k, 2:5), D(k, 6:9));
    res(k, :) = [D(k, 1), r, ke_non, ke_lin, toc];
    fprintf('%3d/%d  id %3d  ratio %.8f  (%.0f s)\n', k, n, D(k, 1), r, res(k, 5));
end

fid = fopen(outfile, 'w');
fprintf(fid, 'id,ratio,max_ke_non,max_ke_lin,seconds,engine\n');
for k = 1:n
    fprintf(fid, '%d,%.15e,%.15e,%.15e,%.1f,%s\n', res(k, 1), res(k, 2), res(k, 3), res(k, 4), res(k, 5), engine);
end
fclose(fid);
fprintf('wrote %s\n', outfile);
