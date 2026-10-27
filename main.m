%% test_SNNA.m -- SNNA test script
clc; clear; close all;

%% Problem definition (Sphere function as an example)
dim = 30;
lb  = -100 * ones(1, dim);
ub  =  100 * ones(1, dim);
fobj = @(x) sum(x.^2);

%% Algorithm parameters
N        = 30;      % Population size
Max_iter = 500;     % Maximum number of iterations

%% Run SNNA
[Best_score, Best_pos, Convergence_curve] = ...
    SNNA(N, Max_iter, lb, ub, dim, fobj);

%% Display results
fprintf('=====================================\n');
fprintf('SNNA Optimization Results\n');
fprintf('=====================================\n');
fprintf('Best fitness: %e\n', Best_score);
fprintf('First 5 dims of best solution: '); disp(Best_pos(1:min(5,dim)));

%% Plot convergence curve
figure('Color','w');
semilogy(1:Max_iter, Convergence_curve, 'b-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Best fitness (log scale)');
title('SNNA Convergence Curve (Sphere function, 30D)');
grid on;