function [Best_score, Best_pos, Convergence_curve] = SNNA(N, Max_iter, lb, ub, dim, fobj)
% SNNA: Hybrid of Snow Ablation Optimizer and Neural Network Algorithm
% Inputs:
%   N         - Population size (even number recommended; odd values auto-incremented)
%   Max_iter  - Maximum number of iterations
%   lb        - Lower bound (scalar or 1 x dim vector)
%   ub        - Upper bound (scalar or 1 x dim vector)
%   dim       - Problem dimensionality
%   fobj      - Objective function handle
%
% Outputs:
%   Best_score        - Global best fitness value
%   Best_pos          - Global best solution vector
%   Convergence_curve - Convergence history

    %% Parameter preprocessing
    if mod(N, 2) ~= 0
        N = N + 1;                          % Ensure even population size
    end
    N_half = floor(N / 2);                  % Subpopulation size

    if isscalar(lb), lb = lb * ones(1, dim); end
    if isscalar(ub), ub = ub * ones(1, dim); end

    %% Population initialization
    X = repmat(lb, N, 1) + rand(N, dim) .* repmat(ub - lb, N, 1);
    fitness = zeros(N, 1);
    for i = 1:N
        fitness(i) = fobj(X(i, :));
    end

    [Best_score, best_idx] = min(fitness);
    Best_pos = X(best_idx, :);

    %% NNA weight matrix initialization (N_half x N_half, row-normalized)
    W = rand(N_half, N_half);
    W = W ./ repmat(sum(W, 2), 1, N_half);  % Each row sums to 1

    %% SAO parameters
    P    = 0.5;                             % Exploration/exploitation switch probability
    beta = 1.0;                             % NNA modification factor (Eq. 5)

    %% Preallocate convergence curve
    Convergence_curve = zeros(1, Max_iter);

    %% Main loop
    for t = 1:Max_iter
        % ---------- Step 1: Rank by fitness and split ----------
        [fitness_sorted, sort_idx] = sort(fitness);
        X = X(sort_idx, :);
        fitness = fitness_sorted;

        X_elite = X(1:N_half, :);            % Elite subpopulation (better fitness)
        X_inf   = X(N_half+1:N, :);          % Inferior subpopulation (worse fitness)

        % ---------- Step 2a: SAO updates the elite subpopulation ----------
        M = (1 - t / Max_iter) ^ (-t / Max_iter);   % Eq. (10) snow ablation rate
        X_mean = mean(X_elite, 1);                    % Mean of elite subpopulation

        X_elite_new = zeros(N_half, dim);
        for i = 1:N_half
            r1 = rand();
            if r1 < P
                % ----- Exploration (vaporization phase) Eq. (11) -----
                Levy = levy_flight(dim);
                X_elite_new(i, :) = M * (Best_pos - X_elite(i, :)) ...
                                  + Levy .* (X_mean - X_elite(i, :));
            else
                % ----- Exploitation (meltwater flow phase) Eq. (12) -----
                % Note: paper uses r2 both as random scalar and index; 
                % here distinct variables are used for clarity
                r2 = rand();
                idx1 = randi(N_half);
                idx2 = randi(N_half);
                X_elite_new(i, :) = M * Best_pos ...
                                  + r2 * (X_elite(idx1, :) - X_elite(idx2, :));
            end
        end
        % Boundary handling (Eq. 13)
        X_elite_new = min(max(X_elite_new, repmat(lb, N_half, 1)), ...
                          repmat(ub, N_half, 1));

        % ---------- Step 2b: NNA updates the inferior subpopulation ----------
        % Generate new population (Eq. 3)
        X_new     = W' * X_inf;               % N_half x dim
        X_inf_new = X_inf + X_new;

        % Update weight matrix (Eq. 4)
        [~, best_inf_idx] = min(fitness(N_half+1:N));
        W_opt = W(best_inf_idx, :);           % Target weight vector (best in inferior group)

        for i = 1:N_half
            alpha = rand();
            W(i, :) = W(i, :) + 2 * alpha * (W_opt - W(i, :));
        end

        % Bias operator and transfer function operator (per individual)
        for i = 1:N_half
            if rand() <= beta
                % ----- Bias operator Eq. (6) -----
                Np = floor(beta * dim);
                if Np > 0
                    P_set = randperm(dim, min(Np, dim));
                    for s = 1:length(P_set)
                        ps = P_set(s);
                        X_inf_new(i, ps) = lb(ps) + (ub(ps) - lb(ps)) * rand();
                    end
                end
                % ----- Bias weight matrix Eq. (7) -----
                Nw = floor(beta * N_half);
                if Nw > 0
                    Q_set = randperm(N_half, min(Nw, N_half));
                    for r = 1:length(Q_set)
                        qr = Q_set(r);
                        W(i, qr) = rand();
                    end
                end
            else
                % ----- Transfer function operator Eq. (8) -----
                d = rand();
                X_inf_new(i, :) = X_inf_new(i, :) ...
                                + 2 * d * (Best_pos - X_inf_new(i, :));
            end
        end
        % Boundary handling
        X_inf_new = min(max(X_inf_new, repmat(lb, N_half, 1)), ...
                        repmat(ub, N_half, 1));

        % ---------- Step 3: Merge subpopulations ----------
        X = [X_elite_new; X_inf_new];

        % ---------- Step 4: Evaluate fitness and update global best ----------
        for i = 1:N
            fitness(i) = fobj(X(i, :));
        end
        [min_fit, min_idx] = min(fitness);
        if min_fit < Best_score
            Best_score = min_fit;
            Best_pos   = X(min_idx, :);
        end

        % ---------- Update modification factor (Eq. 5) ----------
        beta = 0.99 * beta;

        % ---------- Record convergence curve ----------
        Convergence_curve(t) = Best_score;
    end
end

%% ===================== Helper functions =====================
function L = levy_flight(dim)
% Generate Levy flight step sizes using Mantegna's algorithm (lambda = 1.5)
    beta_levy = 1.5;
    sigma = (gamma(1 + beta_levy) * sin(pi * beta_levy / 2) / ...
             (gamma((1 + beta_levy) / 2) * beta_levy * ...
              2^((beta_levy - 1) / 2)))^(1 / beta_levy);
    u = randn(1, dim) * sigma;
    v = randn(1, dim);
    L = u ./ (abs(v).^(1 / beta_levy));
end