% Script to plot price and Delta of up-and-out call options against initial price using three methods:
% 1) Standard Monte Carlo (SMC)
% 2) One-Step Survival Monte Carlo (OSS1)
% 3) One-Step Survival Monte Carlo conditioning on ITM (OSS2)

% =========================================================================
% Parameters
% =========================================================================
clear
seed            = 12345;                                  % fixed random seed
S0_vec          = linspace(30,60,1000);                   % initial asset prices
k               = numel(S0_vec);                          % number of points S0
K               = 50;                                     % strike
B               = 60;                                     % barrier
t               = linspace(0,1,51);                       % observation dates
m               = length(t)-1;                            % number of observation dates
r               = 0.1;                                    % risk-free rate
d               = 0;                                      % dividend yield
sigma           = 0.2;                                    % volatility
N_SMC           = 1e2;                                    % number of simulations for SMC
adj_factorOSS1  = 0.75;                                   % adjustment factor for OSS1 (precomputed in different script)
N_OSS1          = round(adj_factorOSS1 * N_SMC);          % number of simulations for OSS1 (runtime-adjusted)
adj_factorOSS2  = 0.7;                                    % adjustment factor for OSS2 (precomputed in different script)
N_OSS2          = round(adj_factorOSS2 * N_SMC);          % number of simulations for OSS2 (runtime-adjusted)
rng(seed);                                                % ensure same seed for fair comparison
U               = rand(max([N_SMC, N_OSS1, N_OSS2]), m);  % master matrix with random numbers
U_SMC           = U(1:N_SMC, :);                          % random numbers for SMC
U_OSS1          = U(1:N_OSS1, :);                         % random numbers for OSS1
U_OSS2          = U(1:N_OSS2, :);                         % random numbers for OSS2 
h_vec           = 5e-3 * S0_vec;                          % step size in finite difference for Delta
 
% Print parameters
fprintf('=================================================================================== \n');
fprintf('Parameters:\n');
fprintf('  Fixed random seed                                  = %d\n', seed);
fprintf('  Intial prices S0                                   = %.2f to %.2f (step %.4f)\n', S0_vec(1), S0_vec(end), S0_vec(2)-S0_vec(1));
fprintf('  Number of S0 points k                              = %d\n', k);
fprintf('  Strike K                                           = %.2f\n', K);
fprintf('  Barrier B                                          = %.2f\n', B);
fprintf('  Observation dates t                                = %.2f to %.2f (step %.4f)\n', t(1), t(end), t(2)-t(1));
fprintf('  Number of observation dates m                      = %d\n', m);
fprintf('  Interest rate r                                    = %.4f\n', r);
fprintf('  Dividend yield b                                   = %.4f\n', d);
fprintf('  Volatility sigma                                   = %.4f\n', sigma);
fprintf('  Number of simulations for SMC N_SMC                = %d\n', N_SMC);
fprintf('  Random numbers for SMC U_SMC                       ~ U(0,1)\n');
fprintf('  Runtime adjustment-factor for OSS1 adj_factorOSS1  = %.2f\n', adj_factorOSS1)
fprintf('  Number of simulations for OSS N_OSS1               = %d\n', N_OSS1);
fprintf('  Random numbers for OSS U_OSS1                      ~ U(0,1)\n');
fprintf('  Runtime adjustment-factor for OSS2 adj_factorOSS2  = %.2f\n', adj_factorOSS2)
fprintf('  Number of simulations for OSS N_OSS2               = %d\n', N_OSS2);
fprintf('  Random numbers for OSS U_OSS2                      ~ U(0,1)\n');
fprintf('  Step size h range                                  = [%.4f, %.4f]\n\n', min(h_vec), max(h_vec));

% =========================================================================
% Plot 1: Option price
% =========================================================================

V_SMC = upOutCallBarrier_SMC(S0_vec, K, B, t, r, d, sigma, U_SMC);
V_OSS1 = upOutCallBarrier_OSS1(S0_vec, K, B, t, r, d, sigma, U_OSS1);
V_OSS2 = upOutCallBarrier_OSS2(S0_vec, K, B, t, r, d, sigma, U_OSS2);

figure(1);
plot(S0_vec, V_SMC, 'g.', 'DisplayName', 'Standard Monte Carlo');
hold on;
plot(S0_vec, V_OSS1, 'r.', 'DisplayName', 'OSS Monte Carlo');
plot(S0_vec, V_OSS2, 'b.', 'DisplayName', 'OSS Monte Carlo (v.2)');
hold off
legend show;
xlabel('Asset Price S_0')
ylabel('Option Price V');
title('Up-and-Out Call Price: SMC vs OSS vs OSS (v.2)'); 
ylim([0 1])


% =========================================================================
% Plot 2: Delta
% =========================================================================

dV_SMC = Delta_upOutCallBarrier_SMC(S0_vec, K, B, t, r, d, sigma, U_SMC, h_vec);
dV_OSS1 = Delta_upOutCallBarrier_OSS1(S0_vec, K, B, t, r, d, sigma, U_OSS1, h_vec);
dV_OSS2 = Delta_upOutCallBarrier_OSS2(S0_vec, K, B, t, r, d, sigma, U_OSS2, h_vec);

figure(2);
plot(S0_vec, dV_SMC, 'g.','DisplayName', 'Standard Monte Carlo');
hold on;
plot(S0_vec, dV_OSS1, 'r.', 'DisplayName', 'OSS Monte Carlo');
plot(S0_vec, dV_OSS2, 'b.', 'DisplayName', 'OSS Monte Carlo (v.2)');
hold off;
legend show;
xlabel('Asset Price S_0');
ylabel('Delta dV');
title('Up-and-Out Call Delta: SMC vs OSS vs OSS (v.2)');
ylim([-0.15 0.15])

% =========================================================================
% Functions (vertorized form - optimized in MATLAB)
% -> tailored for plotting against S0
% =========================================================================

function V = upOutCallBarrier_SMC(S0_vec, K, B, t, r, d, sigma, U)
    S0_vec = S0_vec(:)';  % ensure S0 is a row vector
    k = numel(S0_vec);
    N = size(U,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu = r-d;
    
    % Initilize variables (Nxk)
    S = repmat(S0_vec,N,1);  % assets' prices
    alive = true(N,k);  % keep track of paths alive
    payoff = zeros(N,k);  % payoffs
    
    % sample from standard normal distribution (Nxm)
    Z = norminv(U);  % inverse sampling

    for j=1:m
        % compute drift and volatilty
        drift = (mu - 0.5*sigma^2)*dt(j);
        vol = sigma * sqrt(dt(j));
        
        Zj = Z(:,j);  % N random numbers for simulations at j-th observation date (Nx1)
        Zj_mat = repmat(Zj,1,k);  % same N random numbers for each initial price S0 (Nxk)
        
        % update paths and their states (Nxk)
        S(alive) = S(alive) .* exp(drift + vol .* Zj_mat(alive));
        hit = alive & (S>=B);
        alive(hit) = false;
    end
    
    % compute (discounted) payoffs (only paths alive, others 0)
    payoff(alive) = exp(-r*(t(m+1)-t(1))) .* max(S(alive)-K,0);  % (Nxk)
    V = mean(payoff,1);  % mean over simulations - one for each initial S0 (1xk)

end

function dV = Delta_upOutCallBarrier_SMC(S0_vec, K, B, t, r, d, sigma, U, h_vec)
    % Forward finite differences Monte Carlo with CRN
    V = upOutCallBarrier_SMC(S0_vec, K, B, t, r, d, sigma, U);
    Vh = upOutCallBarrier_SMC(S0_vec+h_vec, K, B, t, r, d, sigma, U);
    dV = (Vh - V)./h_vec;
end


function V = upOutCallBarrier_OSS1(S0_vec, K, B, t, r, d, sigma, U)

    S0_vec = S0_vec(:)';  % ensure S0 is a row vector
    k = numel(S0_vec);
    N = size(U,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu = r-d;
    
    % initialize variables (Nxk)
    S = repmat(S0_vec,N,1);  % paths
    W = ones(N,k);  % likelihood ratios
    
    for j=1:m
        % compute drift and volatility
        drift = (mu - 0.5*sigma^2) * dt(j);
        vol = sigma * sqrt(dt(j));
    
        % compute survival probabilities
        b = (log(B./S) - drift) / vol;
        p = normcdf(b);
        p = min(p,1-1e-15);  % clamp probability to avoid underflow
        
        % update likelihood ratios
        W = W .* p;

        % sample from truncated standard normal distribution
        Uj = U(:,j);  % N random numbers for simulations at j-th observation date
        Uj_mat = repmat(Uj,1,k);  % same N random numbers for each initial price S0 (Nxk matrix)
        Z = norminv(p .* Uj_mat);  % inverse sampling

        % update paths
        S = S .* exp(drift + vol.*Z);
    end

    % update (discounted) weighted payoffs
    payoff = exp(-r*(t(m+1)-t(1))) .* W .* max(S-K,0);
    % compute option value
    V = mean(payoff,1);  % mean over simulations - one for each initial S0 (1xk)
end

function dV = Delta_upOutCallBarrier_OSS1(S0_vec, K, B, t, r, d, sigma, U, h_vec)
    % Forward finite differences OSS Monte Carlo with CRN
    V = upOutCallBarrier_OSS1(S0_vec, K, B, t, r, d, sigma, U);
    Vh = upOutCallBarrier_OSS1(S0_vec+h_vec, K, B, t, r, d, sigma, U);
    dV = (Vh - V)./h_vec;
end


function V = upOutCallBarrier_OSS2(S0_vec, K, B, t, r, d, sigma, U)

    S0_vec = S0_vec(:)';  % ensure S0 is a row vector
    k = numel(S0_vec);
    N = size(U,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu = r-d;
    
    % initialize variables (Nxk)
    S = repmat(S0_vec,N,1);  % paths
    W = ones(N,k);  % likelihood ratios
    
    for j=1:m-1
        % compute drift and volatility
        drift = (mu - 0.5*sigma^2) * dt(j);
        vol = sigma * sqrt(dt(j));

        % compute survival probabilities
        b = (log(B./S) - drift) / vol;
        p = normcdf(b);
        p = min(p,1-1e-15);  % clampt probability to avoid underflow

        % update likelihood ratios
        W = W .* p;

        % sample from truncated standard normal distribution
        Uj = U(:,j);  % N random numbers for simulations at j-th observation date
        Uj_mat = repmat(Uj,1,k);  % same N random numbers for each initial price S0 (Nxk matrix)
        Z = norminv(p .* Uj_mat);  % inverse sampling

        % update paths
        S = S .* exp(drift + vol.*Z);
    end

    % last step - condition on (K < S_m < B)

    % compute drift and volatility
    drift = (mu - 0.5*sigma^2) * dt(m);
    vol = sigma * sqrt(dt(m));

    % compute survival probabilities (survive barrier AND strike price too!)
    a = (log(K./S) - drift) / vol;
    b = (log(B./S) - drift) / vol;
    p1 = normcdf(a);
    p2 = normcdf(b);
    p = max(p2-p1, 1e-15);  % clamp probability to avoid under/overflow

    % update likelihood ratios
    W = W .* p;

    % sample from truncated standard normal distribution
    Um = U(:,m);  % N random numbers for simulations at j-th observation date
    Um_mat = repmat(Um,1,k);  % same N random numbers for each initial price S0 (Nxk matrix)
    
    arg = min(max(p1 + p .* Um_mat, 1e-15), 1-1e-15);  
    Z = norminv(arg);  % inverse sampling

    % update paths
    S = S .* exp(drift + vol.*Z);

    % update (discounted) weighted payoffs
    payoff = exp(-r*(t(m+1)-t(1))) .* W .* (S-K);
    % compute option value
    V = mean(payoff,1);  % mean over simulations - one for each initial S0 (1xk)
end

function dV = Delta_upOutCallBarrier_OSS2(S0_vec, K, B, t, r, d, sigma, U, h_vec)
    % Forward finite differences OSS (v.2) Monte Carlo with CRN
    V = upOutCallBarrier_OSS2(S0_vec, K, B, t, r, d, sigma, U);
    Vh = upOutCallBarrier_OSS2(S0_vec+h_vec, K, B, t, r, d, sigma, U);
    dV = (Vh - V)./h_vec;
end
