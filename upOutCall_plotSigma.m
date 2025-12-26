% Script to plot price and Vega of up-and-out call options against initial price using three methods:
% 1) Standard Monte Carlo (SMC)
% 2) One-Step Survival Monte Carlo (OSS1)
% 3) One-Step Survival Monte Carlo conditioning on ITM (OSS2)

% =========================================================================
% Parameters
% =========================================================================
clear
seed            = 12345;                                  % fixed random seed
sigma_vec       = linspace(1e-5,0.5,1000);                % vector of volatilities
k               = numel(sigma_vec);                           % number of sigma points
S0              = 50;                                     % initial asset price S0
K               = 50;                                     % strike
B               = 60;                                     % barrier
t               = linspace(0,1,51);                       % observation dates
m               = length(t)-1;                            % number of observation dates
r               = 0.1;                                    % risk-free rate
d               = 0;                                      % dividend yield
N_SMC           = 1e2;                                    % number of simulations for SMC
adj_factorOSS1  = 0.75;                                   % adjustment factor for OSS1 (precomputed in different script)
N_OSS1          = round(adj_factorOSS1 * N_SMC);          % number of simulations for OSS1 (runtime-adjusted)
adj_factorOSS2  = 0.7;                                    % adjustment factor for OSS2 (precomputed in different script)
N_OSS2          = round(adj_factorOSS2 * N_SMC);          % number of simulations for OSS2 (runtime-adjusted)
rng(seed);                                                % ensure same seed for fair comparison (call before any random number)
U               = rand(max([N_SMC, N_OSS1, N_OSS2]), m);  % master matrix with random numbers
U_SMC           = U(1:N_SMC, :);                          % random numbers for SMC
U_OSS1          = U(1:N_OSS1, :);                         % random numbers for OSS1
U_OSS2          = U(1:N_OSS2, :);                         % random numbers for OSS2
h_vec           = 5e-3 * sigma_vec;                       % step size in finite difference for Vega

% Print parameters
fprintf('================================================================================= \n');
fprintf('Parameters used:\n');
fprintf('  Fixed random seed                                  = %d\n', seed);
fprintf('  Volatility sigma                                   = %.2f to %.2f (step %.4f)\n', sigma_vec(1), sigma_vec(end), sigma_vec(2)-sigma_vec(1));
fprintf('  Number of sigma points k                           = %d\n', k);
fprintf('  Intial prices S0                                   = %.2f\n', S0);
fprintf('  Strike K                                           = %.2f\n', K);
fprintf('  Barrier B                                          = %.2f\n', B);
fprintf('  Observation dates t                                = %.2f to %.2f (step %.4f)\n', t(1), t(end), t(2)-t(1));
fprintf('  Number of observation dates m                      = %d\n', m);
fprintf('  Interest rate r                                    = %.4f\n', r);
fprintf('  Dividend yield b                                   = %.4f\n', d);
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

V_SMC = upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma_vec, U_SMC);
V_OSS1 = upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma_vec, U_OSS1);
V_OSS2 = upOutCallBarrier_OSS2(S0, K, B, t, r, d, sigma_vec, U_OSS2);


figure(1);
plot(sigma_vec, V_SMC, 'g.', 'DisplayName', 'Standard Monte Carlo');
hold on;
plot(sigma_vec, V_OSS1, 'r.', 'DisplayName', 'OSS Monte Carlo');
plot(sigma_vec, V_OSS2, 'b.', 'DisplayName', 'OSS Monte Carlo (v.2)');
hold off
legend show;
xlabel('Volatility sigma')
ylabel('Option Price V');
title('Up-and-Out Call Price: SMC vs OSS vs OSS (v.2)'); 
ylim([0 6])

% =========================================================================
% Plot 2: Vega
% =========================================================================

dV_SMC = Vega_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma_vec, U_SMC, h_vec);
dV_OSS1 = Vega_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma_vec, U_OSS1, h_vec);
dV_OSS2 = Vega_upOutCallBarrier_OSS2(S0, K, B, t, r, d, sigma_vec, U_OSS2, h_vec);

figure(2);
plot(sigma_vec, dV_SMC, 'g.','DisplayName', 'Standard Monte Carlo');
hold on;
plot(sigma_vec, dV_OSS1, 'r.', 'DisplayName', 'OSS Monte Carlo');
plot(sigma_vec, dV_OSS2, 'b.', 'DisplayName', 'OSS Monte Carlo (v.2)');
hold off;
legend show;
xlabel('Volatility sigma');
ylabel('Vega dV');
title('Up-and-Out Call Vega: SMC vs OSS vs OSS (v.2)');
ylim([-60 3])


% =========================================================================
% Functions (vertorized form - optimized in MATLAB)
% -> tailored for plotting against sigma_vec
% =========================================================================

function V = upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma_vec, U)

    sigma_vec = sigma_vec(:)';  % ensure S0 is a row vector
    k = numel(sigma_vec);
    N = size(U,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu = r-d;
    
    % Initilize variables (Nxk)
    S = repmat(S0,N,k);  % assets' prices
    alive = true(N,k);  % keep track of paths alive
    payoff = zeros(N,k);  % payoffs
    
    % sample from standard normal distribution (Nxm)
    Z = norminv(U);  % inverse sampling

    for j=1:m
        % compute drift and volatilty
        drift = (mu - 0.5.*sigma_vec.^2).*dt(j);
        drift_mat = repmat(drift,N,1);
        vol = sigma_vec .* sqrt(dt(j));
        vol_mat = repmat(vol,N,1);
        
        Zj = Z(:,j);  % N random numbers for simulations at j-th observation date (Nx1)
        Zj_mat = repmat(Zj,1,k);  % same N random numbers for each initial price S0 (Nxk)
        
        % update paths and their states (Nxk)
        S(alive) = S(alive) .* exp(drift_mat(alive) + vol_mat(alive).*Zj_mat(alive));
        alive = alive & (S < B);
    end
    
    % compute (discounted) payoffs (only paths alive, others 0)
    payoff(alive) = exp(-r*(t(m+1)-t(1))) .* max(S(alive)-K,0);  % (Nxk)
    V = mean(payoff,1);  % mean over simulations - one for each initial S0 (1xk)
end

function dV = Vega_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma_vec, U, h_vec)
    % Forward finite differences Monte Carlo with CRN
    V = upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma_vec, U);
    Vh = upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma_vec+h_vec, U);
    dV = (Vh - V)./h_vec;
end


function V = upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma_vec, U)

    sigma_vec = sigma_vec(:)';  % ensure S0 is a row vector
    k = numel(sigma_vec);
    N = size(U,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu = r-d;
    
    % initialize variables (Nxk)
    S = repmat(S0,N,k);  % paths
    W = ones(N,k);  % likelihood ratios
    
    for j=1:m
        % compute drift and volatility
        drift = (mu - 0.5.*sigma_vec.^2) .* dt(j);
        drift_mat = repmat(drift,N,1);
        vol = sigma_vec .* sqrt(dt(j));
        vol_mat = repmat(vol,N,1);

        % compute survival probabilities
        b = (log(B./S) - drift_mat) ./ vol_mat;
        p = normcdf(b);
        p = min(p,1-1e-15);  % clamp probability to avoid underflow
        
        % update likelihood ratios
        W = W .* p;

        % sample from truncated standard normal distribution
        Uj = U(:,j);  % N random numbers for simulations at j-th observation date
        Uj_mat = repmat(Uj,1,k);  % same N random numbers for each initial price S0 (Nxk matrix)
        Z = norminv(p .* Uj_mat);  % inverse sampling

        % update paths
        S = S .* exp(drift_mat + vol_mat.*Z);
    end

    % update (discounted) weighted payoffs
    payoff = exp(-r*(t(m+1)-t(1))) .* W .* max(S-K,0);
    % compute option value
    V = mean(payoff,1);  % mean over simulations - one for each initial S0 (1xk)
end

function dV = Vega_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma_vec, U, h_vec)
    % Forward finite differences OSS Monte Carlo with CRN
    V = upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma_vec, U);
    Vh = upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma_vec+h_vec, U);
    dV = (Vh - V)./h_vec;
end


function V = upOutCallBarrier_OSS2(S0, K, B, t, r, d, sigma_vec, U)

    sigma_vec = sigma_vec(:)';  % ensure S0 is a row vector
    k = numel(sigma_vec);
    N = size(U,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu = r-d;
    
    % initialize variables (Nxk)
    S = repmat(S0,N,k);  % paths
    W = ones(N,k);  % likelihood ratios
    
    for j=1:m-1
        % compute drift and volatility
        drift = (mu - 0.5.*sigma_vec.^2) .* dt(j);
        drift_mat = repmat(drift,N,1);
        vol = sigma_vec .* sqrt(dt(j));
        vol_mat = repmat(vol,N,1);

        % compute survival probabilities
        b = (log(B./S) - drift_mat) ./ vol_mat;
        p = normcdf(b);
        p = min(p,1-1e-15);  % clampt probability to avoid underflow
        
        % update likelihood ratios
        W = W .* p;

        % sample from truncated standard normal distribution
        Uj = U(:,j);  % N random numbers for simulations at j-th observation date
        Uj_mat = repmat(Uj,1,k);  % same N random numbers for each initial price S0 (Nxk matrix)
        Z = norminv(p .* Uj_mat);  % inverse sampling

        % update paths
        S = S .* exp(drift_mat + vol_mat.*Z);
    end

    % last step - condition on (K < S_m < B)

    % compute drift and volatility
    drift = (mu - 0.5.*sigma_vec.^2) .* dt(m);
    drift_mat = repmat(drift,N,1);
    vol = sigma_vec .* sqrt(dt(m));
    vol_mat = repmat(vol,N,1);

    % compute survival probabilities (survive barrier AND strike price too!)
    a = (log(K./S) - drift_mat) ./ vol_mat;
    b = (log(B./S) - drift_mat) ./ vol_mat;
    p1 = normcdf(a);
    p2 = normcdf(b);
    p = max(p2-p1, 1e-15);  % clamp probability to avoid under/overflow

    % update likelihood ratios
    W = W .* p;

    % sample from truncated standard normal distribution
    Um = U(:,m);  % N random numbers for simulations at j-th observation date
    Um_mat = repmat(Um,1,k);  % same N random numbers for each initial price S0 (Nxk matrix)
    
    arg = min(max(p1 + p .* Um_mat, 1e-15), 1-1e-15);  % clamp argument to avoid floating-point underflow
    Z = norminv(arg);  % inverse sampling

    % update paths
    S = S .* exp(drift_mat + vol_mat.*Z);

    % update (discounted) weighted payoffs
    payoff = exp(-r*(t(m+1)-t(1))) .* W .* (S-K);
    % compute option value
    V = mean(payoff,1);  % mean over simulations - one for each initial S0 (1xk)
end

function dV = Vega_upOutCallBarrier_OSS2(S0, K, B, t, r, d, sigma_vec, U, h_vec)
    % Forward finite differences OSS (v.2) Monte Carlo with CRN
    V = upOutCallBarrier_OSS2(S0, K, B, t, r, d, sigma_vec, U);
    Vh = upOutCallBarrier_OSS2(S0, K, B, t, r, d, sigma_vec+h_vec, U);
    dV = (Vh - V)./h_vec;
end
