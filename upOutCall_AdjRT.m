% Script to find adjusted runtimes for Standard Monte Carlo (SMC)
% and One-Step Survival Monte Carlo (OSS) in up-and-out call options

% =========================================================================
% Parameters
% =========================================================================
clear
seed   = 12345;             % fixed random seed
S0     = 50;                % initial asset price
K      = 50;                % strike
B      = 60;                % barrier
t      = linspace(0,1,51);  % observation dates
m      = length(t)-1;       % number of observation dates
r      = 0.1;               % risk-free rate
d      = 0;                 % dividend yield
sigma  = 0.2;               % volatility
N      = 1e4;               % number of simulations
rng(seed)                   % ensure same seed for fair comparison
U      = rand(N,m);         % random numbers


adj_factorOSS1 = RuntimeFactorOSS1(S0, K, B, t, r, d, sigma, U);
fprintf('Adjustment factor OSS1 to SMC:  %.4f\n', adj_factorOSS1)

adj_factorOSS2 = RuntimeFactorOSS2(S0, K, B, t, r, d, sigma, U);
fprintf('Adjustment factor OSS2 to SMC:  %.4f\n\n', adj_factorOSS2)


% =========================================================================
% Functions (vertorized form)
% =========================================================================

% Standard Monte Carlo
function P = payoffs_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma, U)

    N = size(U,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu = r-d;
    
    % Initilize variables (Nxk)
    S = S0 .* ones(N,1);  % assets' prices
    alive = true(N,1);  % keep track of paths alive
    P = zeros(N,1);  % payoffs
    
    % sample from standard normal distribution (Nxm)
    Z = norminv(U);  % inverse sampling

    for j=1:m
        % compute drift and volatilty
        drift = (mu - 0.5*sigma^2)*dt(j);
        vol = sigma * sqrt(dt(j));
        
        Zj = Z(:,j);  % N random numbers for simulations at j-th observation date (Nx1)
        
        % update paths and their states (Nxk)
        S(alive) = S(alive) .* exp(drift + vol .* Zj(alive));
        hit = alive & (S>=B);
        alive(hit) = false;
    end
    
    % compute (discounted) payoffs (only paths alive, others 0)
    P(alive) = exp(-r*(t(m+1)-t(1))) .* max(S(alive)-K,0);  % (Nxk)
end

% One-Step Survival Monte Carlo without conditioning on "in-the-money"
function P = payoffs_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma, U)
    
    N = size(U,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu = r-d;
    
    % initialize variables (Nxk)
    S = S0 .* ones(N,1);  % paths
    W = ones(N,1);  % likelihood ratios
    
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
        Z = norminv(p .* Uj);  % inverse sampling

        % update paths
        S = S .* exp(drift + vol.*Z);
    end

    % update (discounted) weighted payoffs
    P = exp(-r*(t(m+1)-t(1))) .* W .* max(S-K,0);
end

% One-Step Survival Monte Carlo conditioning on "in-the-money"
function P = payoffs_upOutCallBarrier_OSS2(S0, K, B, t, r, d, sigma, U)
    
    N = size(U,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu = r-d;
    
    % initialize variables (Nxk)
    S = S0 .* ones(N,1);  % paths
    W = ones(N,1);  % likelihood ratios
    
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
        Z = norminv(p .* Uj);  % inverse sampling

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
    
    arg = min(max(p1 + p .* Um, 1e-15), 1-1e-15);  % clamp probability
    Z = norminv(arg);  % inverse sampling

    % update paths
    S = S .* exp(drift + vol.*Z);

    % update (discounted) weighted payoffs
    P = exp(-r*(t(m+1)-t(1))) .* W .* (S-K);
end

% Compute runtime factor for OSS1 wrt SMC
function adj_factor = RuntimeFactorOSS1(S0, K, B, t, r, d, sigma, U)

    % warm up the functions (optional but recommended)
    payoffs_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma, U);
    payoffs_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma, U);

    % define anonymous wrappers for timeit
    f_SMC  = @() payoffs_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma, U);
    f_OSS  = @() payoffs_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma, U);

    % compute average runtime per path using timeit
    N = size(U,1);  % number of simulations
    time_per_pathSMC = timeit(f_SMC) / N;
    time_per_pathOSS1 = timeit(f_OSS) / N;

    % compute adjustment factor
    adj_factor = time_per_pathSMC / time_per_pathOSS1;
end

% Compute runtime factor for OSS2 wrt SMC
function adj_factor = RuntimeFactorOSS2(S0, K, B, t, r, d, sigma, U)

    % warm up the functions (optional but recommended)
    payoffs_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma, U);
    payoffs_upOutCallBarrier_OSS2(S0, K, B, t, r, d, sigma, U);

    % define anonymous wrappers for timeit
    f_SMC  = @() payoffs_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma, U);
    f_OSS  = @() payoffs_upOutCallBarrier_OSS2(S0, K, B, t, r, d, sigma, U);

    % compute average runtime per path using timeit
    N = size(U,1);  % number of simulations
    time_per_pathSMC = timeit(f_SMC) / N;
    time_per_pathOSS2 = timeit(f_OSS) / N;

    % compute adjustment factor
    adj_factor = time_per_pathSMC / time_per_pathOSS2;
end