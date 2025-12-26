% Script to find adjusted runtimes for Standard Monte Carlo (SMC)
% and One-Step Survival Monte Carlo (OSS) in European call options

% =========================================================================
% Parameters
% =========================================================================
clear
seed   = 12345;         % fixed random seed
S0     = 50;            % initial asset prices
K      = 50;            % strike
T      = 1;             % maturity in years
r      = 0.1;           % risk-free rate
d      = 0;             % dividend yield
sigma  = 0.2;           % volatility
N      = 5e5;           % number of simulations
rng(seed)               % ensure same seed for fair comparison
U      = rand(1,N);     % random numbers


adj_factor = RuntimeFactorOSS(S0, K, T, r, d, sigma, U);
fprintf('Adjustment factor OSS to SMC:  %.4f\n', adj_factor)


% =========================================================================
% Functions (vertorized form)
% =========================================================================

% Standard Monte Carlo
function P = payoffs_eurCall_SMC(S0, K, T, r, d, sigma, U)
    
    N = numel(U);

    % compute drift and volatility
    mu = r-d;    
    drift = (mu - 0.5*sigma^2)*T;
    vol = sigma*sqrt(T);
    
    % sample from standard normal distribution (1xN)
    Z = norminv(U);  % inverse sampling 

    % simulate assets' prices at maturity
    ST = ones(1,N) .* S0 .* exp(drift + vol.*Z);  % (1xN)
    
    % compute (discounted) payoffs
    P = exp(-r*T) .* max(ST-K,0);  % (1xN)
end

% One-Step Survival Monte Carlo
function P = payoffs_eurCall_OSS(S0, K, T, r, d, sigma, U)
    
    N = numel(U);
    S0 = S0 .* ones(1,N);
    % compute drift and volatility terms
    mu = r-d;  
    drift = (mu - 0.5*sigma^2)*T;  % drift term
    vol = sigma*sqrt(T);  % volatility term
    
    % compute survival probability
    a = (log(K./S0)-(mu-0.5*sigma^2)*T) / (sigma*sqrt(T));
    p = 1 - normcdf(a);
    p = max(p,1e-15);  % clamp probability to avoid underflow
    
    % sample from truncated standard normal distribution
    Z = norminv(1-p + p.*U);  % row vector (1xN)
        
    % simulate terminal values
    ST = S0 .* exp(drift + vol*Z);  % (1xN)

    % compute (discounted) weighted payoffs
    P = exp(-r*T).*p.*(ST-K);
end

% Compute runtime factor for OSS wrt SMC
function adj_factor = RuntimeFactorOSS(S0, K, T, r, d, sigma, U)

    % warm up the functions (optional but recommended)
    payoffs_eurCall_SMC(S0, K, T, r, d, sigma, U);
    payoffs_eurCall_OSS(S0, K, T, r, d, sigma, U);

    % define anonymous wrappers for timeit
    f_SMC  = @() payoffs_eurCall_SMC(S0, K, T, r, d, sigma, U);
    f_OSS  = @() payoffs_eurCall_OSS(S0, K, T, r, d, sigma, U);

    % compute average runtime per path using timeit
    N = size(U,1);  % number of simulations
    time_per_pathSMC = timeit(f_SMC) / N;
    time_per_pathOSS = timeit(f_OSS) / N;

    % compute adjustment factor
    adj_factor = time_per_pathSMC / time_per_pathOSS;
end