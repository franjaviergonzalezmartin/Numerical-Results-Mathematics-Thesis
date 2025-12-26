% Script to find adjusted runtimes for Standard Monte Carlo (SMC)
% and One-Step Survival Monte Carlo (OSS) univariate autocallable options

% =========================================================================
% Parameters
% =========================================================================
clear
seed    = 12345;        % fixed random seed
S0      = 3500;         % initial asset prices
Sref    = 4000;         % strike
B       = 1;            % barrier
t       = [0,1,2];      % observation dates
m       = length(t)-1;  % number of observation dates
Q       = [110, 120];   % early coupons
r       = 0.04;         % risk-free rate
d       = 0;            % dividend yield
sigma   = 0.3;          % volatility
N       = 1e2;          % number of simulations
rng(seed);              % ensure same seed for fair comparison
U       = rand(N,m);    % Nxm samples from U(0,1)


adj_factorOSS = RuntimeFactorOSS(S0, Sref, B, Q, t, r, d, sigma, U);
fprintf('Adjustment factor OSS to SMC:  %.4f\n', adj_factorOSS)


% =========================================================================
% Functions (vertorized form)
% =========================================================================

% Standard Monte Carlo
function P = payoffs_univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U)
    
    N = size(U,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu = r-d;  % risk-neutral drift parameter

    % initialize variables (Nx1)
    S = S0 .* ones(N,1);  % paths
    alive = true(N,1);  % keep track of paths alive
    P = zeros(N,1);  % payoffs

    % sample from standard normal distribution (Nx1)
    Z = norminv(U);  % inverse sampling

    for j=1:m
        % compute drift and volatility
        drift = (mu - 0.5*sigma^2)*dt(j);
        vol = sigma * sqrt(dt(j));

        Zj = Z(:,j);  % N random numbers for simulations at j-th observation date (Nx1)
        
        % update paths and their states
        S(alive) = S(alive) .* exp(drift + vol .* Zj(alive));
        autocalled = alive & ((S./Sref) >= B);
        alive(autocalled) = false;
        
        % compute (discounted) early payoffs
        P(autocalled) = exp(-r*(t(j+1)-t(1))) * Q(j);
    end
    
    % compute (discounted) redemption payoffs
    P(alive) = exp(-r*(t(m+1)-t(1))) .* 100.*(S(alive)./Sref); % (Nxk)    
end

% One-Step Survival Monte Carlo
function P = payoffs_univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U)
    
    N = size(U,1);
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    mu = r-d;  % risk-neutral drift parameter

    % initilize variables (Nx1)
    S = S0 .* ones(N,1);  % paths
    W = ones(N,1);  % likelihood weights
    P = zeros(N,1);  % payoffs

    for j=1:m
        % compute drift and volatility
        drift = (mu - 0.5*sigma^2) * dt(j);
        vol = sigma * sqrt(dt(j));
        
        % compute survival probabilities
        b = (log((B*Sref)./S) - drift) ./ vol;
        p = normcdf(b);
        p = min(p,1-1e-15);  % clamp probability to avoid underflow
        
        % update payoffs and likelihood ratio
        P = P + W .* (1-p) * Q(j) * exp(-r*(t(j+1)-t(1)));
        W = W .* p;
        
        % sample from truncated standard normal distribution
        Uj = U(:,j);  % (Nx1)
        Z = norminv(p .* Uj);  % inverse sampling

        % update paths
        S = S .* exp(drift + vol.*Z);
    end
    
    % update (discounted) weighted payoffs
    P = P + W .* 100.*(S./Sref) * exp(-r*(t(m+1)-t(1)));  % (Nx1)
end

% Compute runtime factor for OSS wrt SMC
function adj_factor = RuntimeFactorOSS(S0, Sref, B, Q, t, r, d, sigma, U)

    % warm up the functions (optional but recommended)
    payoffs_univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U);
    payoffs_univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U);

    % define anonymous wrappers for timeit
    f_SMC  = @() payoffs_univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U);
    f_OSS  = @() payoffs_univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U);

    % compute average runtime per path using timeit
    N = size(U,1);  % number of simulations
    time_per_pathSMC = timeit(f_SMC) / N;
    time_per_pathOSS = timeit(f_OSS) / N;

    % compute adjustment factor
    adj_factor = time_per_pathSMC / time_per_pathOSS;
end