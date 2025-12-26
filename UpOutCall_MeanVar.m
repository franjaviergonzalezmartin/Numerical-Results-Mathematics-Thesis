% Script to compute mean and variance of option price, Delta and Vega of univariate autocallale of three Monte Carlo methods:
% 1) Standard Monte Carlo (SMC)
% 2) One-Step Survival Monte Carlo (OSS1)
% 3) One-Step Survival Monte Carlo condtioning on ITM (OSS2)

% =========================================================================
% Parameters
% =========================================================================
clear
seed            = 32134;                                  % fixed random seed
S0              = 50;                                     % initial asset prices
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
hS0             = 5e-3 * S0;                              % step size in finite difference for Delta
hSigma          = 5e-3 * sigma;                           % step size in finite difference for Vega

% Print parameters
fprintf('================================================================================= \n');
fprintf('Parameters used:\n');
fprintf('  Fixed random seed                                  = %d\n', seed);
fprintf('  Initial price S0                                   = %.2f\n', S0);
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
fprintf('  Step size in finite differences for Delta hSo      = %.4f\n', hS0');
fprintf('  Step size in finite differences for Delta hSigma   = %.4f\n\n', hSigma');


% =========================================================================
% 1. Option price
% =========================================================================

P_SMC = payoffs_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma, U_SMC);
P_OSS1 = payoffs_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma, U_OSS1);
P_OSS2 = payoffs_upOutCallBarrier_OSS2(S0, K, B, t, r, d, sigma, U_OSS2);
fprintf('Univariate Autocallable Price: Variance\n');
fprintf('  Variance SMC   = %.4f\n', var(P_SMC,0));
fprintf('  Variance OSS1  = %.8f\n', var(P_OSS1,0));
fprintf('  Variance OSS2  = %.8f\n\n', var(P_OSS2,0));

% =========================================================================
% 2. Delta
% =========================================================================

P_deltaSMC = payoffs_Delta_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma, U_SMC, hS0);
P_deltaOSS1 = payoffs_Delta_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma, U_OSS1, hS0);
P_deltaOSS2 = payoffs_Delta_upOutCallBarrier_OSS2(S0, K, B, t, r, d, sigma, U_OSS2, hS0);
fprintf('Univariate Autocallable Delta: Variance\n');
fprintf('  Variance SMC   = %.8f\n', var(P_deltaSMC,0));
fprintf('  Variance OSS1  = %.15f\n', var(P_deltaOSS1,0));
fprintf('  Variance OSS2  = %.15f\n\n', var(P_deltaOSS2,0));

% =========================================================================
% 3. Vega
% =========================================================================

P_vegaSMC = payoffs_Vega_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma, U_SMC, hSigma);
P_vegaOSS1 = payoffs_Vega_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma, U_OSS1, hSigma);
P_vegaOSS2 = payoffs_Vega_upOutCallBarrier_OSS2(S0, K, B, t, r, d, sigma, U_OSS2, hSigma);
fprintf('Univariate Autocallable Vega: Variance\n');
fprintf('  Variance SMC   = %.4f\n', var(P_vegaSMC,0));
fprintf('  Variance OSS1  = %.10f\n', var(P_vegaOSS1,0));
fprintf('  Variance OSS2  = %.10f\n\n', var(P_vegaOSS2,0));

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

function deltaP = payoffs_Delta_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma, U, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma, U);
    Ph = payoffs_upOutCallBarrier_SMC(S0+h, K, B, t, r, d, sigma, U);
    deltaP = (Ph - P)./h;
end

function vegaP = payoffs_Vega_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma, U, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma, U);
    Ph = payoffs_upOutCallBarrier_SMC(S0, K, B, t, r, d, sigma+h, U);
    vegaP = (Ph - P)./h;
end


% One-Step Survival Monte Carlo (without conditioning on "in-the-money")
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

function deltaP = payoffs_Delta_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma, U, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma, U);
    Ph = payoffs_upOutCallBarrier_OSS1(S0+h, K, B, t, r, d, sigma, U);
    deltaP = (Ph - P)./h;
end

function vegaP = payoffs_Vega_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma, U, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma, U);
    Ph = payoffs_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma+h, U);
    vegaP = (Ph - P)./h;
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
        p = min(p,1-1e-15);  % clamp probability to avoid underflow
        
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

function deltaP = payoffs_Delta_upOutCallBarrier_OSS2(S0, K, B, t, r, d, sigma, U, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma, U);
    Ph = payoffs_upOutCallBarrier_OSS1(S0+h, K, B, t, r, d, sigma, U);
    deltaP = (Ph - P)./h;
end

function vegaP = payoffs_Vega_upOutCallBarrier_OSS2(S0, K, B, t, r, d, sigma, U, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma, U);
    Ph = payoffs_upOutCallBarrier_OSS1(S0, K, B, t, r, d, sigma+h, U);
    vegaP = (Ph - P)./h;
end


