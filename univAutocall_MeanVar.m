% Script to compute mean and variance of option price, Delta and Vega of univariate autocallale of two Monte Carlo methods:
% 1) Standard Monte Carlo (SMC)
% 2) One-Step Survival Monte Carlo (OSS)

% =========================================================================
% Parameters
% =========================================================================
clear
seed           = 12345;                         % fixed random seed
S0             = 3500;                          % initial asset prices
Sref           = 4000;                          % strike
B              = 1;                             % barrier
t              = [0,1,2];                       % observation dates
m              = length(t)-1;                   % number of observation dates
Q              = [110, 120];                    % early coupons
r              = 0.04;                          % risk-free rate
d              = 0;                             % dividend yield
sigma          = 0.3;                           % volatility
N_SMC          = 1e2;                           % number of simulations for SMC
adj_factorOSS  = 1.65;                          % adjustment factor for OSS1 (precomputed in different script)
N_OSS          = round(adj_factorOSS * N_SMC);  % number of simulations for OSS1 (runtime-adjusted)
rng(seed);                                      % ensure same seed for fair comparison 
U              = rand(max(N_SMC, N_OSS),m);     % master matrix with random numbers
U_SMC          = U(1:N_SMC,:);                  % random numbers for SMC
U_OSS          = U(1:N_OSS,:);                  % random numbers for OSS
hS0            = 5e-3 * S0;                     % step size in finite difference for Delta
hSigma         = 5e-3 * sigma;                  % step size in finite difference for Sigma

% Print parameters
fprintf('============================================================== \n');
fprintf('Parameters used:\n');
fprintf('  Fixed random seed                                = %d\n', seed);
fprintf('  Initial price S0                                 = %.2f\n', S0);
fprintf('  Reference price Sref                             = %.2f\n', Sref);
fprintf('  Barrier B                                        = %.2f\n', B);
fprintf('  Observation dates t                              = %s\n', mat2str(t));
fprintf('  Number of observation dates m                    = %d\n', m);
fprintf('  Early coupons Q                                  = %s\n', mat2str(Q));
fprintf('  Interest rate r                                  = %.4f\n', r);
fprintf('  Dividend yield b                                 = %.4f\n', d);
fprintf('  Volatility sigma                                 = %.4f\n', sigma);
fprintf('  Number of simulations for SMC N_SMC              = %d\n', N_SMC);
fprintf('  Random numbers for SMC U_SMC                     ~ U(0,1)\n');
fprintf('  Runtime adjustment-factor for OSS adj_factorOSS  = %.2f\n', adj_factorOSS)
fprintf('  Number of simulations for OSS N_OSS              = %d\n', N_OSS);
fprintf('  Random numbers for OSS U_OSS                     ~ U(0,1)\n');
fprintf('  Step size for Delta hSo                          = %.4f\n', hS0');
fprintf('  Step size for Vega hSigma                        = %.4f\n\n', hSigma');


% =========================================================================
% 1. Option price
% =========================================================================

P_SMC = payoffs_univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U_SMC);
P_OSS = payoffs_univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U_OSS);
fprintf('Univariate Autocallable Price: Variance\n');
fprintf('  Variance SMC  = %.6f\n', var(P_SMC,0));
fprintf('  Variance OSS  = %.6f\n\n', var(P_OSS,0));

% =========================================================================
% 2. Delta
% =========================================================================

P_deltaSMC = payoffs_Delta_univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U_SMC, hS0);
P_deltaOSS = payoffs_Delta_univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U_OSS, hS0);
fprintf('Univariate Autocallable Delta: Variance\n');
fprintf('  Variance SMC  = %.6f\n', var(P_deltaSMC,0));
fprintf('  Variance OSS  = %.9f\n\n', var(P_deltaOSS,0));

% =========================================================================
% 3. Vega
% =========================================================================

P_vegaSMC = payoffs_Vega_univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U_SMC, hSigma);
P_vegaOSS = payoffs_Vega_univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U_OSS, hSigma);
fprintf('Univariate Autocallable Vega: Variance\n');
fprintf('  Variance SMC  = %.6f\n', var(P_vegaSMC,0));
fprintf('  Variance OSS  = %.6f\n\n', var(P_vegaOSS,0));

% =========================================================================
% Functions (vertorized form - optimized in MATLAB)
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

function deltaP = payoffs_Delta_univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U);
    Ph = payoffs_univAutocallable_SMC(S0+h, Sref, B, Q, t, r, d, sigma, U);
    deltaP = (Ph - P)./h;
end

function vegaP = payoffs_Vega_univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U);
    Ph = payoffs_univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma+h, U);
    vegaP = (Ph - P)./h;
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

function deltaP = payoffs_Delta_univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U);
    Ph = payoffs_univAutocallable_OSS(S0+h, Sref, B, Q, t, r, d, sigma, U);
    deltaP = (Ph - P)./h;
end

function vegaP = payoffs_Vega_univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U);
    Ph = payoffs_univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma+h, U);
    vegaP = (Ph - P)./h;
end

