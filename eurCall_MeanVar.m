% Script to compute mean and variance of option price, Delta and Vega of European call of two Monte Carlo methods:
% 1) Standard Monte Carlo (SMC)
% 2) One-Step Survival Monte Carlo (OSS)


% =========================================================================
% Parameters
% =========================================================================
clear
seed           = 12345;                         % fixed random seed
S0             = 50;                            % initial asset prices
K              = 50;                            % strike
t0             = 0;                             % current time
tm             = 1;                             % maturity in years
r              = 0.1;                           % risk-free rate
d              = 0;                             % dividend yield
sigma          = 0.2;                           % volatility
N_SMC          = 1e5;                           % number of simulations for SMC
adj_factorOSS  = 0.75;                          % adjustment factor for OSS1 (precomputed in different script)
N_OSS          = round(adj_factorOSS * N_SMC);  % number of simulations for OSS1 (runtime-adjusted)
rng(seed);                                      % ensure same seed for fair comparison 
U              = rand(1,max(N_SMC, N_OSS));     % master matrix with random numbers
U_SMC          = U(1:N_SMC);                    % random numbers for SMC
U_OSS          = U(1:N_OSS);                    % random numbers for OSS
hS0            = 5e-3 * S0;                     % step size in finite difference for Delta
hSigma         = 5e-3 * sigma;                  % step size in finite difference for Vega

% Print parameters
fprintf('===================================================================== \n');
fprintf('Parameters used:\n');
fprintf('  Fixed random seed                                 = %d\n', seed);
fprintf('  Initial price S0                                  = %.2f\n', S0);
fprintf('  Strike K                                          = %.2f\n', K);
fprintf('  Current time t0                                   = %.2f\n', t0);
fprintf('  Maturity tm                                       = %.2f years\n', tm);
fprintf('  Interest rate r                                   = %.4f\n', r);
fprintf('  Dividend yield b                                  = %.4f\n', d);
fprintf('  Volatility sigma                                  = %.4f\n', sigma);
fprintf('  Simulations for SMC N_SMC                         = %d\n', N_SMC);
fprintf('  Random numbers for SMC U_SMC                      ~ U(0,1)\n');
fprintf('  Runtime adjustment-factor for OSS adj_factorOSS   = %.2f\n', adj_factorOSS)
fprintf('  Simulations for OSS N_OSS                         = %d\n', N_OSS);
fprintf('  Random numbers for OSS U_OSS                      ~ U(0,1)\n');
fprintf('  Step size in finite differences for Delta hSo     = %.4f\n', hS0');
fprintf('  Step size in finite differences for Delta hSigma  = %.4f\n\n', hSigma');


% =========================================================================
% 1. Option price
% =========================================================================

P_SMC = eurCall_SMC(S0, K, t0, tm, r, d, sigma, U_SMC);
P_OSS = eurCall_OSS(S0, K, t0, tm, r, d, sigma, U_OSS);
fprintf('European Call Price: Variance\n')
fprintf('  Variance SMC  = %.4g\n', var(P_SMC,0));
fprintf('  Variance OSS  = %.4g\n\n', var(P_OSS,0));


% =========================================================================
% 2. Delta
% =========================================================================

P_deltaSMC = Delta_eurCall_SMC(S0, K, t0, tm, r, d, sigma, U_SMC, hS0);
P_deltaOSS = Delta_eurCall_OSS(S0, K, t0, tm, r, d, sigma, U_OSS, hS0);
fprintf('European Call Delta: Variance\n')
fprintf('  Variance SMC  = %.5g\n', var(P_deltaSMC,0));
fprintf('  Variance OSS  = %.5g\n\n', var(P_deltaOSS,0));


% =========================================================================
% 3. Vega
% =========================================================================

P_vegaSMC = Vega_eurCall_SMC(S0, K, t0, tm, r, d, sigma, U_SMC, hSigma);
P_vegaOSS = Vega_eurCall_OSS(S0, K, t0, tm, r, d, sigma, U_OSS, hSigma);
fprintf('European Call Vega: Variance\n')
fprintf('  Variance SMC  = %.4g\n', var(P_vegaSMC,0));
fprintf('  Variance OSS  = %.4g\n\n', var(P_vegaOSS,0));


% =========================================================================
% Functions (vertorized form - optimized in MATLAB)
% =========================================================================

% Standard Monte Carlo
function P = eurCall_SMC(S0, K, t0, tm, r, d, sigma, U)
    
    N = numel(U);

    % compute drift and volatility
    dt = tm-t0;
    mu = r-d;    
    drift = (mu - 0.5*sigma^2)*dt;
    vol = sigma*sqrt(dt);
    
    % sample from standard normal distribution (1xN)
    Z = norminv(U);  % inverse sampling 

    % simulate assets' prices at maturity
    Sm = ones(1,N) .* S0 .* exp(drift + vol.*Z);  % (1xN)
    
    % compute (discounted) payoffs
    P = exp(-r*dt) .* max(Sm-K,0);  % (1xN)
end

function deltaP = Delta_eurCall_SMC(S0, K, t0, tm, r, d, sigma, U, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = eurCall_SMC(S0, K, t0, tm, r, d, sigma, U);
    Ph = eurCall_SMC(S0+h, K, t0, tm, r, d, sigma, U);
    deltaP = (Ph - P)./h;
end

function vegaP = Vega_eurCall_SMC(S0, K, t0, tm, r, d, sigma, U, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = eurCall_SMC(S0, K, t0, tm, r, d, sigma, U);
    Ph = eurCall_SMC(S0, K, t0, tm, r, d, sigma+h, U);
    vegaP = (Ph - P)./h;
end


% One-Step Survival Monte Carlo
function P = eurCall_OSS(S0, K, t0, tm, r, d, sigma, U)
    
    N = numel(U);
    S0 = S0 .* ones(1,N);

    % compute drift and volatility terms
    dt = tm-t0;
    mu = r-d;  
    drift = (mu - 0.5*sigma^2)*dt;  % drift term
    vol = sigma*sqrt(dt);  % volatility term
    
    % compute survival probability
    a = (log(K./S0)-(mu-0.5*sigma^2)*dt) / (sigma*sqrt(dt));
    p = 1 - normcdf(a);
    p = max(p,1e-15);  % clamp probability to avoid underflow
    
    % sample from truncated standard normal distribution
    Z = norminv(1-p + p.*U);  % row vector (1xN)
        
    % simulate terminal values
    Sm = S0 .* exp(drift + vol*Z);  % (1xN)

    % compute (discounted) weighted payoffs
    P = exp(-r*dt).*p.*(Sm-K);
end

function deltaP = Delta_eurCall_OSS(S0, K, t0, tm, r, d, sigma, U, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = eurCall_OSS(S0, K, t0, tm, r, d, sigma, U);
    Ph = eurCall_OSS(S0+h, K, t0, tm, r, d, sigma, U);
    deltaP = (Ph - P)./h;
end

function vegaP = Vega_eurCall_OSS(S0, K, t0, tm, r, d, sigma, U, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = eurCall_OSS(S0, K, t0, tm, r, d, sigma, U);
    Ph = eurCall_OSS(S0, K, t0, tm, r, d, sigma+h, U);
    vegaP = (Ph - P)./h;
end
