% Script to plot price and Delta of European call option against initial price using three methods:
% 1) Black–Scholes closed-form formula (BSF)
% 2) Standard Monte Carlo (SMC)
% 3) One-Step Survival Monte Carlo (OSS)

% =========================================================================
% Parameters
% =========================================================================
clear
seed           = 12345;                         % fixed random seed
S0_vec         = linspace(1e-5,100,1000);       % vector of initial asset prices
k              = numel(S0_vec);                 % number of points S0
K              = 50;                            % strike
t0             = 0;                             % current time
tm             = 1;                             % maturity in years
r              = 0.1;                           % risk-free rate
d              = 0;                             % dividend yield
sigma          = 0.2;                           % volatility
N_SMC          = 1e2;                           % number of simulations for SMC
adj_factorOSS  = 0.75;                          % adjustment factor for OSS1 (precomputed in different script)
N_OSS          = round(adj_factorOSS * N_SMC);  % number of simulations for OSS1 (runtime-adjusted)
rng(seed);                                      % ensure same seed for fair comparison 
U              = rand(1,max(N_SMC, N_OSS));     % master matrix with random numbers
U_SMC          = U(1:N_SMC);                    % random numbers for SMC
U_OSS          = U(1:N_OSS);                    % random numbers for OSS
h_vec          = 5e-3 * S0_vec;                 % step size in finite difference for Delta

% Print parameters
fprintf('================================================================================== \n');
fprintf('Parameters used:\n');
fprintf('  Fixed random seed                                = %d\n', seed);
fprintf('  Range of initial prices                          = %.2f to %.2f (step %.4f)\n', S0_vec(1), S0_vec(end), S0_vec(2)-S0_vec(1));
fprintf('  Number of S0_vec points k                        = %d\n', k);
fprintf('  Strike K                                         = %.2f\n', K);
fprintf('  Current time t0                                  = %.2f\n', t0);
fprintf('  Maturity tm                                      = %.2f years\n', tm);
fprintf('  Interest rate r                                  = %.4f\n', r);
fprintf('  Dividend yield b                                 = %.4f\n', d);
fprintf('  Volatility sigma                                 = %.4f\n', sigma);
fprintf('  Simulations for SMC N_SMC                        = %d\n', N_SMC);
fprintf('  Random numbers for SMC U_SMC                     ~ U(0,1)\n');
fprintf('  Runtime adjustment-factor for OSS adj_factorOSS  = %.2f\n', adj_factorOSS)
fprintf('  Simulations for OSS N_OSS                        = %d\n', N_OSS);
fprintf('  Random numbers for OSS U_OSS                     ~ U(0,1)\n');
fprintf('  Step size h range                                = [%.4f, %.4f]\n\n', min(h_vec), max(h_vec));


% =========================================================================
% Plot 1: Option price
% =========================================================================
V_BSF = eurCall_BSF(S0_vec, K, t0, tm, r, d, sigma);
V_SMC = eurCall_SMC(S0_vec, K, t0, tm, r, d, sigma, U_SMC);
V_OSS = eurCall_OSS(S0_vec, K, t0, tm, r, d, sigma, U_OSS);

figure(1);
plot(S0_vec, V_BSF, 'k-','DisplayName', 'Black-Scholes Formula');
hold on;
plot(S0_vec, V_SMC, 'g.', 'DisplayName', 'Standard Monte Carlo');
plot(S0_vec, V_OSS, 'r.', 'DisplayName', 'OSS Monte Carlo');
hold off;
legend show;
xlabel('Asset Price S_0');
ylabel('Option Price V');
title('European Call Price: BSF vs SMC vs OSS');
ylim([-10 70])


% =========================================================================
% Plot 2: Delta
% =========================================================================

dV_BSF = Delta_eurCall_BSF(S0_vec, K, t0, tm, r, d, sigma);
dV_SMC = Delta_eurCall_SMC(S0_vec, K, t0, tm, r, d, sigma, U_SMC, h_vec);
dV_OSS = Delta_eurCall_OSS(S0_vec, K, t0, tm, r, d, sigma, U_OSS, h_vec);

figure(2);
plot(S0_vec, dV_BSF, 'k-','DisplayName', 'Black-Scholes Formula');
hold on;
plot(S0_vec, dV_SMC, 'g.', 'DisplayName', 'Standard Monte Carlo');
plot(S0_vec, dV_OSS, 'r.', 'DisplayName', 'OSS Monte Carlo');
hold off;
legend show;
xlabel('Asset Price S_0');
ylabel('Delta dV');
title('European Call Delta: BSF vs SMC vs OSS');
ylim([-0.1 1.1])


% =========================================================================
% Functions (vertorized form)
% -> Tailored for plotting against S0_vec
% =========================================================================

% Black-Scholes Formulae
function V = eurCall_BSF(S0_vec, K, t0, tm, r, d, sigma)
    % Black-Scholes formula
    mu = r-d;
    dt = tm-t0;
    d1 = (log(S0_vec./K) + (mu + 0.5*sigma^2)*dt)/(sigma*sqrt(dt));
    d2 = d1 - sigma*sqrt(dt);
    V = S0_vec.*exp(-d*dt).*normcdf(d1) - K*exp(-r*dt).*normcdf(d2);
end

function dV = Delta_eurCall_BSF(S0_vec, K, t0, tm, r, d, sigma)
    % Delta with Black-Scholes formula (partial derivative wrt S0)
    mu = r-d;
    dt = tm-t0;
    d1 = (log(S0_vec./K) + (mu + 0.5*sigma^2)*dt)/(sigma*sqrt(dt));
    dV = exp(-d*dt).*normcdf(d1);
end

% Standard Monte Carlo
function V = eurCall_SMC(S0_vec, K, t0, tm, r, d, sigma, U)
    
    S0_vec = S0_vec(:);  % ensure S0 is a column vector

    % compute drift and volatility
    dt = tm-t0;
    mu = r-d;    
    drift = (mu - 0.5*sigma^2)*dt;
    vol = sigma*sqrt(dt);
    
    % sample from standard normal distribution (1xN)
    Z = norminv(U);  % inverse sampling 

    % simulate assets' prices at maturity
    Sm = S0_vec .* exp(drift + vol.*Z);  % column vector * row vector
    
    % compute (discounted) payoffs
    payoff = exp(-r*dt) .* max(Sm-K,0);  
    
    % compute option values
    V = mean(payoff,2);  % mean across dimension 2 of the matrix (columns)
end

function dV = Delta_eurCall_SMC(S0_vec, K, t0, tm, r, d, sigma, U, h_vec)
    
    % Forward finite differences Monte Carlo with CRN
    V = eurCall_SMC(S0_vec, K, t0, tm, r, d, sigma, U);
    Vh = eurCall_SMC(S0_vec+h_vec, K, t0, tm, r, d, sigma, U);
    dV = (Vh - V)./h_vec(:);
end

% One-Step Survival Monte Carlo
function V = eurCall_OSS(S0_vec, K, t0, tm, r, d, sigma, U)

    S0_vec = S0_vec(:);  % ensure S0 is a column vector 

    % compute drift and volatility terms
    dt = tm-t0;
    mu = r-d;  
    drift = (mu - 0.5*sigma^2)*dt;  % drift term
    vol = sigma*sqrt(dt);  % volatility term
    
    % compute survival probability
    a = (log(K./S0_vec)-(mu-0.5*sigma^2)*dt) / (sigma*sqrt(dt));
    p = 1 - normcdf(a);
    p = max(p,1e-15);  % clamp probability to avoid underflow

    % sample from truncated standard normal distribution
    Z = norminv(1-p + p.*U);  % inverse sampling (1xN)

    % simulate assets' prices at maturity
    Sm = S0_vec .* exp(drift + vol*Z);  % column vector * row vector

    % compute (discounted) weighted payoffs
    payoff = exp(-r*dt).*p.*(Sm-K);

    % compute option values
    V = mean(payoff,2);  % mean across dimension 2 of the matrix (columns)
end

function dV = Delta_eurCall_OSS(S0_vec, K, t0, tm, r, d, sigma, U, h_vec)
    
    % Forward finite differences OSS Monte Carlo with CRN
    V = eurCall_OSS(S0_vec, K, t0, tm, r, d, sigma, U);
    Vh = eurCall_OSS(S0_vec+h_vec, K, t0, tm, r, d, sigma, U);
    dV = (Vh - V)./h_vec(:);
end
