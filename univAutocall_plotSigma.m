% Script to plot price and Vega of univariate autocallables option against volatility using two methods:
% 1) Standard Monte Carlo (SMC)
% 2) One-Step Survival Monte Carlo (OSS)

% =========================================================================
% Parameters
% =========================================================================
clear
seed           = 12345;                         % fixed random seed
sigma_vec      = linspace(1e-5,0.5,1000);       % volatility
k              = numel(sigma_vec);              % number of sigma points
S0             = 3500;                          % initial asset price S0
Sref           = 4000;                          % reference price
B              = 1;                             % barrier
t              = [0,1,2];                       % observation dates
m              = length(t)-1;                   % number of observation dates
Q              = [110, 120];                    % early coupons
r              = 0.04;                          % risk-free rate
d              = 0;                             % dividend yield
N_SMC          = 1e2;                           % number of simulations for SMC
adj_factorOSS  = 1.65;                          % adjustment factor for OSS (precomputed in different script)
N_OSS          = round(adj_factorOSS * N_SMC);  % number of simulations for OSS (runtime-adjusted)
rng(seed);                                      % ensure same seed for fair comparison 
U = rand(max(N_SMC, N_OSS),k);                  % master matrix with random numbers
U_SMC  = U(1:N_SMC,:);                          % random numbers for SMC
U_OSS = U(1:N_OSS,:);                           % random numbers for OSS
h_vec     = 5e-3 * sigma_vec;                   % step size in finite difference for Vega

% Print parameters
fprintf('=============================================================================== \n');
fprintf('Parameters used:\n');
fprintf('  Fixed random seed                                = %d\n', seed);
fprintf('  Volatility sigma                                 = %.2f to %.2f (step %.4f)\n', sigma_vec(1), sigma_vec(end), sigma_vec(2)-sigma_vec(1));
fprintf('  Number of sigma points k                         = %d\n', k);
fprintf('  Intial prices S0                                 = %.2f\n', S0);
fprintf('  Reference price Sref                             = %.2f\n', Sref);
fprintf('  Barrier B                                        = %.2f\n', B);
fprintf('  Observation dates t                              = %s\n', mat2str(t));
fprintf('  Number of observation dates m                    = %d\n', m);
fprintf('  Early coupons Q                                  = %s\n', mat2str(Q));
fprintf('  Interest rate r                                  = %.4f\n', r);
fprintf('  Dividend yield b                                 = %.4f\n', d);
fprintf('  Number of simulations for SMC N_SMC              = %d\n', N_SMC);
fprintf('  Random numbers for SMC U_SMC                     ~ U(0,1)\n');
fprintf('  Runtime adjustment-factor for OSS adj_factorOSS  = %.2f\n', adj_factorOSS)
fprintf('  Number of simulations for OSS N_OSS              = %d\n', N_OSS);
fprintf('  Random numbers for OSS U_OSS                     ~ U(0,1)\n');
fprintf('  Step size h range = [%.4f, %.4f]\n\n', min(h_vec), max(h_vec));


% =========================================================================
% Plot 1: Option price
% =========================================================================

V_SMC = univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma_vec, U_SMC);
V_OSS = univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma_vec, U_OSS);

figure(1);
plot(sigma_vec, V_SMC, 'g.', 'DisplayName', 'Standard Monte Carlo');
hold on;
plot(sigma_vec, V_OSS, 'r.', 'DisplayName', 'OSS Monte Carlo');
hold off
legend show;
xlabel('Volatility sigma');
ylabel('Option Price V');
title('Univariate Autocallable Price: SMC vs OSS');
ylim([60 100])


% =========================================================================
% Plot 2: Vega
% =========================================================================

dV_SMC = Vega_univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma_vec, U_SMC, h_vec);
dV_OSS = Vega_univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma_vec, U_OSS, h_vec);

figure(2);
plot(sigma_vec, dV_SMC, 'g.', 'DisplayName', 'Standard Monte Carlo');
hold on;
plot(sigma_vec, dV_OSS, 'r.', 'DisplayName', 'OSS Monte Carlo');
hold off
legend show;
xlabel('Volatility sigma');
ylabel('Vega dV');
title('Univariate Autocallable Vega: SMC vs OSS');
ylim([-50 150])


% =========================================================================
% Functions (vertorized form - optimized in MATLAB)
% -> tailored for plotting against Sigma_vec
% =========================================================================


function V = univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U)
   
    sigma = sigma(:)';  % ensure S0 is a row vector
    k = numel(sigma);  % number of inital prices S0
    N = size(U,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu = r-d;

    % initialize variables (Nxk)
    S = repmat(S0,N,k);  % paths
    alive = true(N,k);  % keep track of paths alive
    payoff = zeros(N,k);  % payoffs

    % sample from standard normal distribution (Nxm)
    Z = norminv(U);  % inverse sampling

    for j=1:m
        % compute drift and volatility
        drift = (mu - 0.5.*sigma.^2).*dt(j);
        drift_mat = repmat(drift,N,1);
        vol = sigma .* sqrt(dt(j));
        vol_mat = repmat(vol,N,1);

        Zj = Z(:,j);  % N random numbers for simulations at j-th observation date (Nx1)
        Zj_mat = repmat(Zj,1,k);  % same N random numbers for each initial price S0 (Nxk)
        
        % update paths and their states
        S(alive) = S(alive) .* exp(drift_mat(alive) + vol_mat(alive) .* Zj_mat(alive));
        autocalled = alive & ((S./Sref) >= B);
        alive(autocalled) = false;
        
        % compute (discounted) early payoffs
        payoff(autocalled) = exp(-r*(t(j+1)-t(1))) * Q(j);
    end
    
    % compute (discounted) final payoffs and option value
    payoff(alive) = exp(-r*(t(m+1)-t(1))) .* 100.*(S(alive)./Sref); % (Nxk)
    V = mean(payoff,1);  % mean over simulations - one for each initial S0 (1xk)
end

function dV = Vega_univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U, h_vec)
    % Forward finite differences OSS Monte Carlo with CRN
    V = univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U);
    Vh = univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma+h_vec, U);
    dV = (Vh - V)./h_vec;
end


function V = univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U)

    sigma = sigma(:)';  % ensure S0 is a row vector    
    k = numel(sigma);
    N = size(U,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu = r-d;
    
    % initilize variables (Nxk)
    S = repmat(S0,N,k);  % paths
    W = ones(N,k);  % likelihood weights
    payoff = zeros(N,k);  % payoffs

    for j=1:m
        % compute drift and volatility
        drift = (mu - 0.5.*sigma.^2) * dt(j);
        drift_mat = repmat(drift,N,1);
        vol = sigma .* sqrt(dt(j));
        vol_mat = repmat(vol,N,1);
        
        % compute survival probabilities
        b = (log((B*Sref)./S) - drift_mat) ./ vol_mat;
        p = normcdf(b);
        p = min(p,1-1e-15);  % clamp probability to avoid underflow
        
        % update payoffs and likelihood ratio
        payoff = payoff + W .* (1-p) * Q(j) * exp(-r*(t(j+1)-t(1)));
        W = W .* p;
        p = min(p,1-1e-15);  % clamp probability to avoid underflow
        
        % sample from truncated standard normal distribution
        Uj = U(:,j);  % N random numbers for simulations at j-th observation date (Nx1)
        Uj_mat = repmat(Uj,1,k);  % same N random numbers for each initial price S0 (Nxk)
        Z = norminv(p .* Uj_mat);  % inverse sampling

        % update paths
        S = S .* exp(drift_mat + vol_mat.*Z);
    end
    
    % update (discounted) weighted payoffs
    payoff = payoff + W .* 100.*(S./Sref) * exp(-r*(t(m+1)-t(1)));  % (Nxk)
    % compute option value
    V = mean(payoff,1);  % mean over simulations - one for each initial S0 (1xk)
end

function dV = Vega_univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U, h_vec)
    % Forward finite differences OSS Monte Carlo with CRN
    V = univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U);
    Vh = univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma+h_vec, U);
    dV = (Vh - V)./h_vec;
end
