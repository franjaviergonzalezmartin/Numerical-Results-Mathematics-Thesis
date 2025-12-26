% Script to plot price and Delta of univariate autocallables option against initial price using two methods:
% 1) Standard Monte Carlo (SMC)
% 2) One-Step Survival Monte Carlo (OSS)

% =========================================================================
% Parameters
% =========================================================================
clear
seed           = 12345;                         % fixed random seed
S0_vec         = linspace(1e-5,10000,1000);     % initial asset prices
k              = numel(S0_vec);                 % number of S0 points
Sref           = 4000;                          % reference price
B              = 1;                             % barrier
t              = [0,1,2];                       % observation dates
m              = length(t)-1;                   % number of observation dates
Q              = [110, 120];                    % early coupons
r              = 0.04;                          % risk-free rate
d              = 0;                             % dividend yield
sigma          = 0.3;                           % volatility
N_SMC          = 1e2;                           % number of simulations for SMC
adj_factorOSS  = 1;                             % adjustment factor for OSS (precomputed in different script)
N_OSS          = round(adj_factorOSS * N_SMC);  % number of simulations for OSS (runtime-adjusted)
rng(seed);                                      % ensure same seed for fair comparison 
U              = rand(max(N_SMC, N_OSS),k);     % master matrix with random numbers
U_SMC          = U(1:N_SMC,:);                  % random numbers for SMC
U_OSS          = U(1:N_OSS,:);                  % random numbers for OSS
h_vec          = 5e-3 * S0_vec;                 % step size in finite difference for Delta


% Print parameters
fprintf('================================================================================= \n');
fprintf('Parameters used:\n');
fprintf('  Fixed random seed                                = %d\n', seed);
fprintf('  Intial prices S0                                 = %.2f to %.2f (step %.4f)\n', S0_vec(1), S0_vec(end), S0_vec(2)-S0_vec(1));
fprintf('  Number of S0 points k                            = %d\n', k);
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
fprintf('  Step size h range                                = [%.4f, %.4f]\n\n', min(h_vec), max(h_vec));


% =========================================================================
% Plot 1: Option price
% =========================================================================

V_SMC = univAutocallable_SMC(S0_vec, Sref, B, Q, t, r, d, sigma, U_SMC);
V_OSS = univAutocallable_OSS(S0_vec, Sref, B, Q, t, r, d, sigma, U_OSS);

figure(1);
plot(S0_vec, V_SMC, 'g.', 'DisplayName', 'Standard Monte Carlo');
hold on;
plot(S0_vec, V_OSS, 'r.', 'DisplayName', 'OSS Monte Carlo');
hold off
legend show;
xlabel('Asset Price S_0');
ylabel('Option Price V');
title('Univariate Autocallable Price: SMC vs OSS');
ylim([0 120])


% =========================================================================
% Plot 2: Delta
% =========================================================================

dV_SMC = Delta_univAutocallable_SMC(S0_vec, Sref, B, Q, t, r, d, sigma, U_SMC, h_vec);
dV_OSS = Delta_univAutocallable_OSS(S0_vec, Sref, B, Q, t, r, d, sigma, U_OSS, h_vec);

figure(2);
plot(S0_vec, dV_SMC, 'g.', 'DisplayName', 'Standard Monte Carlo');
hold on;
plot(S0_vec, dV_OSS, 'r.', 'DisplayName', 'OSS Monte Carlo');
hold off
legend show;
xlabel('Asset Price S_0');
ylabel('Delta Price dV');
title('Univariate Autocallable Delta Price: SMC vs OSS');
ylim([-0.01 0.03])


% =========================================================================
% Functions (vertorized form)
% -> tailored for plotting against S0_vec
% =========================================================================

% standard Monte Carlo
function V = univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U)
   
    S0 = S0(:)';  % ensure S0 is a row vector
    k = numel(S0);  % number of inital prices S0
    N = size(U,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu = r-d;

    % initialize variables (Nxk)
    S = repmat(S0,N,1);  % paths
    alive = true(N,k);  % keep track of paths alive
    payoff = zeros(N,k);  % payoffs

    % sample from standard normal distribution (Nxm)
    Z = norminv(U);  % inverse sampling

    for j=1:m
        % compute drift and volatility
        drift = (mu - 0.5*sigma^2)*dt(j);
        vol = sigma * sqrt(dt(j));

        Zj = Z(:,j);  % N random numbers for simulations at j-th observation date (Nx1)
        Zj_mat = repmat(Zj,1,k);  % same N random numbers for each initial price S0 (Nxk)
        
        % update paths and their states
        S(alive) = S(alive) .* exp(drift + vol .* Zj_mat(alive));
        autocalled = alive & ((S./Sref) >= B);
        alive(autocalled) = false;
        
        % compute (discounted) early payoffs
        payoff(autocalled) = exp(-r*(t(j+1)-t(1))) * Q(j);
    end
    
    % compute (discounted) final payoffs and option value
    payoff(alive) = exp(-r*(t(m+1)-t(1))) .* 100.*(S(alive)./Sref); % (Nxk)
    V = mean(payoff,1);  % mean over simulations - one for each initial S0 (1xk)
end

function dV = Delta_univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U, h_vec)
    % Forward finite differences OSS Monte Carlo with CRN
    V = univAutocallable_SMC(S0, Sref, B, Q, t, r, d, sigma, U);
    Vh = univAutocallable_SMC(S0+h_vec, Sref, B, Q, t, r, d, sigma, U);
    dV = (Vh - V)./h_vec;
end


% One-Step Survival Monte Carlo
function V = univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U)

    S0 = S0(:)';  % ensure S0 is a row vector    
    k = numel(S0);
    N = size(U,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu = r-d;
    
    % initilize variables (Nxk)
    S = repmat(S0,N,1);  % paths
    W = ones(N,k);  % likelihood weights
    payoff = zeros(N,k);  % payoffs

    for j=1:m
        % compute drift and volatility
        drift = (mu - 0.5*sigma^2) * dt(j);
        vol = sigma * sqrt(dt(j));
        
        % compute survival probabilities
        b = (log((B*Sref)./S) - drift) ./ vol;
        p = normcdf(b);
        p = min(p,1-1e-15);  % clamp probability to avoid underflow
        
        % update payoffs and likelihood ratio
        payoff = payoff + W .* (1-p) * Q(j) * exp(-r*(t(j+1)-t(1)));
        W = W .* p;
        
        % sample from truncated standard normal distribution
        Uj = U(:,j);  % N random numbers for simulations at j-th observation date (Nx1)
        Uj_mat = repmat(Uj,1,k);  % same N random numbers for each initial price S0 (Nxk)
        Z = norminv(p .* Uj_mat);  % inverse sampling

        % update paths
        S = S .* exp(drift + vol.*Z);
    end
    
    % update (discounted) weighted payoffs
    payoff = payoff + W .* 100.*(S./Sref) * exp(-r*(t(m+1)-t(1)));  % (Nxk)
    % compute option value
    V = mean(payoff,1);  % mean over simulations - one for each initial S0 (1xk)
end

function dV = Delta_univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U, h_vec)
    % Forward finite differences OSS Monte Carlo with CRN
    V = univAutocallable_OSS(S0, Sref, B, Q, t, r, d, sigma, U);
    Vh = univAutocallable_OSS(S0+h_vec, Sref, B, Q, t, r, d, sigma, U);
    dV = (Vh - V)./h_vec;
end
