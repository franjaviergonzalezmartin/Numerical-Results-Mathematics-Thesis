% Script to find adjusted runtimes for Standard Monte Carlo (SMC)
% and One-Step Survival Monte Carlo (OSS) bivariate autocallable options
% (min-dependence)

% =========================================================================
% Parameters
% ========================================================================
clear
seed     = 12345;                      % fixed random seed
S10      = 3500;                       % initial asset price S1
S20      = 7000;                       % initial asset price S2
S1ref    = 4000;                       % reference price for S1
S2ref    = 8000;                       % reference price for S2
B        = 1;                          % barrier level
t        = [0,1,2,3,4,5];              % observation dates
m        = length(t)-1;                % number of observation dates
Q        = [110, 120, 130, 140, 150];  % early coupons
r        = 0.04;                       % risk-free rate
d1       = 0;                          % dividend yield of S1
d2       = 0;                          % dividend yield of S2
sigma1   = 0.3;                        % volatility of S1
sigma2   = 0.4;                        % volatility of S1
cor      = 0.5;                        % correlation in Brownian motion
N        = 1e2;                        % number of simulations
rng(seed);                             % ensure same seed for fair comparison (call before any random number)
U1       = rand(N,m);                  % Nxm samples from U(0,1) for S1
U2       = rand(N,m);                  % Nxm samples from U(0,1) for S2 

adj_factorROSS = RuntimeFactorOSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
fprintf('Adjustment factor OSS to SMC:  %.4f\n', adj_factorROSS)

adj_factorROSS = RuntimeFactorROSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
fprintf('Adjustment factor ROSS to SMC:  %.4f\n', adj_factorROSS)


% =========================================================================
% Functions (vertorized form)
% =========================================================================

% Standard Monte Carlo
function P = payoffs_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2)
    
    N = size(U1,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu1 = r-d1;  % risk-neutral drift parameter for S1
    mu2 = r-d2;  % risk-neutral drift parameter for S2

    % initialize variables (Nx1)
    S1 = S10 .* ones(N,1);  % simulations for S1
    S2 = S20 .* ones(N,1);  % simulations for S2
    alive = true(N,1);  % keep track of paths alive
    P = zeros(N,1);  % payoffs

    % sample from standard normal distribution via inverse sampling
    Z1 = norminv(U1);  % (Nxm)
    Z2 = norminv(U2);  % (Nxm)
    
    % simulate correlation via GHK importance sampling
    Y1 = Z1;
    Y2 = cor.*Z1 + sqrt(1-cor^2).*Z2;

    for j=1:m
        % compute drift and volatility
        drift1 = (mu1 - 0.5*sigma1^2)*dt(j);
        vol1 = sigma1 * sqrt(dt(j));
        drift2 = (mu2 - 0.5*sigma2^2)*dt(j);
        vol2 = sigma2 * sqrt(dt(j));

        Y1j = Y1(:,j);  % N random numbers for S1 simulations at j-th observation date (Nx1)
        Y2j = Y2(:,j);  % N random numbers for S2 simulations at j-th observation date (Nx1)
        
        % update paths and their states
        S1(alive) = S1(alive) .* exp(drift1 + vol1 .* Y1j(alive));
        S2(alive) = S2(alive) .* exp(drift2 + vol2 .* Y2j(alive));
        
        autocalled = alive & (min(S1./S1ref, S2./S2ref) >= B);
        alive(autocalled) = false;
        
        % compute (discounted) early payoffs
        P(autocalled) = exp(-r*(t(j+1)-t(1))) * Q(j);
    end
    
    % compute (discounted) final payoffs
    redem_payoff(alive) = 100 .* min( S1(alive) ./ S1ref, S2(alive) ./ S2ref );  % redemption payoff
    P(alive) = exp(-r*(t(m+1)-t(1))) .* redem_payoff(alive); % (Nx1)    
end

% One-Step Survival Monte Carlo
function P = payoffs_bivAutocallable_OSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2)
    % One-step survival Monte Carlo for bivariate autocallable
        
    %epsu = 1e-12;  % numerical safeguard
    N = size(U1,1);
    m = length(t)-1;
    dt = diff(t);
    mu1 = r-d1;
    mu2 = r-d2;
    
    % initialize variables (Nx1)
    S1 = S10 .* ones(N,1);
    S2 = S20 .* ones(N,1);
    W  = ones(N,1);
    P  = zeros(N,1);

    % precompute normals for Z1 
    Z1 = norminv(U1);  % Nxm
    
    % initialize uncorrelated Z2
    Z2 = zeros(N,m);

    for j = 1:m
        % compute drifts and volatilities
        drift1 = (mu1 - 0.5*sigma1^2) * dt(j);
        vol1   = sigma1 * sqrt(dt(j));
        drift2 = (mu2 - 0.5*sigma2^2) * dt(j);
        vol2   = sigma2 * sqrt(dt(j));
        
        Z1j = Z1(:,j);  % N random numbers for simulations at j-th observation date (Nx1)
        
        % compute barriers for possible truncation
        b1 = ( log( (B*S1ref) ./ S1 ) - drift1 ) ./ vol1;
        b2 = ( log( (B*S2ref) ./ S2 ) - drift2 ) ./ vol2;
        
        % Y1 < b1 => survival guaranteed => no truncation needed
        survZ1 = (Z1j < b1);  % logical Nx1
        
        % compute p = P(Z2 <= threshold | Z1 = z1) where threshold for Z2 is b2
        p = ones(N,1);  % default 1 when Z1 already below b1
        
        % conditional survival when Z1 >= b1
        if any(~survZ1)
            b2trunc = (b2(~survZ1) - cor.*Z1j(~survZ1)) ./ sqrt(1-cor^2);  % truncation upper limit
            p(~survZ1) = normcdf(b2trunc);
            p(~survZ1) = min(p(~survZ1),1-1e-15);  % clamp probability to avoid underflow
        end

        % update payoff and likelihood weights
        P = P + W .* (1 - p) * Q(j) * exp(-r * (t(j+1)-t(1)));
        W = W .* p;
        
        % simulate uncorrelated Z2 (some values truncated, some not)
        U2j = U2(:,j);  % N random numbers for simulations at j-th observation date (Nx1)
        
        Z2j = Z2(:,j);  % Nx1
        Z2j(survZ1) = norminv(U2j(survZ1));  % unconditional branch
        Z2j(~survZ1) = norminv(p(~survZ1) .* U2j(~survZ1));  % conditional branch

        % simulate correlation via GHK importance sampling
        Y1j = Z1j;
        Y2j = cor.*Z1j + sqrt(1-cor^2).*Z2j;
        
        % update asset prices
        S1 = S1 .* exp(drift1 + vol1.*Y1j);
        S2 = S2 .* exp(drift2 + vol2.*Y2j);
    end
    
    % update (discounted) weighted payoff
    redem_payoff = 100 .* min( S1 ./ S1ref, S2 ./ S2ref );  % redemption payoff
    P = P + W .* redem_payoff * exp(-r * (t(m+1)-t(1)));  % update final (discounted) weighted payoff
end

% One-Step Survival Monte Carlo with rotation
function P = payoffs_bivAutocallable_ROSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2)
    % One-step survival Monte Carlo with rotation for bivariate autocallable 

    %epsu = 1e-12;  % numerical safeguard
    N = size(U1,1);
    m = length(t)-1;
    dt = diff(t);
    mu1 = r-d1;
    mu2 = r-d2;
    alpha = 0.5 * (0.5*pi - atan(-cor / sqrt(1-cor^2)));
    
    % initialize paths, weights and payoffs
    S1 = S10 .* ones(N,1);
    S2 = S20 .* ones(N,1);
    W  = ones(N,1);
    P  = zeros(N,1);
    
    % precompute normals for X1
    X1 = norminv(U1);  % Nxm
    
    for j = 1:m
        
        % compute drifts and volatilities
        drift1 = (mu1 - 0.5*sigma1^2) * dt(j);
        vol1   = sigma1 * sqrt(dt(j));
        drift2 = (mu2 - 0.5*sigma2^2) * dt(j);
        vol2   = sigma2 * sqrt(dt(j));

        % compute barriers for possible truncation
        b1 = ( log((B*S1ref) ./ S1) - drift1 ) ./ vol1;
        b2 = ( log((B*S2ref) ./ S2) - drift2 ) ./ vol2;
        
        % compute survival probability
        X1j = X1(:,j);  % Nx1
        num1 = (b1 - X1j.*cos(alpha)) ./ sin(alpha);
        num2 = (b2 - X1j .* (cor*cos(alpha) - sqrt(1-cor^2)*sin(alpha))) ...
                ./ (cor*sin(alpha) + sqrt(1-cor^2)*cos(alpha));
        b = max(num1,num2);
        p = normcdf(b);
        p = min(p,1-1e-15);  % clamp probability to avoid underflow

        % update payoff and likelihood weights
        P = P + W .* (1 - p) .* Q(j) * exp(-r*(t(j+1)-t(1)));
        W = W .* p;
    
        % sample X2 conditional on X1 from truncated distribution
        U2j = U2(:,j);  % Nx1
        X2j = norminv(p .* U2j);

        % rotation step
        Z1j = X1j.*cos(alpha) + X2j.*sin(alpha);
        Z2j = -X1j.*sin(alpha) + X2j.*cos(alpha);

        % simulate correlation via GHK importance sampling
        Y1j = Z1j;
        Y2j = cor.*Y1j + sqrt(1-cor^2).*Z2j;

        % update asset prices
        S1 = S1 .* exp(drift1 + vol1 .* Y1j);
        S2 = S2 .* exp(drift2 + vol2 .* Y2j);
    end
    
    % update (discounted) weighted payoff
    redem_payoff = 100 .* min( S1 ./ S1ref, S2 ./ S2ref );  % redemption payoff
    P = P + W .* redem_payoff .* exp(-r*(t(m+1)-t(1)));
end

% Compute runtime factor for OSS wrt SMC
function adj_factor = RuntimeFactorOSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2)

    % warm up the functions (optional but recommended)
    payoffs_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
    payoffs_bivAutocallable_OSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);

    % define anonymous wrappers for timeit
    f_SMC  = @() payoffs_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
    f_OSS  = @() payoffs_bivAutocallable_OSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);

    % compute average runtime per path using timeit
    N = size(U1,1);  % number of simulations
    time_per_pathSMC = timeit(f_SMC) / N;
    time_per_pathOSS = timeit(f_OSS) / N;

    % compute adjustment factor
    adj_factor = time_per_pathSMC / time_per_pathOSS;
end

% Compute runtime factor for ROSS wrt SMC
function adj_factor = RuntimeFactorROSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2)

    % warm up the functions (optional but recommended)
    payoffs_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
    payoffs_bivAutocallable_ROSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);

    % define anonymous wrappers for timeit
    f_SMC  = @() payoffs_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
    f_ROSS  = @() payoffs_bivAutocallable_ROSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);

    % compute average runtime per path using timeit
    N = size(U1,1);  % number of simulations
    time_per_pathSMC = timeit(f_SMC) / N;
    time_per_pathROSS = timeit(f_ROSS) / N;

    % compute adjustment factor
    adj_factor = time_per_pathSMC / time_per_pathROSS;
end