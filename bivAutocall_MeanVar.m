% Script to compute mean and variance of option price, Delta and Vega
% of bivariate autocallale (min-dependence) of three Monte Carlo methods:
% 1) Standard Monte Carlo (SMC)
% 2) One-Step Survival Monte Carlo (OSS)
% 3) One-Step Survival Monte Carlo with rotation step (ROSS)

% =========================================================================
% Parameters
% =========================================================================
clear
seed            = 12345;                                 % fixed random seed
S10             = 3500;                                  % initial asset price S1
S20             = 7000;                                  % initial asset price S2
S1ref           = 4000;                                  % reference price for S1
S2ref           = 8000;                                  % reference price for S2
B               = 1;                                     % barrier level
t               = [0,1,2,3,4,5];                         % observation dates
m               = length(t)-1;                           % number of observation dates
Q               = [110, 120, 130, 140, 150];             % early coupons
r               = 0.04;                                  % risk-free rate
d1              = 0;                                     % dividend yield of S1
d2              = 0;                                     % dividend yield of S2
sigma1          = 0.3;                                   % volatility of S1
sigma2          = 0.4;                                   % volatility of S1
cor             = 0.5;                                   % correlation in Brownian motion
N_SMC           = 1e2;                                   % number of simulations for SMC
adj_factorOSS   = 0.74;                                  % adjustment factor for OSS1 (precomputed in different script)
N_OSS           = round(adj_factorOSS * N_SMC);          % number of simulations for OSS1 (runtime-adjusted)
adj_factorROSS  = 1.36;                                  % adjustment factor for OSS2 (precomputed in different script)
N_ROSS          = round(adj_factorROSS * N_SMC);         % number of simulations for OSS2 (runtime-adjusted)
rng(seed);                                               % ensure same seed for fair comparison 
U1              = rand(max([N_SMC, N_OSS, N_ROSS]), m);  % master matrix with random numbers for S1
U1_SMC          = U1(1:N_SMC, :);                        % random numbers for SMC
U1_OSS          = U1(1:N_OSS, :);                        % random numbers for OSS
U1_ROSS         = U1(1:N_ROSS, :);                       % random numbers for ROSS
U2              = rand(max([N_SMC, N_OSS, N_ROSS]), m);  % master matrix with random numbers for S1
U2_SMC          = U2(1:N_SMC, :);                        % random numbers for SMC
U2_OSS          = U2(1:N_OSS, :);                        % random numbers for OSS
U2_ROSS         = U2(1:N_ROSS, :);                       % random numbers for ROSS
hS10            = 5e-3 * S10;                            % step size in finite difference for Delta
hSigma1         = 5e-3 * sigma1;                         % step size in finite difference for Sigma

% Print parameters
fprintf('======================================================================================= \n');
fprintf('Parameters used:\n');
fprintf('  Fixed random seed                                  = %d\n', seed);
fprintf('  Initial price S10                                  = %.2f\n', S10);
fprintf('  Initial price S20                                  = %.2f\n', S20);
fprintf('  Reference price S1ref                              = %.2f\n', S1ref);
fprintf('  Reference price S2ref                              = %.2f\n', S2ref);
fprintf('  Barrier B                                          = %.2f\n', B);
fprintf('  Observation dates t                                = %s\n', mat2str(t));
fprintf('  Number of observation dates m                      = %d\n', m);
fprintf('  Early coupons Q                                    = %s\n', mat2str(Q));
fprintf('  Redemption payoff                                  = 100 * min(S1/S1ref, S2/S2ref)\n');
fprintf('  Interest rate r                                    = %.4f\n', r);
fprintf('  Dividend yield d1                                  = %.4f\n', d1);
fprintf('  Dividend yield d2                                  = %.4f\n', d2);
fprintf('  Volatility sigma1                                  = %.4f\n', sigma1);
fprintf('  Volatility sigma2                                  = %.4f\n', sigma2);
fprintf('  Asset correlation                                  = %.4f\n', cor);
fprintf('  Number of simulations for SMC N_SMC                = %d\n', N_SMC);
fprintf('  Random numbers for SMC U_SMC                       ~ U(0,1)\n');
fprintf('  Runtime adjustment-factor for OSS adj_factorOSS    = %.2f\n', adj_factorOSS)
fprintf('  Number of simulations for OSS N_OSS                = %d\n', N_OSS);
fprintf('  Random numbers for ROSS U_ROSS                     ~ U(0,1)\n');
fprintf('  Runtime adjustment-factor for ROSS adj_factorROSS  = %.2f\n', adj_factorROSS)
fprintf('  Number of simulations for ROSS N_ROSS              = %d\n', N_ROSS);
fprintf('  Random numbers for ROSS U_ROSS                     ~ U(0,1)\n');
fprintf('  Step size for Delta hS10                           = %.4f\n', hS10');
fprintf('  Step size for Delta hSigma1                        = %.4f\n\n', hSigma1');


% =========================================================================
% 1. Option price
% =========================================================================

P_SMC = payoffs_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1_SMC, U2_SMC);
P_OSS = payoffs_bivAutocallable_OSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1_OSS, U2_OSS);
P_ROSS = payoffs_bivAutocallable_ROSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1_ROSS, U2_ROSS);
fprintf('Bivariate Autocallable Price: Variance\n');
fprintf('  Variance SMC  = %.4f\n', var(P_SMC,0));
fprintf('  Variance OSS  = %.4f\n', var(P_OSS,0));
fprintf('  Variance ROSS  = %.4f\n', var(P_ROSS,0));

% =========================================================================
% 2. Delta
% =========================================================================

P_deltaSMC = payoffs_Delta_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1_SMC, U2_SMC, hS10);
P_deltaOSS = payoffs_Delta_bivAutocallable_OSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1_OSS, U2_OSS, hS10);
P_deltaROSS = payoffs_Delta_bivAutocallable_ROSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1_ROSS, U2_ROSS, hS10);
fprintf('Bivariate Autocallable Delta: Variance\n');
fprintf('  Variance SMC  = %.8f\n', var(P_deltaSMC,0));
fprintf('  Variance OSS  = %.10f\n', var(P_deltaOSS,0));
fprintf('  Variance ROSS  = %.10f\n', var(P_deltaROSS,0));

% =========================================================================
% 3. Vega
% =========================================================================

P_vegaSMC = payoffs_Vega_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1_SMC, U2_SMC, hSigma1);
P_vegaOSS = payoffs_Vega_bivAutocallable_OSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1_OSS, U2_OSS, hSigma1);
P_vegaROSS = payoffs_Vega_bivAutocallable_ROSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1_ROSS, U2_ROSS, hSigma1);
fprintf('Bivariate Autocallable Vega: Variance\n');
fprintf('  Variance SMC  = %.4f\n', var(P_vegaSMC,0));
fprintf('  Variance OSS  = %.4f\n', var(P_vegaOSS,0));
fprintf('  Variance ROSS  = %.4f\n', var(P_vegaROSS,0));

% =========================================================================
% Functions (vertorized form - optimized in MATLAB)
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

function deltaP = payoffs_Delta_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
    Ph = payoffs_bivAutocallable_SMC(S10+h, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
    deltaP = (Ph - P)./h;
end

function vegaP = payoffs_Vega_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
    Ph = payoffs_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1+h, sigma2, cor, U1, U2);
    vegaP = (Ph - P)./h;
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

function deltaP = payoffs_Delta_bivAutocallable_OSS(S1, S2, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_bivAutocallable_OSS(S1, S2, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
    Ph = payoffs_bivAutocallable_OSS(S1+h, S2, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
    deltaP = (Ph - P)./h;
end

function vegaP = payoffs_Vega_bivAutocallable_OSS(S1, S2, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_bivAutocallable_OSS(S1, S2, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
    Ph = payoffs_bivAutocallable_OSS(S1, S2, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1+h, sigma2, cor, U1, U2);
    vegaP = (Ph - P)./h;
end


% One-Step Survival Monte Carlo with rotation step
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

        % update payoff and likelihood weights
        P = P + W .* (1 - p) .* Q(j) * exp(-r*(t(j+1)-t(1)));
        W = W .* p;
        p = min(p,1-1e-15);  % clamp probability to avoid underflow
    
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

function deltaP = payoffs_Delta_bivAutocallable_ROSS(S1, S2, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_bivAutocallable_ROSS(S1, S2, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
    Ph = payoffs_bivAutocallable_ROSS(S1+h, S2, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
    deltaP = (Ph - P)./h;
end

function vegaP = payoffs_Vega_bivAutocallable_ROSS(S1, S2, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2, h)
    
    % Forward finite differences Monte Carlo with CRN
    P = payoffs_bivAutocallable_ROSS(S1, S2, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1, sigma2, cor, U1, U2);
    Ph = payoffs_bivAutocallable_ROSS(S1, S2, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1+h, sigma2, cor, U1, U2);
    vegaP = (Ph - P)./h;
end

