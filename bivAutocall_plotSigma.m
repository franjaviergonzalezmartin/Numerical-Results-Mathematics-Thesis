% Script to plot price and Delta of bivariate autocallables
% (min-dependence) option against initial prices S10 and S20 using three methods:
% 1) Standard Monte Carlo (SMC)
% 2) One-Step Survival Monte Carlo (OSS)
% 3) One-Step Survival Monte Carlo with rotation step (ROSS)

% =========================================================================
% Parameters
% =========================================================================
clear
seed            = 12345;                                 % fixed random seed
sigma1          = linspace(1e-5,0.5,60);                 % volatility of S1
sigma2          = linspace(1e-5,0.5,60);                 % volatility of S2
S10             = 4000;                                  % initial asset price S1
S20             = 7000;                                  % initial asset price S2
k1              = numel(S10);                            % number of S10 points
k2              = numel(S20);                            % number of S20 points
S1ref           = 4000;                                  % reference price for S1
S2ref           = 8000;                                  % reference price for S2
B               = 1;                                     % barrier level
t               = [0,1,2,3,4,5];                         % observation dates
m               = length(t)-1;                           % number of observation dates
Q               = [110, 120, 130, 140, 150];             % early coupons
r               = 0.04;                                  % risk-free rate
d1              = 0;                                     % dividend yield of S1
d2              = 0;                                     % dividend yield of S2
cor             = 0.5;                                   % correlation in Brownian motion
N_SMC           = 1e2;                                   % number of simulations for SMC
adj_factorOSS   = 0.74;                                  % adjustment factor for OSS (precomputed in different script)
N_OSS           = round(adj_factorOSS * N_SMC);          % number of simulations for OSS (runtime-adjusted)
adj_factorROSS  = 1.36;                                  % adjustment factor for ROSS (precomputed in different script)
N_ROSS          = round(adj_factorROSS * N_SMC);         % number of simulations for ROSS (runtime-adjusted)
rng(seed);                                               % ensure same seed for fair comparison 
U1              = rand(max([N_SMC, N_OSS, N_ROSS]), m);  % master matrix with random numbers for S1
U1_SMC          = U1(1:N_SMC, :);                        % common random numbers for SMC
U1_OSS          = U1(1:N_OSS, :);                        % common random numbers for OSS
U1_ROSS         = U1(1:N_ROSS, :);                       % common random numbers for ROSS
U2              = rand(max([N_SMC, N_OSS, N_ROSS]), m);  % master matrix with random numbers for S2
U2_SMC          = U2(1:N_SMC, :);                        % common random numbers for SMC
U2_OSS          = U2(1:N_OSS, :);                        % common random numbers for OSS
U2_ROSS         = U2(1:N_ROSS, :);                       % common random numbers for ROSS

% Build 2D grid of initial prices
[sigma1g, sigma2g] = meshgrid(sigma1, sigma2);

% Vectorize grid
sigma1_vec = sigma1g(:);
sigma2_vec = sigma2g(:);

h_vec = 5e-3 * sigma1_vec(:)';                           % step size in finite differences for Vega

% Print parameters
fprintf('======================================================================================= \n');
fprintf('Parameters used:\n');
fprintf('  Fixed random seed                                  = %d\n', seed);
fprintf('  Volatility sigma1                                  = %.4f to %.4f (step %.6f)\n', sigma1(1), sigma1(end), sigma1(2)-sigma1(1));
fprintf('  Volatility sigma2                                  = %.4f to %.4f (step %.6f)\n', sigma2(1), sigma2(end), sigma2(2)-sigma2(1));
fprintf('  Number of sigma1 points                            = %d\n', k1);
fprintf('  Number of sigma2 points                            = %d\n', k2);
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
fprintf('  Asset correlation                                  = %.4f\n', cor);
fprintf('  Number of simulations for SMC N_SMC                = %d\n', N_SMC);
fprintf('  Random numbers for SMC U_SMC                       ~ U(0,1)\n');
fprintf('  Runtime adjustment-factor for OSS adj_factorOSS    = %.2f\n', adj_factorOSS)
fprintf('  Number of simulations for OSS N_OSS                = %d\n', N_OSS);
fprintf('  Random numbers for ROSS U_ROSS                     ~ U(0,1)\n');
fprintf('  Runtime adjustment-factor for ROSS adj_factorROSS  = %.2f\n', adj_factorROSS)
fprintf('  Number of simulations for ROSS N_ROSS              = %d\n', N_ROSS);
fprintf('  Random numbers for ROSS U_ROSS                     ~ U(0,1)\n');
fprintf('  Step size h range = [%.4f, %.4f]\n\n', min(h_vec), max(h_vec));


V_SMC  = bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1_SMC, U2_SMC);
dV_SMC = Vega_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1_SMC, U2_SMC, h_vec);

V_OSS  = bivAutocallable_OSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1_OSS, U2_OSS);
dV_OSS = Vega_bivAutocallable_OSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1_OSS, U2_OSS, h_vec);


V_ROSS = bivAutocallable_ROSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1_ROSS, U2_ROSS);
dV_ROSS = Vega_bivAutocallable_ROSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1_ROSS, U2_ROSS, h_vec);

% Reshape back to surfaces
V_SMC  = reshape(V_SMC,  size(sigma1g));
dV_SMC = reshape(dV_SMC, size(sigma1g));
V_OSS  = reshape(V_OSS,  size(sigma1g));
dV_OSS = reshape(dV_OSS, size(sigma1g));
V_ROSS = reshape(V_ROSS, size(sigma1g));
dV_ROSS = reshape(dV_ROSS, size(sigma1g));


% =========================================================================
% Plot 1: Option price with SMC
% =========================================================================
figure(1);
surf(sigma1g, sigma2g, V_SMC, 'EdgeColor', 'k', 'FaceColor', 'flat');
axis tight;
colormap turbo;
colorbar;
xlabel('sigma_{1}');
ylabel('sigma_{2}');
zlabel('Option value');
title('Bivariate autocallable price (SMC)');
view(1.139824218750000e+02,20.0859375);
zlim([59 123]);

% =========================================================================
% Plot 2: Option price with OSS
% =========================================================================
figure(2);
surf(sigma1g, sigma2g, V_OSS, 'EdgeColor', 'k', 'FaceColor', 'flat');
axis tight;
colormap turbo;
colorbar;
xlabel('sigma_{1}');
ylabel('sigma_{2}');
zlabel('Option value');
title('Bivariate autocallable price (OSS)');
view(1.139824218750000e+02,20.0859375);
zlim([59 123]);

% =========================================================================
% Plot 3: Option price with ROSS
% =========================================================================
figure(3);
surf(sigma1g, sigma2g, V_ROSS, 'EdgeColor', 'k', 'FaceColor', 'flat');
axis tight;
colormap turbo;
colorbar;
xlabel('sigma_{1}');
ylabel('sigma_{2}');
zlabel('Option value');
title('Bivariate autocallable price (ROSS)');
view(1.139824218750000e+02,20.0859375);
zlim([59 123]);

% =========================================================================
% Plot 4: Option delta with SMC
% =========================================================================
figure(4);
surf(sigma1g, sigma2g, dV_SMC, 'EdgeColor', 'k', 'FaceColor', 'flat');
axis tight;
colormap turbo;
colorbar;
xlabel('sigma_{1}');
ylabel('sigma_{2}');
zlabel('Option delta');
title('Bivariate autocallable price (SMC)');
view(32.167968750000014,14.515625);
zlim([-470 50]);

% =========================================================================
% Plot 5: Option delta with OSS
% =========================================================================
figure(5);
surf(sigma1g, sigma2g, dV_OSS, 'EdgeColor', 'k', 'FaceColor', 'flat');
axis tight;
colormap turbo;
colorbar;
xlabel('sigma_{1}');
ylabel('sigma_{2}');
zlabel('Option delta');
title('Bivariate autocallable delta (OSS)');
view(32.167968750000014,14.515625);
zlim([-470 50]);

% =========================================================================
% Plot 6: Option delta with ROSS
% =========================================================================
figure(6);
surf(sigma1g, sigma2g, dV_ROSS, 'EdgeColor', 'k', 'FaceColor', 'flat');
axis tight;
colormap turbo;
colorbar;
xlabel('sigma_{1}');
ylabel('sigma_{2}');
zlabel('Option delta');
title('Bivariate autocallable delta (ROSS)');
view(32.167968750000014,14.515625);
zlim([-470 50]);


% =========================================================================
% Functions (vertorized form)
% -> Tailored for plotting against vectors sigma1 and sigma2
% =========================================================================

function V = bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1, U2)

    sigma1_vec = sigma1_vec(:)';  % ensure sigma1 is a row vector
    sigma2_vec = sigma2_vec(:)';  % ensure sigma2 is a row vector
    k = numel(sigma1_vec);  % number of values for sigma1 and sigma2 (same)
    N = size(U1,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu1 = r-d1;  
    mu2 = r-d2;

    % initialize variables (Nxk)
    S1 = repmat(S10,N,k);  % paths for S1
    S2 = repmat(S20,N,k);  % paths for S2
    alive = true(N,k);  % keep track of paths alive
    payoff = zeros(N,k);  % payoffs
    
    % sample from standard normal distribution via inverse sampling
    Z1 = norminv(U1);  % Nxm
    Z2 = norminv(U2);  % Nxm

    % simulate correlation via GHK importance sampling
    Y1 = Z1;
    Y2 = cor.*Z1 + sqrt(1-cor^2).*Z2;

    for j=1:m
        % compute drift and volatility
        drift1 = (mu1 - 0.5.*sigma1_vec.^2).*dt(j);
        vol1 = sigma1_vec .* sqrt(dt(j));
        drift2 = (mu2 - 0.5.*sigma2_vec.^2).*dt(j);
        vol2 = sigma2_vec .* sqrt(dt(j));

        Y1j = Y1(:,j);  % N random numbers for S1 at j-th observation date (Nx1)
        Y1j_mat = repmat(Y1j,1,k);  % same N random numbers for each initial price S10 (Nxk)
        Y2j = Y2(:,j);  % N random numbers for S2 at j-th observation date (Nx1)
        Y2j_mat = repmat(Y2j,1,k);  % same N random numbers for each initial price S20 (Nxk)

        % update paths and their states
        growth1 = exp(drift1 + vol1 .* Y1j_mat);   % N×k
        S1 = S1 .* (alive .* growth1 + ~alive);

        growth2 = exp(drift2 + vol2 .* Y2j_mat);
        S2 = S2 .* (alive .* growth2 + ~alive);

        autocalled = alive & (min(S1./S1ref, S2./S2ref) >= B);
        alive(autocalled) = false;

        % compute (discounted) early payoffs
        payoff(autocalled) = exp(-r*(t(j+1)-t(1))) * Q(j);
    end

    % compute (discounted) final payoffs and option value
    redem_payoff(alive) = 100 .* min( S1(alive) ./ S1ref, S2(alive) ./ S2ref );  % redemption payoff
    payoff(alive) = exp(-r*(t(m+1)-t(1))) .* redem_payoff(alive); % (Nxk)
    V = mean(payoff,1);  % mean over simulations - one for each initial S10 and S20 (1xk)
end

function dV = Vega_bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1, U2, h_vec)
    % Forward finite differences OSS Monte Carlo with CRN
    sigma1_vec = sigma1_vec(:)';
    h_vec   = h_vec(:)';
    
    V = bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1, U2);
    Vh = bivAutocallable_SMC(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec+h_vec, sigma2_vec, cor, U1, U2);
    dV = (Vh - V)./h_vec;
end


function V = bivAutocallable_OSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1, U2)
    
    sigma1_vec = sigma1_vec(:)';  % ensure S10 is a row vector
    sigma2_vec = sigma2_vec(:)';  % ensure S20 is a row vector
    k = numel(sigma1_vec);  % number of initial S10 and S20 points (same)
    N = size(U1,1);  % number of simulations
    m = length(t)-1;  % t = (t0,t1,...,tm) = (t(1),t(2),...,t(m+1))
    dt = diff(t);  % (t(2)-t(1),...,t(m+1)-t(m))
    mu1 = r-d1;  % risk-neutral drift in S1
    mu2 = r-d2;  % risk-neutral drift in S2
    
    % initilize variables (Nxk)
    S1 = repmat(S10,N,k);  % paths for S1
    S2 = repmat(S20,N,k);  % paths for S2
    W = ones(N,k);  % likelihood weights
    payoff = zeros(N,k);  % payoffs

    % precompute normals for Z1 
    Z1 = norminv(U1);  % Nxm

    for j=1:m

        % compute drifts and volatilities
        drift1 = (mu1 - 0.5.*sigma1_vec.^2) .* dt(j);
        vol1   = sigma1_vec .* sqrt(dt(j));
        drift2 = (mu2 - 0.5.*sigma2_vec.^2) .* dt(j);
        vol2   = sigma2_vec .* sqrt(dt(j));
        
        % draw Y1 first (precomputed standard normal distribution)
        Z1j = Z1(:,j);  % Nx1
        Z1j_mat = repmat(Z1j,1,k);  % same N random numbers for each initial price S10 (Nxk)
        
        % compute barriers
        b1 = ( log((B*S1ref) ./ S1) - drift1 ) ./ vol1;
        b2 = ( log((B*S2ref) ./ S2) - drift2 ) ./ vol2;
        
        % survival indicator via Z1
        survZ1 = (Z1j_mat < b1);
        
        % initialize survival probability
        p = ones(N,k);
        
        % conditional survival when Z1 >= b1
        if any(~survZ1(:))
            b2trunc = (b2(~survZ1) - cor.*Z1j_mat(~survZ1)) ./ sqrt(1-cor^2);  % truncation upper limit
            p(~survZ1) = normcdf(b2trunc);
            p(~survZ1) = min(p(~survZ1),1-1e-15);  % clamp probability to avoid underflow
        end
        
        % update payoffs and likelihood weights
        payoff = payoff + W .* (1-p) .* Q(j) .* exp(-r*(t(j+1)-t(1)));
        W = W .* p;
                
        % sample Z2 conditioned on Z1
        U2j = U2(:,j);  % Nx1
        U2j_mat = repmat(U2j,1,k);  % same N random numbers for each initial price (Nxk)
        
        Z2j_mat = zeros(N,k);  % initialize uncorrelated Z2
        Z2j_mat(survZ1) = norminv(U2j_mat(survZ1));  % unconditional branch
        Z2j_mat(~survZ1) = norminv(p(~survZ1) .* U2j_mat(~survZ1));  % truncated branch
        
        % simulate correlation via GHK importance sampling
        Y1j_mat = Z1j_mat;
        Y2j_mat = cor.*Z1j_mat + sqrt(1-cor^2).*Z2j_mat;
        
        % update asset prices
        S1 = S1 .* exp(drift1 + vol1 .* Y1j_mat);
        S2 = S2 .* exp(drift2 + vol2 .* Y2j_mat);
    end
    
    % update final (discounted) weighted payoff
    redem_payoff = 100 .* min( S1 ./ S1ref, S2 ./ S2ref );  % redemption payoff
    payoff = payoff + W .* redem_payoff * exp(-r * (t(m+1)-t(1)));

    V = mean(payoff,1);  % mean over simulations - one for each initial price (1xk)
end

function dV = Vega_bivAutocallable_OSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1, U2, h_vec)
    % Forward finite differences OSS Monte Carlo with CRN
    sigma1_vec = sigma1_vec(:)';
    h_vec   = h_vec(:)';
    
    V = bivAutocallable_OSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1, U2);
    Vh = bivAutocallable_OSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec+h_vec, sigma2_vec, cor, U1, U2);
    dV = (Vh - V)./h_vec;
end


function V = bivAutocallable_ROSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1, U2)

    %epsu = 1e-12;  % numerical safeguard
    sigma1_vec = sigma1_vec(:)';  % ensure S10 is a row vector
    sigma2_vec = sigma2_vec(:)';  % ensure S20 is a row vector
    k = numel(sigma1_vec);  % number of initial S10 and S20 points (same)
    N = size(U1,1);
    m = length(t)-1;
    dt = diff(t);
    mu1 = r-d1;
    mu2 = r-d2;
    alpha = 0.5 * (0.5*pi - atan(-cor / sqrt(1-cor^2)));
    
    % initialize paths, weights and payoffs
    S1 = repmat(S10,N,k);  % paths for S1
    S2 = repmat(S20,N,k);  % paths for S2
    W = ones(N,k);  % likelihood weights
    payoff = zeros(N,k);  % payoffs
    
    % precompute normals for X1
    X1 = norminv(U1);  % Nxm

    for j = 1:m
        
        % compute drifts and volatilities
        drift1 = (mu1 - 0.5.*sigma1_vec.^2) .* dt(j);
        vol1   = sigma1_vec .* sqrt(dt(j));
        drift2 = (mu2 - 0.5.*sigma2_vec.^2) .* dt(j);
        vol2   = sigma2_vec .* sqrt(dt(j));

        % compute barriers for possible truncation
        b1 = ( log((B*S1ref) ./ S1) - drift1 ) ./ vol1;
        b2 = ( log((B*S2ref) ./ S2) - drift2 ) ./ vol2;

        % compute survival probability
        X1j = X1(:,j);  % Nx1
        X1j_mat = repmat(X1j,1,k);  % Nxk
        num1 = (b1 - X1j_mat.*cos(alpha)) ./ sin(alpha);
        num2 = (b2 - X1j_mat .* (cor*cos(alpha) - sqrt(1-cor^2)*sin(alpha))) ...
                ./ (cor*sin(alpha) + sqrt(1-cor^2)*cos(alpha));
        b = max(num1,num2);
        p = normcdf(b);
        p = min(p,1-1e-15);  % clamp probability to avoid underflow
        

        % update payoff and likelihood weights
        payoff = payoff + W .* (1 - p) .* Q(j) * exp(-r*(t(j+1)-t(1)));
        W = W .* p;
    
        % sample X2 conditional on X1 from truncated distribution
        U2j = U2(:,j);  % Nx1
        U2j_mat = repmat(U2j,1,k);  % Nxk
        X2j_mat = norminv(p .* U2j_mat);  % Nxk

        % rotation step
        Z1j_mat = X1j_mat.*cos(alpha) + X2j_mat.*sin(alpha);
        Z2j_mat = -X1j_mat.*sin(alpha) + X2j_mat.*cos(alpha);

        % simulate correlation via GHK importance sampling
        Y1j_mat = Z1j_mat;
        Y2j_mat = cor.*Y1j_mat + sqrt(1-cor^2).*Z2j_mat;

        % update asset prices
        S1 = S1 .* exp(drift1 + vol1 .* Y1j_mat);
        S2 = S2 .* exp(drift2 + vol2 .* Y2j_mat);
    end
    
    % update (discounted) weighted payoff
    redem_payoff = 100 .* min( S1 ./ S1ref, S2 ./ S2ref );  % redemption payoff
    payoff = payoff + W .* redem_payoff .* exp(-r*(t(m+1)-t(1)));
    % compute option value
    V = mean(payoff,1);  % mean over simulations - one for each initial S0 (1xk)
end

function dV = Vega_bivAutocallable_ROSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1, U2, h_vec)
    % Forward finite differences OSS Monte Carlo with CRN
    sigma1_vec = sigma1_vec(:)';
    h_vec   = h_vec(:)';
    
    V = bivAutocallable_ROSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec, sigma2_vec, cor, U1, U2);
    Vh = bivAutocallable_ROSS(S10, S20, S1ref, S2ref, B, Q, t, r, d1, d2, sigma1_vec+h_vec, sigma2_vec, cor, U1, U2);
    dV = (Vh - V)./h_vec;
end


