% Script to simulate and plot paths in an up-and-out call option via GBM

% =========================================================================
% Parameters
% =========================================================================
clear
seed   = 12345;
rng(seed);                  % fixed random seed
S0     = 50;                % initial asset price
B      = 60;                % barrier level
dt     = 1/365;             % daily discretization
k      = floor(1/dt);       % number of steps = 365
t      = linspace(0,1,51);  % weekly observation dates
r      = 0.1;               % risk-free interest rate
d      = 0;                 % divident yield
sigma  = 0.2;               % volatility
N      = 25;                % number of simulated paths
U      = rand(N,k);         % uniform numbers for inverse transform


% =========================================================================
% Plot
% =========================================================================

S = simulate_paths(S0, B, dt, t, r, d, sigma, U);

figure;
hold on;
% colors for all paths
cmap = lines(N);
% observation step indices
obs_idx = floor(t/dt) + 1;
for i = 1:N
    path = S(i,:);
    plot(path,'Color',cmap(i,:),'LineWidth',1.2);

    % find knockout time
    ko = find(path(obs_idx) >= B, 1);
    if ~isempty(ko)
        j = obs_idx(ko);
        plot(j,path(j),'o','MarkerSize',6,'MarkerFaceColor',cmap(i,:),'MarkerEdgeColor','k');
    end
end
% barrier line
yline(B,'--k','Barrier','FontWeight','bold','LabelHorizontalAlignment','right','LabelVerticalAlignment','bottom');
xlabel('Step index');
ylabel('Asset price');
title('GBM Paths with Weekly Up-and-Out Barrier');
grid on;
ax = gca;
ax.GridColor = [0.8 0.8 0.8];  
ax.GridAlpha = 0.3;            
hold off;


% =========================================================================
% Path simulation function
% =========================================================================

function S = simulate_paths(S0, B, dt, t, r, d, sigma, U)

    N = size(U,1);
    k = size(U,2);
    mu = r-d;

    drift = (mu - 0.5*sigma^2) * dt;
    vol   = sigma * sqrt(dt);

    % map observation times to discrete step indices
    obs_idx = floor(t/dt) + 1;   % +1 to index the column of S

    % initialize storage
    S = NaN(N, k+1);
    S(:,1) = S0;

    alive = true(N,1);

    Z = norminv(U);

    % simulate step-by-step
    for j = 2:(k+1)
        idx = j-1;      % Z column index

        % update alive paths
        S(alive,j) = S(alive,j-1) .* exp( drift + vol .* Z(alive,idx) );

        % check barrier at observation dates
        if ismember(j, obs_idx)
            autocalled = alive & (S(:,j) >= B);
            alive(autocalled) = false;     % kill them
        end
    end
end

