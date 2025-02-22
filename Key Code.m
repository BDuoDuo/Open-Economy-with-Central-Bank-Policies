% Solve Small Open Economy Model
clear;
clc;
close all;

% Part 4 Benchmark
%=========================================================================
% 4.1. Set Parameters
%=========================================================================
% Fixed parameters
beta = 0.97;     % Discount factor
eta = 1.5;       % Curvature of utility from leisure
alpha_H = 0.5;   % No Home bias
kappa = 1;       % Borrowing limit
tau = 0.25;      % Labor tax rate
theta_star = 3;  % Foreign demand elasticity
L_star = 10000;  % Foreign labor supply
Y_star = L_star; % Foreign output equals foreign labor
G_H = 0.2;       % Government Spending
BG = 0.5;        % Bond of government

% Productivity process parameters
rho = 0.967;     % AR(1) persistence
sigma = 0.017;   % Standard deviation of shock
N = 7;           % Number of grid points for productivity

% Create productivity grid using Rouwenhorst method
[Z, P_trans] = rouwenhorst_fixed(N, rho, sigma);
% Asset grid parameters
na = 200;        % Number of asset grid points
a_min = -kappa;  % Minimum assets (borrowing constraint)
a_max = 20;      % Maximum assets
a_grid = linspace(a_min, a_max, na)';

% Numerical parameters
tol = 1e-6;
maxiter = 40;    % Increased iterations
tol_eq = 1e-4;   % Tolerance for equilibrium

%=========================================================================
% 4.2. Find Equilibrium
%=========================================================================
% Initial guesses
R_min = 1.001;  % Minimum interest rate
R_max = 1.05;   % Maximum interest rate
maxiter_eq = 2; % Increased equilibrium iterations

% Bisection on interest rate
for iter_eq = 1:maxiter_eq
R = (R_min + R_max)/2;
% Exchange rate normalized to 1
e = 1;
% Law of one price
PF = e;        % Foreign good price
PH = e;        % Initial guess for home good price (normalized by monetary policy)
% Terms of trade
x = PH/e;
% Foreign demand for home goods
CH_star = alpha_H * x^(-theta_star) * Y_star;

% Initialize value function and policy functions
V = zeros(na, N);   % Value function
gc = zeros(na, N);  % Consumption policy
ga = zeros(na, N);  % Asset policy
gl = zeros(na, N);  % Labor policy
gch = zeros(na, N); % Home good consumption
gcf = zeros(na, N); % Foreign good consumption

% Value Function Iteration
diff = 1;
iter = 0;
while diff > tol && iter < maxiter
iter = iter + 1;
V_old = V;
for ia = 1:na
for iz = 1:N
% Current state
a = a_grid(ia);
s = Z(iz);
% Grid for labor choice
l_grid = linspace(0, 2, 20)';
% Find optimal choices
value = -1e10 * ones(length(l_grid), na);
for il = 1:length(l_grid)
l = l_grid(il);
% Labor income (wage equals PH from firm FOC)
labor_income = (1 - tau) * PH * s * l;
for ia_next = 1:na
a_next = a_grid(ia_next);
if a_next >= -kappa % Check borrowing constraint
% Resources including interest earnings
resources = labor_income + R*PH*a - PH*a_next;
if resources > 0
% Price index
price_index = (PH/alpha_H)^alpha_H * (PF/(1-alpha_H))^(1-alpha_H);
% Optimal allocation between home and foreign goods
ch = alpha_H * (resources/PH);
cf = (1-alpha_H) * (resources/PF);
% Composite consumption
c = (ch^alpha_H) * (cf^(1-alpha_H));
% Current period utility (fixed parentheses)
utility = log(c) - l^(1+eta)/(1+eta);
% Expected continuation value
EV = 0;
for iz_next = 1:N
EV = EV + P_trans(iz,iz_next) * V_old(ia_next,iz_next);
end
value(il,ia_next) = utility + beta*EV;
end
end
end
end


% Find optimal choices
[maxval, idx] = max(value(:));
[il_opt, ia_next_opt] = ind2sub(size(value), idx);
% Store optimal choices
V(ia,iz) = maxval;
gl(ia,iz) = l_grid(il_opt);
ga(ia,iz) = a_grid(ia_next_opt);
% Store consumption choices
labor_income = (1 - tau) * PH * s * gl(ia,iz);
resources = labor_income + R*PH*a - PH*ga(ia,iz);
gch(ia,iz) = alpha_H * (resources/PH);
gcf(ia,iz) = (1-alpha_H) * (resources/PF);
gc(ia,iz) = (gch(ia,iz)^alpha_H) * (gcf(ia,iz)^(1-alpha_H));
end
end
diff = max(abs(V(:) - V_old(:)));
if mod(iter,10) == 0
fprintf('VFI Iteration %d, diff = %f\n', iter, diff);
end
end

% Find stationary distribution
D = ones(na, N)/(na*N);
dist_diff = 1;
while dist_diff > tol
D_new = zeros(na, N);
for ia = 1:na
for iz = 1:N
% Find next period assets
a_next = ga(ia,iz);
% Find grid location and ensure it's within bounds
[~,ia_next] = min(abs(a_grid - a_next));
ia_next = min(max(ia_next, 1), na);
% Linear interpolation weights
if ia_next < na
weight = (a_next - a_grid(ia_next))/(a_grid(ia_next+1) - a_grid(ia_next));
weight = min(max(weight, 0), 1);
else
weight = 0;
ia_next = na - 1;
end
% Distribute mass according to transition probabilities
for iz_next = 1:N
D_new(ia_next,iz_next) = D_new(ia_next,iz_next) + ...
(1-weight) * D(ia,iz) * P_trans(iz,iz_next);
D_new(ia_next+1,iz_next) = D_new(ia_next+1,iz_next) + ...
weight * D(ia,iz) * P_trans(iz,iz_next);
end
end
end

% Normalize distribution
D_new = D_new / sum(D_new(:));
dist_diff = max(abs(D_new(:) - D(:)));
D = D_new;
end
% Calculate aggregates
L = 0; % Aggregate labor
CH = 0; % Aggregate home consumption
CF = 0; % Aggregate foreign consumption
B = 0; % Aggregate private bonds
Y = 0; % Aggregate output
for ia = 1:na
for iz = 1:N
L = L + gl(ia,iz) * Z(iz) * D(ia,iz);
CH = CH + gch(ia,iz) * D(ia,iz);
CF = CF + gcf(ia,iz) * D(ia,iz);
B = B + a_grid(ia) * D(ia,iz);
Y = Y + PH * Z(iz) * gl(ia,iz) * D(ia,iz); % Nominal output
end
end

% Government policy
tax_revenue = tau * Y/PH; % Real tax revenue
BG_next = G_H + BG*R - tax_revenue; % Next period government bonds
% Check market clearing conditions
goods_market = Y/PH - (CH + G_H + CH_star); % Real goods market clearing
bond_market = B + BG; % Total bond market clearing (private + public)
% Update interest rate bounds based on bond market clearing
if bond_market > 0
R_max = R;
else
R_min = R;
end
% Check convergence
if max(abs([goods_market, bond_market])) < tol_eq && abs(BG_next - BG) < tol_eq
break;
end
% Update government bonds
BG = BG_next;
fprintf('Iteration %d: R = %.4f, Bond Market = %.6f, Goods Market = %.6f\n', ...
iter_eq, R, bond_market, goods_market);
end

%=========================================================================
% 4.3. Plot Results
%=========================================================================
figure('Position', [100, 100, 800, 800]);
% Plot consumption policy function
subplot(4,1,1);
plot(a_grid, gc(:,1), 'b-', a_grid, gc(:,end), 'r--', 'LineWidth', 1.5);
title('Consumption Policy Function');
xlabel('Assets');
ylabel('Consumption');
legend('Low Productivity', 'High Productivity');
grid on;
% Plot asset policy function
subplot(4,1,2);
plot(a_grid, ga(:,1), 'b-', a_grid, ga(:,end), 'r--', 'LineWidth', 1.5);
title('Asset Policy Function');
xlabel('Assets');
ylabel('Next Period Assets');
grid on;
% Plot labor policy function
subplot(4,1,3);
plot(a_grid, gl(:,1), 'b-', a_grid, gl(:,end), 'r--', 'LineWidth', 1.5);
title('Labor Supply Policy Function');
xlabel('Assets');
ylabel('Labor Supply');
grid on;
% Plot wealth distribution
subplot(4,1,4);
plot(a_grid, sum(D,2), 'LineWidth', 1.5);
title('Wealth Distribution');
xlabel('Assets');
ylabel('Density');
grid on;
% Print equilibrium results
fprintf('\nEquilibrium Results:\n');
fprintf('Interest Rate = %.4f\n', R);
fprintf('Output = %.4f\n', Y/PH);
fprintf('Home Consumption = %.4f\n', CH);
fprintf('Foreign Consumption = %.4f\n', CF);
fprintf('Government Spending = %.4f\n', G_H);
fprintf('Total Bonds = %.4f\n', B + BG);
% [Rouwenhorst function remains the same]
% Helper function for Rouwenhorst method
function [Z, P] = rouwenhorst_fixed(n, rho, sigma_e)
% Get state space
sigma_y = sigma_e / sqrt(1 - rho^2);
y = linspace(-sigma_y*sqrt(n-1), sigma_y*sqrt(n-1), n)';
Z = exp(y); % Convert to levels
% Special case n = 2
if n == 2
p = (1 + rho) / 2;
P = [p, 1-p; 1-p, p];
return
end
% General case n > 2
p = (1 + rho) / 2;
P = [p 1-p; 1-p p];
for i = 3:n
P1 = zeros(i,i);
% Fill in new transition matrix
P1(1:i-1,1:i-1) = p*P;
P1(1:i-1,2:i) = P1(1:i-1,2:i) + (1-p)*P;
P1(2:i,1:i-1) = P1(2:i,1:i-1) + (1-p)*P;
P1(2:i,2:i) = P1(2:i,2:i) + p*P;
P1 = P1/2;
P = P1;
end
% Ensure rows sum to 1
P = bsxfun(@rdivide, P, sum(P,2));
end
% After running the main model code and obtaining the stationary distribution D
% Calculate net wealth for each state combination
net_wealth = zeros(na, N);
for ia = 1:na
for iz = 1:N
% Net wealth is assets
net_wealth(ia, iz) = a_grid(ia);
end
end
% Flatten the matrices for plotting
wealth_vector = net_wealth(:);
bonds_vector = reshape(a_grid * ones(1,N), [], 1);
density_vector = D(:);
% Sort by wealth for better visualization
[sorted_wealth, sort_idx] = sort(wealth_vector);
sorted_bonds = bonds_vector(sort_idx);
sorted_density = density_vector(sort_idx);
% Create figure
figure('Position', [100, 100, 800, 600]);
% Create the main scatter plot
scatter(sorted_wealth, sorted_bonds, 50, sorted_density, 'filled');
colormap('jet');
colorbar;
% Add labels and title
xlabel('Household Net Wealth', 'FontSize', 12);
ylabel('Bond Holdings', 'FontSize', 12);
title('Distribution of Bond Holdings Across Households', 'FontSize', 14);
% Add grid
grid on;
% Add a horizontal line at zero
hold on;
yline(0, '--k', 'Zero Bond Holdings');
% Add text box with summary statistics
text_x = min(sorted_wealth) + 0.1*(max(sorted_wealth)-min(sorted_wealth));
text_y = max(sorted_bonds) - 0.2*(max(sorted_bonds)-min(sorted_bonds));
% Calculate summary statistics
mean_bonds = sum(sum(D .* reshape(a_grid * ones(1,N), na, N)));
median_idx = find(cumsum(sorted_density) >= 0.5, 1, 'first');
median_bonds = sorted_bonds(median_idx);
positive_bonds = sum(sorted_density(sorted_bonds > 0));
negative_bonds = sum(sorted_density(sorted_bonds < 0));
stats_text = sprintf(['Summary Statistics:\n' ...
'Mean Bonds: %.2f\n' ...
'Median Bonds: %.2f\n' ...
'Fraction with Positive Bonds: %.1f%%\n' ...
'Fraction with Negative Bonds: %.1f%%'], ...
mean_bonds, median_bonds, ...
positive_bonds*100, negative_bonds*100);
text(text_x, text_y, stats_text, 'FontSize', 10, 'BackgroundColor', 'white', ...
'EdgeColor', 'black', 'Margin', 5);
% Adjust axis limits to show full range with some padding
xlim([min(sorted_wealth)*1.1, max(sorted_wealth)*1.1]);
ylim([min(sorted_bonds)*1.1, max(sorted_bonds)*1.1]);
% Make the plot look nicer
set(gca, 'FontSize', 10);
box on;
% Print key statistics to console
fprintf('\nBond Holdings Distribution Statistics:\n');
fprintf('Mean bond holdings: %.4f\n', mean_bonds);
fprintf('Median bond holdings: %.4f\n', median_bonds);
fprintf('Fraction of households with positive bonds: %.1f%%\n', positive_bonds*100);
fprintf('Fraction of households with negative bonds: %.1f%%\n', negative_bonds*100);


% Store baseline results first
R_baseline = R;
V_baseline = V;
gc_baseline = gc;
ga_baseline = ga;
gl_baseline = gl;
D_baseline = D;
Y_baseline = Y;
CH_baseline = CH;
CF_baseline = CF;
B_baseline = B;

%=========================================================================
% Part 5: Model with Foreign Savings and Tax Adjustment
%=========================================================================
% Set foreign savings target (negative indicates foreign lending to domestic)
foreign_savings = -0.2;

% Initialize tax rate search
tau_min = 0.2;
tau_max = 0.4;
maxiter_tau = 20;

for iter_tau = 1:maxiter_tau
    tau = (tau_min + tau_max)/2;
    
    % Reset interest rate bounds for each tax iteration
    R_min = 1.001;
    R_max = 1.05;
    
    % Bisection on interest rate
    for iter_eq = 1:maxiter_eq
        R = (R_min + R_max)/2;
        
        % Exchange rate normalized to 1
        e = 1;
        PF = e;
        PH = e;
        x = PH/e;
        CH_star = alpha_H * x^(-theta_star) * Y_star;
        
        % Initialize value function and policy functions
        V = zeros(na, N);
        gc = zeros(na, N);
        ga = zeros(na, N);
        gl = zeros(na, N);
        gch = zeros(na, N);
        gcf = zeros(na, N);
        
        % Value Function Iteration (same as baseline)
        diff = 1;
        iter = 0;
        while diff > tol && iter < maxiter
            iter = iter + 1;
            V_old = V;
            for ia = 1:na
                for iz = 1:N
                    % Current state
                    a = a_grid(ia);
                    s = Z(iz);
                    
                    % Grid for labor choice
                    l_grid = linspace(0, 2, 20)';
                    
                    % Find optimal choices
                    value = -1e10 * ones(length(l_grid), na);
                    for il = 1:length(l_grid)
                        l = l_grid(il);
                        labor_income = (1 - tau) * PH * s * l;
                        
                        for ia_next = 1:na
                            a_next = a_grid(ia_next);
                            if a_next >= -kappa
                                resources = labor_income + R*PH*a - PH*a_next;
                                if resources > 0
                                    price_index = (PH/alpha_H)^alpha_H * (PF/(1-alpha_H))^(1-alpha_H);
                                    ch = alpha_H * (resources/PH);
                                    cf = (1-alpha_H) * (resources/PF);
                                    c = (ch^alpha_H) * (cf^(1-alpha_H));
                                    utility = log(c) - l^(1+eta)/(1+eta);
                                    
                                    EV = 0;
                                    for iz_next = 1:N
                                        EV = EV + P_trans(iz,iz_next) * V_old(ia_next,iz_next);
                                    end
                                    value(il,ia_next) = utility + beta*EV;
                                end
                            end
                        end
                    end
                    
                    % Find optimal choices
                    [maxval, idx] = max(value(:));
                    [il_opt, ia_next_opt] = ind2sub(size(value), idx);
                    
                    % Store optimal choices
                    V(ia,iz) = maxval;
                    gl(ia,iz) = l_grid(il_opt);
                    ga(ia,iz) = a_grid(ia_next_opt);
                    
                    % Store consumption choices
                    labor_income = (1 - tau) * PH * s * gl(ia,iz);
                    resources = labor_income + R*PH*a - PH*ga(ia,iz);
                    gch(ia,iz) = alpha_H * (resources/PH);
                    gcf(ia,iz) = (1-alpha_H) * (resources/PF);
                    gc(ia,iz) = (gch(ia,iz)^alpha_H) * (gcf(ia,iz)^(1-alpha_H));
                end
            end
            diff = max(abs(V(:) - V_old(:)));
        end
        
        % Compute stationary distribution (same as baseline)
        D = ones(na, N)/(na*N);
        dist_diff = 1;
        while dist_diff > tol
            D_new = zeros(na, N);
            for ia = 1:na
                for iz = 1:N
                    a_next = ga(ia,iz);
                    [~,ia_next] = min(abs(a_grid - a_next));
                    ia_next = min(max(ia_next, 1), na);
                    
                    if ia_next < na
                        weight = (a_next - a_grid(ia_next))/(a_grid(ia_next+1) - a_grid(ia_next));
                        weight = min(max(weight, 0), 1);
                    else
                        weight = 0;
                        ia_next = na - 1;
                    end
                    
                    for iz_next = 1:N
                        D_new(ia_next,iz_next) = D_new(ia_next,iz_next) + ...
                            (1-weight) * D(ia,iz) * P_trans(iz,iz_next);
                        D_new(ia_next+1,iz_next) = D_new(ia_next+1,iz_next) + ...
                            weight * D(ia,iz) * P_trans(iz,iz_next);
                    end
                end
            end
            D_new = D_new / sum(D_new(:));
            dist_diff = max(abs(D_new(:) - D(:)));
            D = D_new;
        end
        
        % Calculate aggregates
        L = 0; CH = 0; CF = 0; B = 0; Y = 0;
        for ia = 1:na
            for iz = 1:N
                L = L + gl(ia,iz) * Z(iz) * D(ia,iz);
                CH = CH + gch(ia,iz) * D(ia,iz);
                CF = CF + gcf(ia,iz) * D(ia,iz);
                B = B + a_grid(ia) * D(ia,iz);
                Y = Y + PH * Z(iz) * gl(ia,iz) * D(ia,iz);
            end
        end
        
        % Government budget with new tax rate
        tax_revenue = tau * Y/PH;
        BG_next = G_H + BG*R - tax_revenue;
        
        % Modified bond market clearing with foreign savings
        bond_market = B + BG + foreign_savings;
        goods_market = Y/PH - (CH + G_H + CH_star);
        
        % Update interest rate bounds
        if bond_market > 0
            R_max = R;
        else
            R_min = R;
        end
        
        % Check convergence
        if max(abs([goods_market, bond_market])) < tol_eq && abs(BG_next - BG) < tol_eq
            break;
        end
        
        BG = BG_next;
    end
    
    % Check government budget balance
    gov_budget = G_H + BG*R - tax_revenue;
    
    % Update tax rate bounds
    if gov_budget > 0
        tau_min = tau;
    else
        tau_max = tau;
    end
    
    % Check convergence of tax adjustment
    if abs(gov_budget) < tol_eq
        break;
    end
    
    fprintf('Tax iteration %d: tau = %.4f, Gov Budget = %.6f\n', iter_tau, tau, gov_budget);
end

% Store results with foreign savings
R_foreign = R;
V_foreign = V;
gc_foreign = gc;
ga_foreign = ga;
gl_foreign = gl;
D_foreign = D;
tau_foreign = tau;

% Plot comparison of results
figure('Position', [100, 100, 1000, 800]);

% Plot consumption policy function comparison
subplot(2,2,1);
plot(a_grid, gc_baseline(:,1), 'b-', a_grid, gc_foreign(:,1), 'r--', 'LineWidth', 1.5);
title('Consumption Policy Function (Low Productivity)');
legend('Baseline', 'With Foreign Savings');
xlabel('Assets');
ylabel('Consumption');
grid on;

% Plot asset policy function comparison
subplot(2,2,2);
plot(a_grid, ga_baseline(:,1), 'b-', a_grid, ga_foreign(:,1), 'r--', 'LineWidth', 1.5);
title('Asset Policy Function (Low Productivity)');
legend('Baseline', 'With Foreign Savings');
xlabel('Assets');
ylabel('Next Period Assets');
grid on;

% Plot labor supply comparison
subplot(2,2,3);
plot(a_grid, gl_baseline(:,1), 'b-', a_grid, gl_foreign(:,1), 'r--', 'LineWidth', 1.5);
title('Labor Supply (Low Productivity)');
legend('Baseline', 'With Foreign Savings');
xlabel('Assets');
ylabel('Labor Supply');
grid on;

% Plot wealth distribution comparison
subplot(2,2,4);
plot(a_grid, sum(D_baseline,2), 'b-', a_grid, sum(D_foreign,2), 'r--', 'LineWidth', 1.5);
title('Wealth Distribution');
legend('Baseline', 'With Foreign Savings');
xlabel('Assets');
ylabel('Density');
grid on;

% Print comparison results
fprintf('\nComparison of Results:\n');
fprintf('Interest Rate: Baseline = %.4f, Foreign Savings = %.4f\n', R_baseline, R_foreign);
fprintf('Tax Rate: Baseline = %.4f, Foreign Savings = %.4f\n', tau, tau_foreign);
fprintf('Change in Interest Rate: %.4f\n', R_foreign - R_baseline);
fprintf('Change in Tax Rate: %.4f\n', tau_foreign - tau);


%=========================================================================
% Part 6: Model with Foreign Savings and Government Spending Adjustment
%=========================================================================
% Reset tax to baseline and keep foreign savings
tau = 0.25;  % Reset to baseline tax rate
foreign_savings = -0.2;  % Keep foreign savings

% Initialize government spending search
G_min = 0.1;
G_max = 0.4;
maxiter_G = 30;

for iter_G = 1:maxiter_G
    G_H = (G_min + G_max)/2;
    
    % Reset interest rate bounds for each spending iteration
    R_min = 1.001;
    R_max = 1.05;
    
    % Bisection on interest rate
    for iter_eq = 1:maxiter_eq
        R = (R_min + R_max)/2;
        
        % Exchange rate normalized to 1
        e = 1;
        PF = e;
        PH = e;
        x = PH/e;
        CH_star = alpha_H * x^(-theta_star) * Y_star;
        
        % Initialize value function and policy functions
        V = zeros(na, N);
        gc = zeros(na, N);
        ga = zeros(na, N);
        gl = zeros(na, N);
        gch = zeros(na, N);
        gcf = zeros(na, N);
        
        % Value Function Iteration (same as before)
        diff = 1;
        iter = 0;
        while diff > tol && iter < maxiter
            iter = iter + 1;
            V_old = V;
            for ia = 1:na
                for iz = 1:N
                    % Current state
                    a = a_grid(ia);
                    s = Z(iz);
                    
                    % Grid for labor choice
                    l_grid = linspace(0, 2, 20)';
                    
                    % Find optimal choices
                    value = -1e10 * ones(length(l_grid), na);
                    for il = 1:length(l_grid)
                        l = l_grid(il);
                        labor_income = (1 - tau) * PH * s * l;
                        
                        for ia_next = 1:na
                            a_next = a_grid(ia_next);
                            if a_next >= -kappa
                                resources = labor_income + R*PH*a - PH*a_next;
                                if resources > 0
                                    price_index = (PH/alpha_H)^alpha_H * (PF/(1-alpha_H))^(1-alpha_H);
                                    ch = alpha_H * (resources/PH);
                                    cf = (1-alpha_H) * (resources/PF);
                                    c = (ch^alpha_H) * (cf^(1-alpha_H));
                                    utility = log(c) - l^(1+eta)/(1+eta);
                                    
                                    EV = 0;
                                    for iz_next = 1:N
                                        EV = EV + P_trans(iz,iz_next) * V_old(ia_next,iz_next);
                                    end
                                    value(il,ia_next) = utility + beta*EV;
                                end
                            end
                        end
                    end
                    
                    % Find optimal choices
                    [maxval, idx] = max(value(:));
                    [il_opt, ia_next_opt] = ind2sub(size(value), idx);
                    
                    % Store optimal choices
                    V(ia,iz) = maxval;
                    gl(ia,iz) = l_grid(il_opt);
                    ga(ia,iz) = a_grid(ia_next_opt);
                    
                    % Store consumption choices
                    labor_income = (1 - tau) * PH * s * gl(ia,iz);
                    resources = labor_income + R*PH*a - PH*ga(ia,iz);
                    gch(ia,iz) = alpha_H * (resources/PH);
                    gcf(ia,iz) = (1-alpha_H) * (resources/PF);
                    gc(ia,iz) = (gch(ia,iz)^alpha_H) * (gcf(ia,iz)^(1-alpha_H));
                end
            end
            diff = max(abs(V(:) - V_old(:)));
        end
        
        % Compute stationary distribution (same as before)
        D = ones(na, N)/(na*N);
        dist_diff = 1;
        while dist_diff > tol
            D_new = zeros(na, N);
            for ia = 1:na
                for iz = 1:N
                    a_next = ga(ia,iz);
                    [~,ia_next] = min(abs(a_grid - a_next));
                    ia_next = min(max(ia_next, 1), na);
                    
                    if ia_next < na
                        weight = (a_next - a_grid(ia_next))/(a_grid(ia_next+1) - a_grid(ia_next));
                        weight = min(max(weight, 0), 1);
                    else
                        weight = 0;
                        ia_next = na - 1;
                    end
                    
                    for iz_next = 1:N
                        D_new(ia_next,iz_next) = D_new(ia_next,iz_next) + ...
                            (1-weight) * D(ia,iz) * P_trans(iz,iz_next);
                        D_new(ia_next+1,iz_next) = D_new(ia_next+1,iz_next) + ...
                            weight * D(ia,iz) * P_trans(iz,iz_next);
                    end
                end
            end
            D_new = D_new / sum(D_new(:));
            dist_diff = max(abs(D_new(:) - D(:)));
            D = D_new;
        end
        
        % Calculate aggregates
        L = 0; CH = 0; CF = 0; B = 0; Y = 0;
        for ia = 1:na
            for iz = 1:N
                L = L + gl(ia,iz) * Z(iz) * D(ia,iz);
                CH = CH + gch(ia,iz) * D(ia,iz);
                CF = CF + gcf(ia,iz) * D(ia,iz);
                B = B + a_grid(ia) * D(ia,iz);
                Y = Y + PH * Z(iz) * gl(ia,iz) * D(ia,iz);
            end
        end
        
        % Government budget with fixed tax rate
        tax_revenue = tau * Y/PH;
        BG_next = G_H + BG*R - tax_revenue;
        
        % Modified bond market clearing with foreign savings
        bond_market = B + BG + foreign_savings;
        goods_market = Y/PH - (CH + G_H + CH_star);
        
        % Update interest rate bounds
        if bond_market > 0
            R_max = R;
        else
            R_min = R;
        end
        
        % Check convergence
        if max(abs([goods_market, bond_market])) < tol_eq && abs(BG_next - BG) < tol_eq
            break;
        end
        
        BG = BG_next;
    end
    
    % Check government budget balance
    gov_budget = G_H + BG*R - tax_revenue;
    
    % Update government spending bounds
    if gov_budget > 0
        G_max = G_H;
    else
        G_min = G_H;
    end
    
    % Check convergence of spending adjustment
    if abs(gov_budget) < tol_eq
        break;
    end
    
    fprintf('G iteration %d: G_H = %.4f, Gov Budget = %.6f\n', iter_G, G_H, gov_budget);
end

% Store results with spending adjustment
R_spending = R;
V_spending = V;
gc_spending = gc;
ga_spending = ga;
gl_spending = gl;
D_spending = D;
G_spending = G_H;

% Compare all three cases (baseline, tax adjustment, and spending adjustment)
figure('Position', [100, 100, 1200, 800]);

% Plot consumption policy function comparison
subplot(2,2,1);
plot(a_grid, gc_baseline(:,1), 'b-', ...
     a_grid, gc_foreign(:,1), 'r--', ...
     a_grid, gc_spending(:,1), 'g:', 'LineWidth', 1.5);
title('Consumption Policy Function (Low Productivity)');
legend('Baseline', 'Tax Adjustment', 'Spending Adjustment');
xlabel('Assets');
ylabel('Consumption');
grid on;

% Plot asset policy function comparison
subplot(2,2,2);
plot(a_grid, ga_baseline(:,1), 'b-', ...
     a_grid, ga_foreign(:,1), 'r--', ...
     a_grid, ga_spending(:,1), 'g:', 'LineWidth', 1.5);
title('Asset Policy Function (Low Productivity)');
legend('Baseline', 'Tax Adjustment', 'Spending Adjustment');
xlabel('Assets');
ylabel('Next Period Assets');
grid on;

% Plot labor supply comparison
subplot(2,2,3);
plot(a_grid, gl_baseline(:,1), 'b-', ...
     a_grid, gl_foreign(:,1), 'r--', ...
     a_grid, gl_spending(:,1), 'g:', 'LineWidth', 1.5);
title('Labor Supply (Low Productivity)');
legend('Baseline', 'Tax Adjustment', 'Spending Adjustment');
xlabel('Assets');
ylabel('Labor Supply');
grid on;

% Plot wealth distribution comparison
subplot(2,2,4);
plot(a_grid, sum(D_baseline,2), 'b-', ...
     a_grid, sum(D_foreign,2), 'r--', ...
     a_grid, sum(D_spending,2), 'g:', 'LineWidth', 1.5);
title('Wealth Distribution');
legend('Baseline', 'Tax Adjustment', 'Spending Adjustment');
xlabel('Assets');
ylabel('Density');
grid on;

% Print comparison results for all cases
fprintf('\nComparison of All Results:\n');
fprintf('Interest Rates:\n');
fprintf('Baseline = %.4f\n', R_baseline);
fprintf('Tax Adjustment = %.4f\n', R_foreign);
fprintf('Spending Adjustment = %.4f\n', R_spending);
fprintf('\nFiscal Policy:\n');
fprintf('Baseline: tau = %.4f, G = %.4f\n', 0.25, 0.2);
fprintf('Tax Adjustment: tau = %.4f, G = %.4f\n', tau_foreign, 0.2);
fprintf('Spending Adjustment: tau = %.4f, G = %.4f\n', 0.25, G_spending);