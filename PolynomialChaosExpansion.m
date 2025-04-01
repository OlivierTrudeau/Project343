%% PCE-based Surrogate Model for Sallen-Key Filter

% Nominal values
R1_nom = 30000;        % Ohms
R2_nom = 18000;        % Ohms
C1_nom = 0.01e-6;      % Farads
C2_nom = 0.0047e-6;    % Farads

% Total polynomial order and number of variables
p = 2;      % Maximum total order
nvars = 4;  % R1, R2, C1, C2

%% Generate multi-index for basis functions (all indices with sum <= p)
multiIndex = [];
for i = 0:p
    for j = 0:p
        for k = 0:p
            for l = 0:p
                if (i + j + k + l <= p)
                    multiIndex = [multiIndex; i j k l];
                end
            end
        end
    end
end
nTerms = size(multiIndex,1);  % should be 15 for 4 variables, order 2

%% Step 1: Generate training samples in normalized space ξ ∈ [-1,1]
M = 100;  % number of training samples
xi = 2*rand(M, nvars) - 1;  % each row: [ξ1, ξ2, ξ3, ξ4]

% Map normalized variables to physical parameters:
R1_samples = R1_nom * (1 + 0.05 * xi(:,1));
R2_samples = R2_nom * (1 + 0.05 * xi(:,2));
C1_samples = C1_nom * (1 + 0.05 * xi(:,3));
C2_samples = C2_nom * (1 + 0.05 * xi(:,4));

%% Step 2: Evaluate the circuit at each sample
y = zeros(M,1);  % output |Vout|
for m = 1:M
    % Evaluate the circuit function (assumed to return complex number)
    Vout = simulate_sallenKeyFilter(R1_samples(m), R2_samples(m), C1_samples(m), C2_samples(m));
    y(m) = abs(Vout);  % take the magnitude
end

%% Step 3: Construct the design matrix using Legendre polynomials
% Define univariate Legendre polynomial function up to order 2
P = @(x, order) (order==0).*1 + (order==1).*x + (order==2).* (0.5*(3*x.^2 - 1));

A = zeros(M, nTerms);
for m = 1:M
    for term = 1:nTerms
        prod_val = 1;
        for var = 1:nvars
            order = multiIndex(term, var);
            prod_val = prod_val * P(xi(m, var), order);
        end
        A(m, term) = prod_val;
    end
end

%% Step 4: Solve for the PCE coefficients via regression
c = A \ y;  % least-squares solution

%% The surrogate model is now:
% y_hat(xi) = sum_{term=1}^{nTerms} c(term) * prod_{var=1}^{nvars} P(xi(var), multiIndex(term,var))

%% Step 5: Use the surrogate to perform statistical analysis
% Generate many test samples in the ξ-space
M2 = 10000;
xi_test = 2*rand(M2, nvars) - 1;
y_surrogate = zeros(M2,1);
for m = 1:M2
    s = 0;
    for term = 1:nTerms
        prod_val = 1;
        for var = 1:nvars
            order = multiIndex(term, var);
            prod_val = prod_val * P(xi_test(m, var), order);
        end
        s = s + c(term) * prod_val;
    end
    y_surrogate(m) = s;
end

% Compute statistics on the surrogate predictions
mean_surrogate = mean(y_surrogate);
std_surrogate = std(y_surrogate);

fprintf('PCE Surrogate Mean |Vout| = %f\n', mean_surrogate);
fprintf('PCE Surrogate Std  |Vout| = %f\n', std_surrogate);

%% Step 6: Plot the histogram of the surrogate predictions
figure;
histogram(y_surrogate, 50);
title('Histogram of |V_{out}| from PCE Surrogate');
xlabel('|V_{out}|');
ylabel('Frequency');
