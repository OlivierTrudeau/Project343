% Monte Carlo Simulation for Sallen-Key Filter
clear;
clc;
% Nominal values
R1_nom = 30000;          % in Ohms
R2_nom = 18000;          % in Ohms
C1_nom = 0.01e-6;        % in Farads
C2_nom = 0.0047e-6;      % in Farads

% Number of Monte Carlo iterations
N = 1000000;

% Pre-allocate array for storing output voltages
Vout = zeros(N,1);
X_test = zeros(4, N);  % test input parameters (as columns)

%Produce input
% Run Monte Carlo simulation
for i = 1:N
    % Generate random samples uniformly distributed within ±5% of the nominal value
    R1 = R1_nom * (0.95 + 0.1*rand()); % Random value between 0.95*R1_nom and 1.05*R1_nom
    R2 = R2_nom * (0.95 + 0.1*rand());
    C1 = C1_nom * (0.95 + 0.1*rand());
    C2 = C2_nom * (0.95 + 0.1*rand());

    % Store the values 
    X_test(:, i) = [R1; R2; C1; C2];
end

%Run the monte carlo method
tic;
for i = 1:N
    % Compute the filter output using the provided function
    R1 = X_test(1, i);
    R2 = X_test(2, i);
    C1 = X_test(3, i);
    C2 = X_test(4, i);

    Vout(i) = simulate_sallenKeyFilter(R1, R2, C1, C2);
end 
computation_time = toc;

% Compute the magnitude of Vout (since it is complex)
Vout_abs = abs(Vout);

% Calculate statistical properties
mean_Vout = mean(Vout_abs);
std_Vout = std(Vout_abs);

% Calculate the expected mean given nominal component values
Expectedmean = abs(simulate_sallenKeyFilter(R1_nom, R2_nom, C1_nom, C2_nom));

% Calculate PDF
[pdf_values, xi_values] = ksdensity(Vout_abs);

% Display results
fprintf('Iterations = %d\n', N);
fprintf('Mean |Vout| = %f\n', mean_Vout);
fprintf('Expected Mean = %f\n', Expectedmean);
fprintf('Relative Difference = %f\n', abs(100*(Expectedmean-mean_Vout)/Expectedmean)); % expressed in %
fprintf('Std  |Vout| = %f\n', std_Vout);
fprintf('Computation Time (s) = %f\n',computation_time);

% Plot the histogram of |Vout|
figure;
histogram(Vout_abs, 50);
title('Histogram of |V_{out}| from Monte Carlo Simulation');
xlabel('|V_{out}|');
ylabel('Frequency');

% Plot the PDF curve
figure;
hold on
plot(xi_values, pdf_values, 'r-', 'LineWidth', 1.5);
title('PDF of |V_{out}| from Monte Carlo Simulation');
xlabel('|V_{out}|');
ylabel('Probability Density');

