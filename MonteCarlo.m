% Monte Carlo Simulation for Sallen-Key Filter

% Nominal values
R1_nom = 30000;          % in Ohms
R2_nom = 18000;          % in Ohms
C1_nom = 0.01e-6;        % in Farads
C2_nom = 0.0047e-6;      % in Farads

% Number of Monte Carlo iterations
N = 10000;

% Pre-allocate array for storing output voltages
Vout = zeros(N,1);

% Run Monte Carlo simulation
for i = 1:N
    % Generate random samples uniformly distributed within ±5% of the nominal value
    R1 = R1_nom * (0.95 + 0.1*rand()); % Random value between 0.95*R1_nom and 1.05*R1_nom
    R2 = R2_nom * (0.95 + 0.1*rand());
    C1 = C1_nom * (0.95 + 0.1*rand());
    C2 = C2_nom * (0.95 + 0.1*rand());
    
    % Compute the filter output using the provided function
    Vout(i) = simulate_sallenKeyFilter(R1, R2, C1, C2);
end

% Compute the magnitude of Vout (since it is complex)
Vout_abs = abs(Vout);

% Calculate statistical properties
mean_Vout = mean(Vout_abs);
std_Vout = std(Vout_abs);

% Display results
fprintf('Mean |Vout| = %f\n', mean_Vout);
fprintf('Std  |Vout| = %f\n', std_Vout);

% Plot the histogram of |Vout|
figure;
histogram(Vout_abs, 50);
title('Histogram of |V_{out}| from Monte Carlo Simulation');
xlabel('|V_{out}|');
ylabel('Frequency');
