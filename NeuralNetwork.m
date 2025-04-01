%% Neural Network Surrogate for Sallen-Key Filter

% Nominal values
R1_nom = 30000;          % in Ohms
R2_nom = 18000;          % in Ohms
C1_nom = 0.01e-6;        % in Farads
C2_nom = 0.0047e-6;      % in Farads

%% Step 1: Generate Training Data
M = 200;  % number of training samples
X_train = zeros(M, 4);  % each row: [R1, R2, C1, C2]
Y_train = zeros(M, 1);  % corresponding |Vout|

for i = 1:M
    % Generate random samples within ±5% uncertainty
    R1 = R1_nom * (0.95 + 0.1 * rand());
    R2 = R2_nom * (0.95 + 0.1 * rand());
    C1 = C1_nom * (0.95 + 0.1 * rand());
    C2 = C2_nom * (0.95 + 0.1 * rand());
    
    % Save the input parameters
    X_train(i, :) = [R1, R2, C1, C2];
    
    % Compute the circuit output and take its magnitude
    Vout = simulate_sallenKeyFilter(R1, R2, C1, C2);
    Y_train(i) = abs(Vout);
end

% Transpose data for training (inputs as columns)
X_train = X_train';
Y_train = Y_train';

%% Step 2: Define and Train the Neural Network
hiddenLayerSize = 10; % you can adjust the number of neurons as needed
net = fitnet(hiddenLayerSize);  % creates a feedforward neural network

% (Optional) Customize training parameters here
net.trainParam.showWindow = true;  % shows the training window

% Train the network using the training data
[net, tr] = train(net, X_train, Y_train);

%% Step 3: Evaluate the Neural Network on a Test Set
M2 = 10000;  % number of test samples
X_test = zeros(4, M2);  % test input parameters (as columns)
Y_test_sim = zeros(M2, 1);  % circuit outputs from simulation

for i = 1:M2
    % Generate random test samples within ±5% uncertainty
    R1 = R1_nom * (0.95 + 0.1 * rand());
    R2 = R2_nom * (0.95 + 0.1 * rand());
    C1 = C1_nom * (0.95 + 0.1 * rand());
    C2 = C2_nom * (0.95 + 0.1 * rand());
    
    % Store the test sample parameters
    X_test(:, i) = [R1; R2; C1; C2];
    
    % Compute the true |Vout| using the simulation
    Vout = simulate_sallenKeyFilter(R1, R2, C1, C2);
    Y_test_sim(i) = abs(Vout);
end

% Use the neural network surrogate to predict |Vout|
Y_pred = net(X_test);

%% Step 4: Statistical Analysis and Visualization
% Compute statistics from the neural network predictions
mean_pred = mean(Y_pred);
std_pred = std(Y_pred);

fprintf('Neural Network surrogate mean |Vout| = %f\n', mean_pred);
fprintf('Neural Network surrogate std  |Vout| = %f\n', std_pred);

% Plot the histogram of the neural network predicted |Vout|
figure;
histogram(Y_pred, 50);
title('Histogram of |V_{out}| from Neural Network Surrogate');
xlabel('|V_{out}|');
ylabel('Frequency');
