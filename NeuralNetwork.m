%Setup variables with nominal values
R1_nom = 30000;
R2_nom = 18000;          
C1_nom = 0.01e-6;     
C2_nom = 0.0047e-6; 

%Step 1: Generate Training Data
N = 2000;  % number of training samples
X_train = zeros(N, 4);  % [R1, R2, C1, C2]
Y_train = zeros(N, 1);  % |Vout|

for i = 1:N
    %Random distribution within ±5%
    R1 = R1_nom * (0.95 + 0.1 * rand());
    R2 = R2_nom * (0.95 + 0.1 * rand());
    C1 = C1_nom * (0.95 + 0.1 * rand());
    C2 = C2_nom * (0.95 + 0.1 * rand());
    
    % Store the values
    X_train(i, :) = [R1, R2, C1, C2];
    
    % Compute and store |Vout| using the simulation
    Vout = simulate_sallenKeyFilter(R1, R2, C1, C2);
    Y_train(i) = abs(Vout);
end

% Transpose data for training 
X_train = X_train';
Y_train = Y_train';

% Step 2: Define and Train the Neural Network
hiddenLayerSize = 10; % arbitrary number set through trial and error
net = fitnet(hiddenLayerSize);  % creates neural network

% Train the network using the training data
[net,] = train(net, X_train, Y_train); %Stores trained network in net

% Step 3: Evaluate the Neural Network 
N2 = 10000;  % number of test samples
X_test = zeros(4, N2);  % test input parameters (as columns)
Y_expected = zeros(N2, 1);  % circuit outputs from simulation

for i = 1:N2
    %Random distribution within ±5%
    R1 = R1_nom * (0.95 + 0.1 * rand());
    R2 = R2_nom * (0.95 + 0.1 * rand());
    C1 = C1_nom * (0.95 + 0.1 * rand());
    C2 = C2_nom * (0.95 + 0.1 * rand());
    
    % Store the values 
    X_test(:, i) = [R1; R2; C1; C2];
    
    % Compute and store |Vout| using the simulation
    Vout = simulate_sallenKeyFilter(R1, R2, C1, C2);
    Y_expected(i) = abs(Vout);
end


% Use the neural network surrogate to predict |Vout|
tic;
Y_pred = net(X_test);
computation_time = toc;

% Step 4: Compute statistics from the neural network predictions 
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
