%Setup variables with nominal values
R1_nom = 30000;
R2_nom = 18000;          
C1_nom = 0.01e-6;     
C2_nom = 0.0047e-6; 

%Step 1: Generate Training Data
N = 100000;  % number of training samples
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
hiddenLayerSize = [20 10]; % arbitrary number set through trial and error
net = fitnet(hiddenLayerSize);  % creates neural network

% Train the network using the training data
tic;
[net,] = train(net, X_train, Y_train); %Stores trained network in net
training_time = toc;

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
diff = norm(Y_expected' - Y_pred);
mean_exp = mean(Y_expected);

% Calculate PDF
[pdf_values, xi_values] = ksdensity(Y_pred);
[pdf_exp,xi_exp] = ksdensity(Y_train); % expected PDF from MonteCarlo of training data

% Print Values used to evaluate model performance
fprintf('Training Data N = %d\n',N);
fprintf('Neural Network surrogate difference with expected = %f\n', diff);
fprintf('Neural Network surrogate mean |Vout| = %f\n', mean_pred);
fprintf('Neural Network surrogate std  |Vout| = %f\n', std_pred);
fprintf('expected Mean |Vout| = %f\n', mean_exp);
fprintf('Relative Mean Difference = %f\n', abs(100*(mean_exp-mean_pred)/mean_exp)); % expressed in %
fprintf('Computation Time (s) = %f\n',computation_time);
fprintf('Training Time (s) = %f\n',training_time);

% Plot the histogram of the neural network predicted |Vout|
figure;
histogram(Y_pred, 50);
title('Histogram of |V_{out}| from Neural Network Surrogate');
xlabel('|V_{out}|');
ylabel('Frequency');


% Plot the PDF curve
figure;
hold on
plot(xi_values, pdf_values, 'r-', 'LineWidth', 1.5);
plot(xi_exp, pdf_exp, 'b-', 'LineWidth', 1.5);
title('PDF of |V_{out}| from Neural Network Surrogate');
xlabel('|V_{out}|');
ylabel('Probability Density');
legend('Simulated PDF','Expected PDF','Location','Best');
