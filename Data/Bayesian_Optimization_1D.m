% Define prior input data
priorX = [0; 30; 50; 70; 100]; % Input values
priorY = [15; 17; 25; 30; 27]; % Corresponding objective function values

% Create a table for prior data
priorData = table(priorX, priorY, 'VariableNames', {'x', 'Objective'});

% Define the variable for optimization
xLimits = optimizableVariable('x', [0, 110]);

% Set up the Bayesian optimization
results = bayesopt(@(x) getObjective(x.x, priorData), ...
                   xLimits, ...
                   'MaxObjectiveEvaluations', 20, ...
                   'IsObjectiveDeterministic', true, ...
                   'AcquisitionFunctionName', 'expected-improvement', ...
                   'Verbose', 1);

% Display the best result
bestX = results.XAtMinObjective.x;
bestY = results.MinObjective; % Get the best objective value
fprintf('Best x: %.4f, Best f(x): %.4f\n', bestX, bestY);

% Function to get objective value based on prior data
function objVal = getObjective(x, priorData)
    % Fit a Gaussian process model to the prior data
    gpModel = fitrgp(priorData.x, priorData.Objective, 'KernelFunction', 'squaredexponential');
    
    % Predict the objective value at x
    [predictedY, ~] = predict(gpModel, x);
    objVal = -predictedY; % Return negative for maximization
end
