function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    theta_0 = theta(1);
    theta_1 = theta(2);

    temp = 0;
    for m_iter = 1:m
      temp = temp + ((theta' * X'(:, m_iter)) - y'(:, m_iter)) * X'(1, m_iter);
    end

    theta_0 = theta_0 - alpha * (1 / m) * temp;

    temp = 0;
    for m_iter = 1:m
      temp = temp + ((theta' * X'(:, m_iter)) - y'(:, m_iter)) * X'(2, m_iter);
    end

    theta_1 = theta_1 - alpha * (1 / m) * temp;

    theta = [theta_0; theta_1];

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
