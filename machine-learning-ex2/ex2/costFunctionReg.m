function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

dim = length(theta);
h = sigmoid(theta' * X');
J = (sum(((-y .* log(h)') - ((1 - y) .* log(1 - h)'))(:)) / m) + ((lambda / (2 * m)) * sum((theta(2:end, :) .^ 2)(:)));
diff = h - y';

for d = 1:dim
  temp = 0;
  for m_iter = 1:m
    temp = temp + diff(1, m_iter) * X'(d, m_iter);
  end

  grad(d, 1) = temp / m;

  if d > 1
    grad(d, 1) =  grad(d, 1) + (lambda / m) * theta(d, 1);
  end
end




% =============================================================

end
