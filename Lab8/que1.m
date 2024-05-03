%------------------------------------------------------------
% Hopfield Network Example
% Demonstration of pattern storage
% CS308 Introduction to Artificial Intelligence
% Author: Pratik Shah
% Date: 3 April, 2019
% Place: IIITV, Gandhinagar
% Ref: Information, Inference and Learning Algorithms, D McKay
%------------------------------------------------------------
clear all;      % Clear all variables from memory
close all;      % Close all figures
clc;            % Clear the command window
%--------------------------------------------
% Patterns to store
% D, J, C, M
%--------------------------------------------
X = [1 1 1 1 -1 -1 1 -1 -1 1 -1 1 -1 -1 1 -1 1 -1 -1 1 -1 1 1 1 -1;
    1 1 1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 1 -1 -1 1 -1 1 1 1 -1 -1;
   -1 1 1 1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 -1 1 1 1 1;
    1 -1 -1 -1 1 1 1 -1 1 1 1 -1 1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1]';
%--------------------------------------------
% Learn the weights according to Hebb's rule
%--------------------------------------------
[m,n] = size(X);        % Get the dimensions of input patterns (25 x 4)
W = zeros(m,m);         % Initialize weight matrix (25 x 25)
for i = 1:n             % For each pattern
   W = W + X(:,i)*X(:,i)';     % Update weights using Hebb's rule
end
W(logical(eye(size(W)))) = 0;  % Set diagonal elements to zero (no self-connections)
W = W/n;                % Normalize weights
pattern = X(:,1);       % Select a pattern (D => 1; J => 2; C => 3; M => 4;)
min_err = 25;           % Initialize minimum error
avg_err = 0;            % Initialize average error
% Perform flipping and retrieval
for j = 1:1000          % Repeat the retrieval process 1000 times
   for i = 1:25        % For each possible number of flipped bits
       pattern_copy = pattern;        % Create a copy of the pattern
    indices_to_change = randperm(length(pattern_copy), i);   % Randomly  select indices to flip
       pattern_copy(indices_to_change) = -pattern_copy(indices_to_change);   % Flip selected bits
      output = sign(W*pattern_copy);     % Calculate output using the weight matrix
       err = norm(output - pattern);      % Calculate error between output and original pattern
       k = 25;     % Maximum iterations for convergence
       while err > 1 && k > 0   % While error is greater than 1 and maximum iterations are not reached
           output = sign(W*output);   % Update output using the weight matrix
           err = norm(output - pattern);  % Calculate new error
           k = k - 1;  % Decrement iteration counter
       end
       if err > 1     % If error is still greater than 1 after convergence
           imshow(reshape(-output, 5, 5)');   % Display the retrieved pattern
           min_err = min(i, min_err);   % Update minimum error
           avg_err = avg_err + i;       % Update average error
           break   % Break the loop
       end
   end
end
disp(min_err);  % Display minimum error
disp(avg_err/1000);  % Display average error
