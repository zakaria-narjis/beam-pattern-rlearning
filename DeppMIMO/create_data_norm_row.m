% Initialize the result table with zeros
resultTable = zeros(32, 20100);

% Loop through each cell and extract the channel data
for i = 1:20100
    % Extract the struct from the current cell
    currentStruct = a{1,1}.user{1,i};
    
    % Convert the single precision complex numbers to double precision
    complexData = double(currentStruct.channel);
    
    % Store the complex data into the result table
    resultTable(:, i) = complexData;
end

% Normalize each row of the resultTable
for row = 1:32
    % Compute the max absolute value for the entire row (across 20100 columns)
    channel_max = max(abs(resultTable(row, :)));
    
    % Avoid division by zero, normalize the row if channel_max is non-zero
    if channel_max ~= 0
        resultTable(row, :) = resultTable(row, :) / channel_max;
    end
end

% Display the normalized result table
disp(resultTable);

% Save the normalized result table as a .mat file
matfilename = 'resultTable.mat';
save(matfilename, 'resultTable', '-v7.3');
