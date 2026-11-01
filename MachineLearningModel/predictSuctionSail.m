function [CL, CD] = predictSuctionSail(lambda, Re, AoA, x, Cq)

% Load trained models
load('trainedModel_CL.mat', 'trainedModel_CL');
load('trainedModel_CD.mat', 'trainedModel_CD');

% Number of AoA points
n = length(AoA);

% Create input table
T = table( ...
    repmat(lambda, n, 1), ...
    repmat(Re, n, 1), ...
    AoA(:), ...
    repmat(x, n, 1), ...
    repmat(Cq, n, 1), ...
    'VariableNames', {'lambda', 'Re', 'AoA', 'x', 'Cq'});

% Predict lift coefficient
CL = trainedModel_CL.predictFcn(T);

% Predict drag coefficient
CD = trainedModel_CD.predictFcn(T);

end