
clear;
clc;
close all;

%% 1. Define the input parameters

lambda = 0.5;        % Thickness ratio B/L
Re = 6000000;        % Reynolds number
x = 0.625;           % Flap position (dimensionless)
Cq = 0.05;           % Suction coefficient
AoA = 0:5:30;        % Angles of attack to evaluate

%% 2. Predict CL and CD using the trained models

[CL, CD] = predictSuctionSail(lambda, Re, AoA, x, Cq);

%% 3. Plot lift coefficient CL

figure;

plot(AoA, CL, '-o', ...
    'LineWidth', 1.5, ...
    'MarkerSize', 4);

grid on;
xlabel('Angle of Attack, AoA (deg)');
ylabel('Lift Coefficient, C_L');
title('Lift Polar');

%% 4. Plot drag coefficient CD

figure;

plot(AoA, CD, '-s', ...
    'LineWidth', 1.5, ...
    'MarkerSize', 4);

grid on;
xlabel('Angle of Attack, AoA (deg)');
ylabel('Drag Coefficient, C_D');
title('Drag Polar');

%% 5. Display the predicted values

Results = table(AoA(:), CL(:), CD(:), ...
    'VariableNames', {'AoA_deg', 'CL', 'CD'});

disp(Results);
