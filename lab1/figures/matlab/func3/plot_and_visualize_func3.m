%% FUNC3: Objective function and constraint visualization
clc; clear; close all;

% Grid
x = linspace(-10, 10, 600);
y = linspace(-10, 10, 600);
[X, Y] = meshgrid(x, y);

% Objective function
% f(x,y) = 9x^2 + 13y^2 + 18xy - 4
F = 9*X.^2 + 13*Y.^2 + 18*X.*Y - 4;

% Constraint
% x^2 + y^2 + 2x = 16
% → (x+1)^2 + y^2 = 17 (circle)
h1 = (X + 1).^2 + Y.^2 - 17;

% Feasible region band for visualization
feasible = abs(h1) < 0.05;

% Contour plot
figure;
contour(X, Y, F, 40, 'LineWidth', 1.2);
hold on;

% Constraint circle
fimplicit(@(x,y) (x+1).^2 + y.^2 - 17, 'r', 'LineWidth', 2);

scatter(X(feasible), Y(feasible), 8, 'filled', ...
    'MarkerFaceColor',[0.2 0.8 0.2], 'MarkerFaceAlpha',0.4);

title('FUNC3: Contour Plot with Constraint (x+1)^2 + y^2 = 17');
xlabel('x'); ylabel('y');
legend('f(x,y) contour','Constraint circle','Feasible region');
grid on; axis equal;


% Surface plot
figure;
surf(X, Y, F, 'EdgeColor', 'none');
colormap turbo;
hold on;

% Constraint curve on surface
theta = linspace(0, 2*pi, 400);
xc = -1 + sqrt(17)*cos(theta);
yc = sqrt(17)*sin(theta);
zc = 9*xc.^2 + 13*yc.^2 + 18*xc.*yc - 4;

plot3(xc, yc, zc, 'k', 'LineWidth', 2);

title('FUNC3: Surface Plot with Constraint Curve');
xlabel('x'); ylabel('y'); zlabel('f(x,y)');
grid on;