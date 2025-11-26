%% Objective function and constraints visualization
clc; clear; close all;

% Define grid
x = linspace(-2, 2, 400);
y = linspace(-2, 2, 400);
[X, Y] = meshgrid(x, y);

% Objective function
% f(x,y) = 2x^2 + y^2 - 2xy - 3x - 2y
F = 2*X.^2 + Y.^2 - 2*X.*Y - 3*X - 2*Y;

% Constraints
% y - x <= 0  -->  feasible when Y <= X
% x^2 + y^2 - 1 = 0  --> unit circle
g1 = Y - X;                % inequality
h1 = X.^2 + Y.^2 - 1;      % equality

feasible_region = (g1 <= 0) & (abs(h1) < 0.02);  
% NOTE: abs(h1)<0.02 → small band to visualize feasible region near the circle

%% Contour plot
figure;
contour(X, Y, F, 30, 'LineWidth', 1.2);
hold on;

% Plot constraints
fimplicit(@(x,y) y - x, 'r', 'LineWidth', 1.8);           % y = x line
fimplicit(@(x,y) x.^2 + y.^2 - 1, 'b', 'LineWidth', 1.8); % circle

% Shade feasible region (where both constraints hold)
scatter(X(feasible_region), Y(feasible_region), 8, 'filled', 'MarkerFaceColor', [0.2 0.8 0.2], 'MarkerFaceAlpha', 0.4);

title('Contour Plot of Objective Function with Constraints');
xlabel('x'); ylabel('y');
legend('f(x,y) contour', 'y = x', 'x^2 + y^2 = 1', 'Feasible region');
grid on;
axis equal;


%% Surface plot
figure;
surf(X, Y, F, 'EdgeColor', 'none');
colormap turbo;
hold on;

% Overlay constraints on surface
[Cx, Cy] = meshgrid(linspace(-1,1,200), linspace(-1,1,200));
Cz = 2*Cx.^2 + Cy.^2 - 2*Cx.*Cy - 3*Cx - 2*Cy;

% Plot unit circle constraint
theta = linspace(0, 2*pi, 300);
xc = cos(theta);
yc = cos(theta + pi/2); % or sin(theta)
zc = 2*xc.^2 + yc.^2 - 2*xc.*yc - 3*xc - 2*yc;
plot3(xc, yc, zc, 'k', 'LineWidth', 2);

title('Surface Plot of f(x,y) with Constraints');
xlabel('x'); ylabel('y'); zlabel('f(x,y)');
grid on;