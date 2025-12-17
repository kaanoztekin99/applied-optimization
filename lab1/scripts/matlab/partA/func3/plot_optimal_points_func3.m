%% FUNC3: Optimal points on constraint (numerically computed)
clc; clear; close all;

out_dir = "../../../../figures/matlab/func3";
if ~exist(out_dir, "dir")
    mkdir(out_dir);
end

% 1) Feasible curve parametrization
% Constraint: (x+1)^2 + y^2 = 17
R = sqrt(17);
t = linspace(0, 2*pi, 20000);

x = -1 + R*cos(t);
y = R*sin(t);

% Objective values along feasible curve
fvals = 9*x.^2 + 13*y.^2 + 18*x.*y - 4;

% Numerical min and max
[Fmin, i_min] = min(fvals);
[Fmax, i_max] = max(fvals);

min_pt = [x(i_min), y(i_min), Fmin];
max_pt = [x(i_max), y(i_max), Fmax];

fprintf("\n=== FUNC3 Optimal Points ===\n");
fprintf("Min point: (%.4f, %.4f), f = %.4f\n", min_pt(1), min_pt(2), min_pt(3));
fprintf("Max point: (%.4f, %.4f), f = %.4f\n", max_pt(1), max_pt(2), max_pt(3));


% Contour plot with optimal points
[xg, yg] = meshgrid(linspace(-10,10,600));
Fg = 9*xg.^2 + 13*yg.^2 + 18*xg.*yg - 4;

figure;
contour(xg, yg, Fg, 40, 'LineWidth', 1.2);
hold on;

% Feasible circle
plot(x, y, 'r', 'LineWidth', 2);

% Min & Max points
scatter(min_pt(1), min_pt(2), 80, 'm', 'filled');
scatter(max_pt(1), max_pt(2), 80, 'y', 'filled');

title("FUNC3: Contour Plot with Optimal Points");
xlabel("x"); ylabel("y");
legend("f(x,y) contour", "Constraint circle", "Min", "Max");
axis equal; grid on;

saveas(gcf, fullfile(out_dir, "fig_contour_optimal.png"));


% Surface plot with optimal points
figure;
surf(xg, yg, Fg, 'EdgeColor','none');
colormap turbo;
hold on;

% Constraint curve on surface
zc = 9*x.^2 + 13*y.^2 + 18*x.*y - 4;
plot3(x, y, zc, 'k', 'LineWidth', 2);

% Optimal points
plot3(min_pt(1), min_pt(2), min_pt(3), 'mo', 'MarkerSize', 8, "MarkerFaceColor","m");
plot3(max_pt(1), max_pt(2), max_pt(3), 'yo', 'MarkerSize', 8, "MarkerFaceColor","y");

title("FUNC3: Surface Plot with Optimal Points");
xlabel("x"); ylabel("y"); zlabel("f(x,y)");
grid on;

saveas(gcf, fullfile(out_dir, "fig_surface_optimal.png"));