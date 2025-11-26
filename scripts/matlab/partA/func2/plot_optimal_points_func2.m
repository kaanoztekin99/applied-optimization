%% FUNC2: Numerically computed optimal point on x + y = 4
clc; clear; close all;

out_dir = "../../../../figures/matlab/func2";
if ~exist(out_dir, "dir")
    mkdir(out_dir);
end

% Parameterize constraint: x + y = 4 → y = 4 - x
xs = linspace(-20, 20, 50000);
ys = 4 - xs;

% Objective values along constraint
fvals = 4*xs.^2 + 3*ys.^2 - 5*xs.*ys - 8*xs;

% Find global minimum numerically
[Fmin, i_min] = min(fvals);
min_pt = [xs(i_min), ys(i_min), Fmin];

fprintf("\n=== FUNC2 Optimal Points ===\n");
fprintf("Global minimum point: (%.4f, %.4f), f = %.4f\n", min_pt(1), min_pt(2), min_pt(3));
fprintf("Global maximum: DOES NOT EXIST (unbounded above along the line).\n");

% Contour plot with minimum point
[xg, yg] = meshgrid(linspace(-10,10,600));
Fg = 4*xg.^2 + 3*yg.^2 - 5*xg.*yg - 8*xg;

figure;
contour(xg, yg, Fg, 35, 'LineWidth', 1.3);
hold on;

% constraint line
xl = linspace(-10,10,400);
yl = 4 - xl;
plot(xl, yl, 'r', 'LineWidth', 2);

% optimal minimum
scatter(min_pt(1), min_pt(2), 80, 'm', 'filled');

title("FUNC2: Contour Plot with Minimum Point");
xlabel("x"); ylabel("y");
legend("f(x,y) contours", "x + y = 4", "Min");
axis equal; grid on;

saveas(gcf, fullfile(out_dir, "fig_contour_optimal.png"));


% Surface plot with minimum point
figure;
surf(xg, yg, Fg, 'EdgeColor','none');
colormap turbo;
hold on;

% constraint curve on surface
zl = 4*xl.^2 + 3*yl.^2 - 5*xl.*yl - 8*xl;
plot3(xl, yl, zl, 'k', 'LineWidth', 2);

% minimum point
plot3(min_pt(1), min_pt(2), min_pt(3), 'mo', ...
    'MarkerSize', 8, "MarkerFaceColor","m");

title("FUNC2: Surface Plot with Minimum Point");
xlabel("x"); ylabel("y"); zlabel("f(x,y)");
grid on;

saveas(gcf, fullfile(out_dir, "fig_surface_optimal.png"));