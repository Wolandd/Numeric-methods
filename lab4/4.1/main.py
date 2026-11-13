from math import exp, isclose, isfinite, sqrt


def acceleration(x, y):
    # При z = y' получаем y' = z, z' = 2y + 4x² exp(x²).
    return 2 * y + 4 * x * x * exp(x * x)


def exact_solution(x):
    return exp(x * x) + exp(x * sqrt(2)) + exp(-x * sqrt(2))


def euler(grid, h, y0, z0):
    y_values = [y0]
    z_values = [z0]
    for i in range(len(grid) - 1):
        y = y_values[i]
        z = z_values[i]
        # Формула (4.2): обе правые части берём в старой точке.
        y_values.append(y + h * z)
        z_values.append(z + h * acceleration(grid[i], y))
    return y_values, z_values


def rk4_step(x, y, z, h):
    # Формула (4.16): K — приращения y, L — приращения z.
    k1 = h * z
    l1 = h * acceleration(x, y)
    k2 = h * (z + l1 / 2)
    l2 = h * acceleration(x + h / 2, y + k1 / 2)
    k3 = h * (z + l2 / 2)
    l3 = h * acceleration(x + h / 2, y + k2 / 2)
    k4 = h * (z + l3)
    l4 = h * acceleration(x + h, y + k3)

    y_new = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    z_new = z + (l1 + 2 * l2 + 2 * l3 + l4) / 6
    return y_new, z_new


def runge_kutta(grid, h, y0, z0):
    y_values = [y0]
    z_values = [z0]
    for i in range(len(grid) - 1):
        y, z = rk4_step(grid[i], y_values[i], z_values[i], h)
        y_values.append(y)
        z_values.append(z)
    return y_values, z_values


def adams(grid, h, y0, z0):
    if len(grid) < 5:
        raise ValueError("Для метода Адамса нужно хотя бы четыре шага.")

    # Начальная точка и три шага РК4 дают первые четыре пары.
    y_values, z_values = runge_kutta(grid[:4], h, y0, z0)
    accelerations = []
    for i in range(4):
        accelerations.append(acceleration(grid[i], y_values[i]))

    # Формула (4.25) для каждой компоненты, начиная с i = 3.
    for i in range(3, len(grid) - 1):
        y_sum = 55 * z_values[i] - 59 * z_values[i - 1]
        y_sum = y_sum + 37 * z_values[i - 2] - 9 * z_values[i - 3]
        z_sum = 55 * accelerations[i] - 59 * accelerations[i - 1]
        z_sum = z_sum + 37 * accelerations[i - 2] - 9 * accelerations[i - 3]
        y_new = y_values[i] + h * y_sum / 24
        z_new = z_values[i] + h * z_sum / 24
        y_values.append(y_new)
        z_values.append(z_new)
        accelerations.append(acceleration(grid[i + 1], y_new))
    return y_values, z_values


def make_grid(left, h, steps):
    grid = []
    for i in range(steps + 1):
        grid.append(left + i * h)
    return grid


def error_summary(grid, coarse, fine, order):
    coarse_error = 0.0
    fine_error = 0.0
    runge_romberg = 0.0
    for i in range(len(grid)):
        exact = exact_solution(grid[i])
        coarse_error = max(coarse_error, abs(coarse[i] - exact))
        fine_error = max(fine_error, abs(fine[2 * i] - exact))
        estimate = abs(fine[2 * i] - coarse[i]) / (2 ** order - 1)
        runge_romberg = max(runge_romberg, estimate)
    return coarse_error, fine_error, runge_romberg


def main():
    left = 0.0
    right = 1.0
    y0 = 3.0
    z0 = 0.0
    h = 0.1

    if not isfinite(h) or h <= 0:
        print("Ошибка: шаг должен быть положительным конечным числом.")
        return
    steps = round((right - left) / h)
    if steps < 4 or not isclose(steps * h, right - left, rel_tol=1e-12):
        print("Ошибка: шаг должен делить отрезок на целое число частей, не меньше четырёх.")
        return

    grid = make_grid(left, h, steps)
    fine_grid = make_grid(left, h / 2, 2 * steps)
    methods = [("Эйлер", euler, 1), ("Рунге–Кутта", runge_kutta, 4), ("Адамс", adams, 4)]
    results = []
    fine_results = []
    for name, method, order in methods:
        results.append(method(grid, h, y0, z0))
        fine_results.append(method(fine_grid, h / 2, y0, z0))

    print("Вариант №3. Задача Коши для y'' = 2y + 4x² exp(x²)")
    print(f"Отрезок [{left:g}, {right:g}], y({left:g}) = {y0:g}, y'({left:g}) = {z0:g}")
    print(f"Основной шаг h = {h:g}, мелкий шаг h/2 = {h / 2:g}")
    print("     x |        Эйлер |  Рунге–Кутта |        Адамс |      Точное y")
    for i, x in enumerate(grid):
        print(f"{x:6.3f} | {results[0][0][i]:12.8f} | {results[1][0][i]:12.8f} | "
              f"{results[2][0][i]:12.8f} | {exact_solution(x):12.8f}")

    print("\nМаксимальные ошибки y по общим узлам")
    print("Метод        | p |   Ошибка h | Ошибка h/2 | Оценка Р–Р для h/2")
    for i, (name, method, order) in enumerate(methods):
        coarse_error, fine_error, estimate = error_summary(
            grid, results[i][0], fine_results[i][0], order
        )
        print(f"{name:12s} | {order} | {coarse_error:10.3e} | {fine_error:10.3e} | {estimate:.3e}")

    for step, solutions in [(h, results), (h / 2, fine_results)]:
        initial_error = 0.0
        for y_values, z_values in solutions:
            initial_error = max(initial_error, abs(y_values[0] - y0), abs(z_values[0] - z0))
        rk_y, rk_z = solutions[1]
        adams_y, adams_z = solutions[2]
        start_error = 0.0
        for i in range(1, 4):
            start_error = max(start_error, abs(rk_y[i] - adams_y[i]), abs(rk_z[i] - adams_z[i]))
        print(f"Шаг {step:g}: невязка начальных условий = {initial_error:.3e}, "
              f"расхождение старта Адамса с РК4 = {start_error:.3e}.")


if __name__ == "__main__":
    main()
