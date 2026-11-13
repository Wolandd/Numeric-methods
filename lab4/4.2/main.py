from math import isclose, isfinite


def acceleration(x, y):
    return 2 * y / (x * x * (x + 1))


def exact_solution(x):
    return 1 + 1 / x


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


def shoot_once(grid, h, eta, left_derivative, right_y, right_z, right_value):
    # Задана левая производная. В качестве параметра берём y(1) = eta.
    y = eta
    z = left_derivative
    values = [y]
    for i in range(len(grid) - 1):
        y, z = rk4_step(grid[i], y, z, h)
        values.append(y)
    residual = right_y * y + right_z * z - right_value
    return values, residual


def shooting(grid, h, left_derivative, right_y, right_z, right_value,
             eta0, eta1, epsilon, max_iterations):
    values0, residual0 = shoot_once(
        grid, h, eta0, left_derivative, right_y, right_z, right_value
    )
    if abs(residual0) <= epsilon:
        return values0, residual0, 1
    values1, residual1 = shoot_once(
        grid, h, eta1, left_derivative, right_y, right_z, right_value
    )
    shots = 2

    for _ in range(max_iterations):
        if abs(residual1) <= epsilon:
            return values1, residual1, shots
        denominator = residual1 - residual0
        scale = max(abs(residual1), abs(residual0))
        if not isfinite(denominator) or abs(denominator) <= 1e-12 * scale:
            raise ValueError("Слишком малая или недопустимая разность невязок в методе секущих.")

        # Формула (4.34): секущая для правой краевой невязки.
        eta2 = eta1 - residual1 * (eta1 - eta0) / denominator
        values2, residual2 = shoot_once(
            grid, h, eta2, left_derivative, right_y, right_z, right_value
        )
        eta0, eta1 = eta1, eta2
        residual0, residual1 = residual1, residual2
        values1 = values2
        shots = shots + 1

    if abs(residual1) <= epsilon:
        return values1, residual1, shots
    raise ValueError(f"Допуск стрельбы не достигнут за {shots} пристрелок.")


def progonka(lower, diagonal, upper, rhs):
    n = len(diagonal)
    P = [0.0] * n
    Q = [0.0] * n

    # Прямой ход, как в ЛР 1.2. Ищем y[i] = P[i]*y[i+1] + Q[i].
    for i in range(n):
        denominator = diagonal[i]
        right_part = rhs[i]
        if i > 0:
            denominator = denominator + lower[i] * P[i - 1]
            right_part = right_part - lower[i] * Q[i - 1]
        scale = max(abs(lower[i]), abs(diagonal[i]), abs(upper[i]))
        if abs(denominator) <= 1e-12 * scale:
            raise ValueError(f"Слишком малый знаменатель прогонки в строке {i + 1}.")
        P[i] = -upper[i] / denominator
        Q[i] = right_part / denominator

    result = [0.0] * n
    result[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        result[i] = P[i] * result[i + 1] + Q[i]
    return result


def finite_difference(grid, h, left_derivative, right_y, right_z, right_value):
    # Левая граница, формула (4.40): -y[0] + y[1] = h*y'(1).
    lower = [0.0]
    diagonal = [-1.0]
    upper = [1.0]
    rhs = [h * left_derivative]

    # Внутренние узлы: центральная вторая разность, уравнение умножено на h².
    for i in range(1, len(grid) - 1):
        lower.append(1.0)
        diagonal.append(-2 - h * h * acceleration(grid[i], 1.0))
        upper.append(1.0)
        rhs.append(0.0)

    # Правая граница: right_y*y[N] + right_z*(y[N]-y[N-1])/h = right_value.
    lower.append(-right_z)
    diagonal.append(h * right_y + right_z)
    upper.append(0.0)
    rhs.append(h * right_value)

    y = progonka(lower, diagonal, upper, rhs)
    left_residual = diagonal[0] * y[0] + upper[0] * y[1] - rhs[0]
    right_residual = lower[-1] * y[-2] + diagonal[-1] * y[-1] - rhs[-1]
    interior_residual = 0.0
    for i in range(1, len(grid) - 1):
        row_value = lower[i] * y[i - 1] + diagonal[i] * y[i] + upper[i] * y[i + 1]
        interior_residual = max(interior_residual, abs(row_value - rhs[i]))
    return y, left_residual, interior_residual, right_residual


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
    left = 1.0
    right = 2.0
    left_derivative = -1.0
    right_y = 2.0
    right_z = -4.0
    right_value = 4.0
    h = 0.1
    eta0 = 0.0
    eta1 = 1.0
    epsilon = 1e-12
    max_iterations = 20

    if not isfinite(h) or h <= 0:
        print("Ошибка: шаг должен быть положительным конечным числом.")
        return
    steps = round((right - left) / h)
    if steps < 2 or not isclose(steps * h, right - left, rel_tol=1e-12):
        print("Ошибка: шаг должен разбивать отрезок на целое число частей, не меньше двух.")
        return

    grid = make_grid(left, h, steps)
    fine_grid = make_grid(left, h / 2, 2 * steps)
    boundary = (left_derivative, right_y, right_z, right_value)
    try:
        shot = shooting(grid, h, *boundary, eta0, eta1, epsilon, max_iterations)
        fine_shot = shooting(fine_grid, h / 2, *boundary, eta0, eta1, epsilon, max_iterations)
        difference = finite_difference(grid, h, *boundary)
        fine_difference = finite_difference(fine_grid, h / 2, *boundary)
    except ValueError as error:
        print(error)
        return

    print("Вариант №3. Краевая задача для x²(x+1)y'' - 2y = 0")
    print(f"Отрезок [{left:g}, {right:g}], y'({left:g}) = {left_derivative:g}")
    print(f"Правая граница: {right_y:g}y({right:g}) {right_z:+g}y'({right:g}) = {right_value:g}")
    print(f"Основной шаг h = {h:g}, мелкий шаг h/2 = {h / 2:g}")
    print("     x |     Стрельба |   Кон. разности |      Точное y")
    for i, x in enumerate(grid):
        print(f"{x:6.3f} | {shot[0][i]:12.8f} | {difference[0][i]:15.8f} | {exact_solution(x):12.8f}")

    # РК4 имеет порядок 4. В разностной схеме границы (4.40) дают порядок 1.
    methods = [("Стрельба", shot[0], fine_shot[0], 4),
               ("Кон. разности", difference[0], fine_difference[0], 1)]
    print("\nМаксимальные ошибки y по общим узлам")
    print("Метод         | p |   Ошибка h | Ошибка h/2 | Оценка Р–Р для h/2")
    for name, coarse, fine, order in methods:
        coarse_error, fine_error, estimate = error_summary(grid, coarse, fine, order)
        print(f"{name:13s} | {order} | {coarse_error:10.3e} | {fine_error:10.3e} | {estimate:.3e}")

    print("\nКраевые и разностные проверки")
    for step, shot_result, difference_result in [(h, shot, difference), (h / 2, fine_shot, fine_difference)]:
        y, residual, shots = shot_result
        print(f"Шаг {step:g}, стрельба: y({left:g}) ≈ {y[0]:.10f}, пристрелок — {shots}, "
              f"правая невязка = {residual:+.3e}.")
        _, left_residual, interior_residual, right_residual = difference_result
        print(f"Шаг {step:g}, невязки разностных строк: левая {left_residual:+.3e}, "
              f"внутренние (максимум) {interior_residual:.3e}, правая {right_residual:+.3e}.")


if __name__ == "__main__":
    main()
