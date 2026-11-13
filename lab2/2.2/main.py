from math import cos, isfinite, pi, sin, sqrt


def equations(x1, x2, a):
    f1 = (x1 * x1 + a * a) * x2 - a ** 3
    f2 = (x1 - a / 2) ** 2 + (x2 - a / 2) ** 2 - a * a
    return f1, f2


def phi(x1, x2, a):
    # Правая ветвь окружности и x2, выраженный из первого уравнения.
    new_x1 = a / 2 + sqrt(a * a - (x2 - a / 2) ** 2)
    new_x2 = a ** 3 / (x1 * x1 + a * a)
    return new_x1, new_x2


def simple_iteration(start, a, bounds, epsilon, q, max_iterations):
    x1, x2 = start
    left, right, bottom, top = bounds
    history = [start]

    for _ in range(max_iterations):
        # Обе новые координаты вычисляем из одной старой пары.
        new_x1, new_x2 = phi(x1, x2, a)
        if not (left <= new_x1 <= right and bottom <= new_x2 <= top):
            return history, "Приближение вышло за пределы области G."

        history.append((new_x1, new_x2))
        step = max(abs(new_x1 - x1), abs(new_x2 - x2))
        # Формула (2.26): оценка погрешности через норму шага.
        if q / (1 - q) * step <= epsilon:
            return history, None

        x1, x2 = new_x1, new_x2

    return history, "Лимит итераций исчерпан."


def newton(start, a, bounds, epsilon, max_iterations):
    x1, x2 = start
    left, right, bottom, top = bounds
    history = [start]

    for _ in range(max_iterations):
        f1, f2 = equations(x1, x2, a)
        j11 = 2 * x1 * x2
        j12 = x1 * x1 + a * a
        j21 = 2 * (x1 - a / 2)
        j22 = 2 * (x2 - a / 2)
        determinant = j11 * j22 - j12 * j21
        scale = max(abs(j11 * j22), abs(j12 * j21))
        if abs(determinant) <= 1e-12 * scale:
            return history, "Определитель Якобиана слишком мал для деления."

        # Формула (2.20): решаем J * delta = -F по правилу Крамера.
        delta1 = (-f1 * j22 + j12 * f2) / determinant
        delta2 = (-j11 * f2 + f1 * j21) / determinant
        x1 = x1 + delta1
        x2 = x2 + delta2
        if not (left <= x1 <= right and bottom <= x2 <= top):
            return history, "Приближение вышло за пределы области G."

        history.append((x1, x2))
        if max(abs(delta1), abs(delta2)) <= epsilon:
            return history, None

    return history, "Лимит итераций исчерпан."


def print_result(name, history, stop_reason, control, a, q):
    print(f"\n{name}")
    print("Итерация | Отклонение от контрольной пары в норме max")
    iterations = len(history) - 1
    for k in range(min(len(history), 9)):
        x1, x2 = history[k]
        error = max(abs(x1 - control[0]), abs(x2 - control[1]))
        print(f"{k:8d} | {error:.3e}")

    x1, x2 = history[-1]
    final_error = max(abs(x1 - control[0]), abs(x2 - control[1]))
    if iterations >= 9:
        print(f"     ...\n{iterations:8d} | {final_error:.3e}")

    print(f"Число итераций: {iterations}")
    if stop_reason is None and x1 > 0 and x2 > 0:
        print(f"Положительное решение: x1 ≈ {x1:.12g}, x2 ≈ {x2:.12g}")
        start1, start2 = history[0]
        initial_error = max(abs(start1 - control[0]), abs(start2 - control[1]))
        print(f"Отклонение уменьшилось с {initial_error:.3e} до {final_error:.3e}.")
        phi1, phi2 = phi(x1, x2, a)
        error_bound = max(abs(x1 - phi1), abs(x2 - phi2)) / (1 - q)
        print(f"Оценка ошибки по невязке x = φ(x): {error_bound:.3e}")
    else:
        print(stop_reason or "Полученная пара не является положительной.")
        print(f"Последнее приближение: x1 = {x1:.12g}, x2 = {x2:.12g}")

    f1, f2 = equations(x1, x2, a)
    print(f"Невязки исходной системы: F1 = {f1:+.3e}, F2 = {f2:+.3e}")


def plot_curves(a, bounds, start):
    from matplotlib import pyplot as plt

    left, right, bottom, top = bounds
    curve_x = []
    curve_y = []
    circle_x = []
    circle_y = []
    for i in range(361):
        x1 = left + (right - left) * i / 360
        curve_x.append(x1)
        curve_y.append(a ** 3 / (x1 * x1 + a * a))
        angle = 2 * pi * i / 360
        circle_x.append(a / 2 + a * cos(angle))
        circle_y.append(a / 2 + a * sin(angle))

    plt.plot(curve_x, curve_y, label="Первое уравнение")
    plt.plot(circle_x, circle_y, label="Второе уравнение (окружность)")
    plt.scatter(start[0], start[1], label="Начальное приближение", zorder=3)
    plt.xlim(left, right)
    plt.ylim(bottom, top)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Положительное пересечение в области G")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    a = 4
    bounds = (5.8, 6.0, 1.2, 1.4)
    start = (5.9, 1.25)  # По пересечению кривых в области G.
    epsilon = 1e-6
    q = 0.32  # Верхняя граница нормы матрицы производных φ на G.
    max_iterations = 100

    if not isfinite(epsilon) or epsilon <= 0:
        print("Ошибка: epsilon должна быть положительным конечным числом.")
        return

    print(f"Вариант №3. Параметр a = {a}, epsilon = {epsilon:g}")
    print(f"Начальное приближение по графику: {start}")
    iteration_history, iteration_stop = simple_iteration(
        start, a, bounds, epsilon, q, max_iterations
    )
    newton_history, newton_stop = newton(start, a, bounds, epsilon, max_iterations)

    # Контрольная пара получена делением пополам после исключения x2.
    control = (5.929262756907046, 1.251071307010313)
    print_result("Метод простой итерации", iteration_history, iteration_stop, control, a, q)
    print_result("Метод Ньютона", newton_history, newton_stop, control, a, q)

    if iteration_stop is None and newton_stop is None:
        if len(newton_history) < len(iteration_history):
            print("\nДля этой точности метод Ньютона потребовал меньше итераций.")
        elif len(iteration_history) < len(newton_history):
            print("\nДля этой точности метод простой итерации потребовал меньше итераций.")
        else:
            print("\nОба метода потребовали одинаковое число итераций.")

    plot_curves(a, bounds, start)


if __name__ == "__main__":
    main()
