from math import exp, isfinite, log, sqrt


def f(x):
    return sqrt(1 - x * x) - exp(x) + 0.1


def df(x):
    return -x / sqrt(1 - x * x) - exp(x)


def simple_iteration(x0, left, right, epsilon, q, max_iterations):
    x = x0
    history = [x]

    for _ in range(max_iterations):
        # Формулы (2.5)–(2.6): из e^x = sqrt(1 - x^2) + 0.1 выделяем x.
        x_new = log(sqrt(1 - x * x) + 0.1)
        if not left <= x_new <= right:
            return history, "Итерация вышла за пределы отрезка."

        history.append(x_new)
        # Формула (2.7): оценка ошибки по двум соседним приближениям.
        error_bound = q / (1 - q) * abs(x_new - x)
        if error_bound <= epsilon:
            return history, None

        x = x_new

    return history, "Лимит итераций исчерпан."


def newton(x0, left, right, epsilon, max_iterations):
    x = x0
    history = [x]

    for _ in range(max_iterations):
        derivative = df(x)
        if abs(derivative) <= 1e-12:
            return history, "Производная слишком мала для деления."

        # Формула (2.2): следующий x — пересечение касательной с осью Ox.
        x_new = x - f(x) / derivative
        if not left <= x_new <= right:
            return history, "Итерация вышла за пределы отрезка."

        history.append(x_new)
        if abs(x_new - x) <= epsilon and abs(f(x_new)) <= epsilon:
            return history, None

        x = x_new

    return history, "Лимит итераций исчерпан."


def print_result(name, history, stop_reason, control_root):
    print(f"\n{name}")
    print("Итерация | Отклонение от контрольного корня")
    iterations = len(history) - 1
    for k in range(min(iterations + 1, 8)):
        print(f"{k:8d} | {abs(history[k] - control_root):.3e}")

    x = history[-1]
    final_error = abs(x - control_root)
    if iterations >= 8:
        print(f"     ...\n{iterations:8d} | {final_error:.3e}")

    print(f"Число итераций: {iterations}")
    if stop_reason is None and x > 0:
        print(f"Положительный корень x ≈ {x:.16g}")
        initial_error = abs(history[0] - control_root)
        print(f"Отклонение уменьшилось с {initial_error:.3e} до {final_error:.3e}.")
    else:
        print(stop_reason or "Полученное приближение не является положительным.")
        print(f"Последнее приближение x = {x:.16g}")
    print(f"Невязка |f(x)| = {abs(f(x)):.3e}")


def plot_curves(left, right):
    from matplotlib import pyplot as plt

    width = right - left
    plot_left = max(0.0, left - width)
    plot_right = min(1.0, right + width)
    x_values = []
    sqrt_values = []
    exp_values = []
    for i in range(101):
        x = plot_left + (plot_right - plot_left) * i / 100
        x_values.append(x)
        sqrt_values.append(sqrt(1 - x * x) + 0.1)
        exp_values.append(exp(x))

    plt.plot(x_values, sqrt_values, label=r"$\sqrt{1-x^2}+0.1$")
    plt.plot(x_values, exp_values, label=r"$e^x$")
    plt.axvline(left, color="gray", linestyle="--", label="Границы отрезка")
    plt.axvline(right, color="gray", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Графическое отделение положительного корня")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    # Вариант №3. Отрезок выбран по графику пересечения двух кривых.
    left = 0.08
    right = 0.10
    x0 = right  # Здесь f(x0) < 0 и f''(x0) < 0, поэтому f(x0)*f''(x0) > 0.
    epsilon = 1e-6
    q = 0.10  # Верхняя граница |phi'(x)| на выбранном отрезке.
    max_iterations = 100

    if not isfinite(epsilon) or epsilon <= 0:
        print("Ошибка: epsilon должна быть положительным конечным числом.")
        return
    if not 0 <= left < right < 1:
        print("Ошибка: отрезок должен лежать в области 0 <= x < 1.")
        return

    print("Вариант №3. Уравнение sqrt(1 - x^2) - exp(x) + 0.1 = 0")
    print(f"Отрезок [{left}, {right}], начальное приближение x0 = {x0:g}")
    f_left = f(left)
    f_right = f(right)
    print(f"f({left:g}) = {f_left:+.6e}, f({right:g}) = {f_right:+.6e}")
    if f_left * f_right >= 0:
        print("На концах отрезка нет смены знака.")
        return
    print(f"Заданная точность epsilon = {epsilon:g}")

    iteration_history, iteration_stop = simple_iteration(
        x0, left, right, epsilon, q, max_iterations
    )
    newton_history, newton_stop = newton(x0, left, right, epsilon, max_iterations)

    # Независимое приближение из деления отрезка пополам служит только проверкой.
    control_root = 0.0914901486357767
    print_result("Метод простой итерации", iteration_history, iteration_stop, control_root)
    print_result("Метод Ньютона", newton_history, newton_stop, control_root)

    if iteration_stop is None and newton_stop is None:
        if len(newton_history) < len(iteration_history):
            print("\nДля этой точности метод Ньютона потребовал меньше итераций.")
        elif len(iteration_history) < len(newton_history):
            print("\nДля этой точности метод простой итерации потребовал меньше итераций.")
        else:
            print("\nОба метода потребовали одинаковое число итераций.")

    plot_curves(left, right)


if __name__ == "__main__":
    main()
