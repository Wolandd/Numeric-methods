from math import isclose, isfinite


def f(x):
    return x / (3 * x + 4) ** 3


def midpoint_rule(left, h, n):
    result = 0.0
    for i in range(n):
        midpoint = left + (i + 0.5) * h
        result = result + h * f(midpoint)
    return result


def trapezoid_rule(left, h, n):
    result = 0.0
    for i in range(n):
        x0 = left + i * h
        x1 = left + (i + 1) * h
        result = result + h * (f(x0) + f(x1)) / 2
    return result


def simpson_rule(left, h, n):
    result = 0.0
    # Формула (3.28): каждая пара соседних подынтервалов даёт веса 1, 4, 1.
    for i in range(0, n, 2):
        x0 = left + i * h
        x1 = left + (i + 1) * h
        x2 = left + (i + 2) * h
        result = result + h * (f(x0) + 4 * f(x1) + f(x2)) / 3
    return result


def antiderivative(x):
    t = 3 * x + 4
    return (-1 / t + 2 / t ** 2) / 9


def main():
    left = -1.0
    right = 1.0
    h1 = 0.5
    h2 = 0.25

    if left >= right or left <= -4 / 3 <= right:
        print("Ошибка: нужен возрастающий отрезок интегрирования без полюса x = -4/3.")
        return

    counts = []
    for h in (h1, h2):
        if not isfinite(h) or h <= 0:
            print("Ошибка: шаг должен быть положительным конечным числом.")
            return
        n = round((right - left) / h)
        if n < 2 or n % 2 != 0 or not isclose(n * h, right - left, rel_tol=1e-12):
            print("Ошибка: шаг должен разбивать отрезок на чётное число подынтервалов.")
            return
        counts.append(n)
    if h2 >= h1:
        print("Ошибка: для оценки Рунге–Ромберга второй шаг должен быть меньше первого.")
        return

    methods = [
        ("Средние прямоугольники", midpoint_rule, 2),
        ("Трапеции", trapezoid_rule, 2),
        ("Симпсон", simpson_rule, 4),
    ]
    exact = antiderivative(right) - antiderivative(left)
    print("Вариант №3. Интеграл x/(3x+4)^3")
    print(f"Отрезок [{left:g}, {right:g}], h1 = {h1:g}, h2 = {h2:g}")
    print(f"Точное значение по первообразной ≈ {exact:.10f}")
    print("Метод                  | p |        I(h1) |        I(h2) | Оценка Р–Р | Ошибка h2")
    for name, method, order in methods:
        coarse = method(left, h1, counts[0])
        fine = method(left, h2, counts[1])
        # Формула (3.30): модуль поправки оценивает ошибку мелкой сетки.
        runge_romberg = abs(fine - coarse) / ((h1 / h2) ** order - 1)
        error = abs(fine - exact)
        print(f"{name:22s} | {order} | {coarse:12.9f} | {fine:12.9f} | "
              f"{runge_romberg:10.3e} | {error:.3e}")


if __name__ == "__main__":
    main()
