def progonka(lower, diagonal, upper, rhs):
    n = len(diagonal)
    P = [0.0] * n
    Q = [0.0] * n

    # Прямой ход, как в ЛР 1.2. Ищем c[i] = P[i]*c[i+1] + Q[i].
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

    # Обратный ход от последней неизвестной к первой.
    result = [0.0] * n
    result[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        result[i] = P[i] * result[i + 1] + Q[i]
    return result


def build_spline(x, values):
    if len(x) != len(values) or len(x) < 2:
        raise ValueError("Нужно не менее двух узлов и по одному значению для каждого.")

    widths = []
    slopes = []
    for i in range(len(x) - 1):
        h = x[i + 1] - x[i]
        if h <= 0:
            raise ValueError("Узлы должны строго возрастать.")
        widths.append(h)
        slopes.append((values[i + 1] - values[i]) / h)

    # Формула (3.13) для внутренних c[i] = S''(x[i]) / 2.
    lower = []
    diagonal = []
    upper = []
    rhs = []
    for i in range(1, len(x) - 1):
        lower.append(widths[i - 1])
        diagonal.append(2 * (widths[i - 1] + widths[i]))
        upper.append(widths[i])
        rhs.append(3 * (slopes[i] - slopes[i - 1]))

    inner_c = []
    if diagonal:
        # Крайние c равны нулю, поэтому их слагаемые исчезают из системы.
        lower[0] = 0.0
        upper[-1] = 0.0
        inner_c = progonka(lower, diagonal, upper, rhs)
    c = [0.0] + inner_c + [0.0]

    coefficients = []
    for i in range(len(widths)):
        h = widths[i]
        a = values[i]
        b = slopes[i] - h * (2 * c[i] + c[i + 1]) / 3
        d = (c[i + 1] - c[i]) / (3 * h)
        coefficients.append((a, b, c[i], d))
    return coefficients


def polynomial_value(coefficients, t):
    a, b, c, d = coefficients
    return a + b * t + c * t ** 2 + d * t ** 3


def main():
    x = [0.0, 0.9, 1.8, 2.7, 3.6]
    values = [0.0, 0.36892, 0.85408, 1.7856, 6.3138]
    x_star = 1.5

    try:
        coefficients = build_spline(x, values)
    except ValueError as error:
        print(error)
        return
    if not x[0] <= x_star <= x[-1]:
        print("Ошибка: X* лежит вне отрезка интерполяции.")
        return

    print("Вариант №3. Естественный кубический сплайн")
    print("S_i(x) = a + b*t + c*t² + d*t³, где t = x - x_i")
    print("Интервал       |          a |          b |          c |          d")
    for i, (a, b, c, d) in enumerate(coefficients):
        print(f"[{x[i]:.1f}, {x[i + 1]:.1f}]     | {a:10.6f} | {b:10.6f} | {c:10.6f} | {d:10.6f}")

    for i in range(len(coefficients)):
        if x[i] <= x_star <= x[i + 1]:
            value = polynomial_value(coefficients[i], x_star - x[i])
            print(f"\nS({x_star:g}) ≈ {value:.10f} на интервале [{x[i]:g}, {x[i + 1]:g}].")
            break

    # Проверяем значения по обоим концам участков и производные в стыках.
    node_error = 0.0
    first_jump = 0.0
    second_jump = 0.0
    for i, (a, b, c, d) in enumerate(coefficients):
        h = x[i + 1] - x[i]
        right_value = polynomial_value(coefficients[i], h)
        node_error = max(node_error, abs(a - values[i]), abs(right_value - values[i + 1]))
        if i + 1 < len(coefficients):
            next_b = coefficients[i + 1][1]
            next_c = coefficients[i + 1][2]
            first_jump = max(first_jump, abs(b + 2 * c * h + 3 * d * h * h - next_b))
            second_jump = max(second_jump, abs(2 * c + 6 * d * h - 2 * next_c))

    last_c = coefficients[-1][2]
    last_d = coefficients[-1][3]
    last_h = x[-1] - x[-2]
    end_curvature = max(abs(2 * coefficients[0][2]), abs(2 * last_c + 6 * last_d * last_h))
    print(f"Максимальное отклонение в узлах: {node_error:.3e}")
    print(f"Максимальные скачки в стыках: S' — {first_jump:.3e}, S'' — {second_jump:.3e}")
    print(f"Максимальный модуль S'' на концах: {end_curvature:.3e}")


if __name__ == "__main__":
    main()
