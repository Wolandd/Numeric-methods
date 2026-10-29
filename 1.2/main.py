def progonka(a, b, c, d):
    n = len(b)
    if len(a) != n or len(c) != n or len(d) != n:
        raise ValueError("Массивы a, b, c, d должны иметь одинаковую длину.")
    if n < 2:
        raise ValueError("Размерность системы должна быть не меньше 2.")

    # Ищем неизвестные в виде x[i] = P[i] * x[i + 1] + Q[i].
    P = [0.0] * n
    Q = [0.0] * n
    eps = 1e-12  # Порог для знаменателя относительно коэффициентов строки.

    # Шаг 1. Первые коэффициенты P и Q, формулы (1.5).
    scale = max(abs(b[0]), abs(c[0]))
    if abs(b[0]) <= eps * scale:
        raise ValueError(
            "Прогонку без дополнительных преобразований нельзя продолжить: "
            "знаменатель в строке 1 нулевой или слишком мал."
        )

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    # Шаг 2. Прямой ход: остальные P и Q, формулы (1.4) и (1.6).
    for i in range(1, n):
        denominator = b[i] + a[i] * P[i - 1]
        scale = max(abs(a[i]), abs(b[i]), abs(c[i]))
        if abs(denominator) <= eps * scale:
            raise ValueError(
                "Прогонку без дополнительных преобразований нельзя продолжить: "
                f"знаменатель в строке {i + 1} нулевой или слишком мал."
            )

        P[i] = -c[i] / denominator
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denominator

    # Шаг 3. Обратный ход, формула (1.7). При c[n - 1] = 0 имеем P[n - 1] = 0.
    x = [0.0] * n
    x[n - 1] = Q[n - 1]

    # Идём от предпоследней неизвестной к первой: x[i + 1] уже найдено.
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return P, Q, x


def main():
    # Исходные данные: три диагонали и правые части.
    # a[0] и c[n - 1] — служебные нули; x1 соответствует x[0].
    a = [0, -4, -1, 6, 4]
    b = [13, 9, -12, 20, 5]
    c = [-5, -5, -6, -5, 0]
    d = [-66, -47, -43, -74, 14]
    n = len(b)

    try:
        P, Q, x = progonka(a, b, c, d)
    except ValueError as error:
        print(error)
        return

    print("Исходные данные:")
    print("Нижняя диагональ a =", a)
    print("Главная диагональ b =", b)
    print("Верхняя диагональ c =", c)
    print("Вектор правых частей d =", d)
    print("Размерность системы n =", n)

    print("\nПрогоночные коэффициенты P:")
    for i in range(n):
        print(f"P{i + 1} = {P[i]:.6f}")

    print("\nПрогоночные коэффициенты Q:")
    for i in range(n):
        print(f"Q{i + 1} = {Q[i]:.6f}")

    print("\nРешение СЛАУ:")
    for i in range(n):
        print(f"x{i + 1} = {x[i]:.6f}")

    # Шаг 4. Проверяем решение подстановкой в исходные уравнения.
    max_residual = 0.0
    for i in range(n):
        left_part = b[i] * x[i]
        if i > 0:
            left_part = left_part + a[i] * x[i - 1]
        if i < n - 1:
            left_part = left_part + c[i] * x[i + 1]

        residual = left_part - d[i]
        if abs(residual) > max_residual:
            max_residual = abs(residual)

    print(f"\nМаксимальная абсолютная невязка: {max_residual:.6e}")


# При запуске файла выполнение начинается с вызова main().
if __name__ == "__main__":
    main()
