from math import isfinite  # Проверяет, что число конечное: без бесконечности и NaN.


def metod_yakobi(alpha, beta, epsilon, q, max_iterations):
    n = len(beta)
    x_old = beta.copy()
    iterations = 0
    error = float("inf")

    while iterations < max_iterations:
        iterations = iterations + 1
        x_new = [0.0] * n

        # Формула (1.19): все компоненты считаем через старый вектор.
        for i in range(n):
            new_value = beta[i]

            for j in range(n):
                new_value = new_value + alpha[i][j] * x_old[j]

            if not isfinite(new_value):
                return None, iterations, None, False

            x_new[i] = new_value

        max_diff = 0.0
        for i in range(n):
            diff = abs(x_new[i] - x_old[i])
            if diff > max_diff:
                max_diff = diff

        if q < 1:
            # Оценка погрешности по формуле (1.20).
            error = q / (1 - q) * max_diff
        else:
            # Практический критерий без гарантии точности решения.
            error = max_diff

        if not isfinite(error):
            return None, iterations, None, False

        if error <= epsilon:
            return x_new, iterations, error, True

        x_old = x_new.copy()

    return x_old, iterations, error, False


def metod_zeidelya(alpha, beta, epsilon, q, norm_C, max_iterations):
    n = len(beta)
    x_old = beta.copy()
    iterations = 0
    error = float("inf")

    while iterations < max_iterations:
        iterations = iterations + 1
        x_new = [0.0] * n

        for i in range(n):
            new_value = beta[i]

            # Предыдущие неизвестные уже обновлены на этой итерации.
            for j in range(i):
                new_value = new_value + alpha[i][j] * x_new[j]

            # Следующие неизвестные берём с предыдущей итерации.
            for j in range(i + 1, n):
                new_value = new_value + alpha[i][j] * x_old[j]

            if not isfinite(new_value):
                return None, iterations, None, False

            x_new[i] = new_value

        max_diff = 0.0
        for i in range(n):
            diff = abs(x_new[i] - x_old[i])
            if diff > max_diff:
                max_diff = diff

        if q < 1:
            # Оценка погрешности метода Зейделя из раздела 1.1.4.
            error = norm_C / (1 - q) * max_diff
        else:
            error = max_diff

        if not isfinite(error):
            return None, iterations, None, False

        if error <= epsilon:
            return x_new, iterations, error, True

        x_old = x_new.copy()

    return x_old, iterations, error, False


def print_result(name, x, iterations, error, success, q):
    print(f"\nМетод {name}:")
    print("Количество итераций:", iterations)

    if x is None:
        print("Вычисления остановлены: получены недопустимые числовые значения.")
        return

    if success:
        if q < 1:
            print("Заданная точность достигнута по оценке погрешности.")
        else:
            print("Выполнен практический критерий max_diff <= epsilon.")
        print("Решение:")
    else:
        print("Лимит итераций исчерпан, критерий остановки не выполнен.")
        print("Последнее приближение:")

    for i in range(len(x)):
        print(f"x{i + 1} = {x[i]:.9f}")

    if q < 1:
        print(f"Итоговая оценка погрешности: {error:.9e}")
    else:
        print(f"Разность соседних приближений max_diff: {error:.9e}")


def max_nevyazka(A, b, x):
    n = len(A)
    max_residual = 0.0

    for i in range(n):
        left_part = 0.0
        for j in range(n):
            left_part = left_part + A[i][j] * x[j]

        residual = left_part - b[i]
        if not isfinite(residual):
            return None

        abs_residual = abs(residual)
        if abs_residual > max_residual:
            max_residual = abs_residual

    return max_residual


def main():
    # Исходные данные. Для другой СЛАУ достаточно изменить A, b и epsilon.
    A = [
        [-23, -7, 5, 2],
        [-7, -21, 4, 9],
        [9, 5, -31, -8],
        [0, 1, -2, 10]
    ]
    b = [-26, -55, -58, -24]
    epsilon = 0.000001
    n = len(A)
    max_iterations = 10000

    # Проверяем исходные данные перед делением на диагональные элементы.
    if n < 1:
        print("Ошибка: размерность системы должна быть не меньше 1.")
        return
    if len(b) != n:
        print("Ошибка: длина b должна совпадать с размерностью A.")
        return
    if not isfinite(epsilon) or epsilon <= 0:
        print("Ошибка: epsilon должна быть положительным конечным числом.")
        return

    for i in range(n):
        if len(A[i]) != n:
            print("Ошибка: матрица A должна быть квадратной.")
            return
        if A[i][i] == 0:
            print(f"Ошибка: диагональный элемент в строке {i + 1} равен нулю.")
            return
        if not isfinite(b[i]):
            print("Ошибка: вектор b содержит недопустимое числовое значение.")
            return
        for j in range(n):
            if not isfinite(A[i][j]):
                print("Ошибка: матрица A содержит недопустимое числовое значение.")
                return

    print("Исходная матрица A:")
    for i in range(n):
        print(A[i])
    print("Вектор правых частей b =", b)
    print("Заданная точность epsilon =", epsilon)
    print("Размерность системы n =", n)

    # Шаг 1. Приводим СЛАУ к виду x = beta + alpha*x, формулы (1.18).
    alpha = []
    beta = [0.0] * n

    for i in range(n):
        alpha.append([0.0] * n)
        beta[i] = b[i] / A[i][i]
        if not isfinite(beta[i]):
            print("Ошибка: при вычислении beta получено недопустимое значение.")
            return

        for j in range(n):
            if i == j:
                alpha[i][i] = 0
            else:
                alpha[i][j] = -A[i][j] / A[i][i]

            if not isfinite(alpha[i][j]):
                print("Ошибка: при вычислении alpha получено недопустимое значение.")
                return

    print("\nПреобразованная матрица alpha:")
    for i in range(n):
        for j in range(n):
            print(f"{alpha[i][j]:13.9f}", end=" ")
        print()

    print("\nВектор beta и начальное приближение x^(0) = beta для обоих методов:")
    for i in range(n):
        print(f"beta{i + 1} = {beta[i]:.9f}")

    # Шаг 2. Вычисляем бесконечные нормы alpha и её верхней части C.
    q = 0.0
    norm_C = 0.0

    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            row_sum = row_sum + abs(alpha[i][j])
        if row_sum > q:
            q = row_sum

        # Диагональ alpha нулевая, поэтому в C учитываем только j > i.
        row_sum_C = 0.0
        for j in range(i + 1, n):
            row_sum_C = row_sum_C + abs(alpha[i][j])
        if row_sum_C > norm_C:
            norm_C = row_sum_C

    if not isfinite(q):
        print("Ошибка: при вычислении нормы получено недопустимое значение.")
        return

    print(f"\nНорма q = ||alpha|| = {q:.9f}")
    print(f"Норма C = {norm_C:.9f}")
    if q < 1:
        print("q < 1: достаточное условие сходимости обоих методов выполнено.")
    else:
        print("Предупреждение: q >= 1, достаточное условие сходимости не выполнено.")
        print("Это не означает, что методы обязательно расходятся.")
        print("Остановка по max_diff <= epsilon не гарантирует точность истинного решения.")

    # Шаг 3. Метод Якоби: все новые значения зависят от старых.
    x_yakobi, iterations_yakobi, error_yakobi, success_yakobi = metod_yakobi(
        alpha, beta, epsilon, q, max_iterations
    )

    # Шаг 4. Метод Зейделя: сразу используем уже обновлённые значения.
    x_zeidel, iterations_zeidel, error_zeidel, success_zeidel = metod_zeidelya(
        alpha, beta, epsilon, q, norm_C, max_iterations
    )

    print_result("Якоби", x_yakobi, iterations_yakobi, error_yakobi, success_yakobi, q)
    print_result("Зейделя", x_zeidel, iterations_zeidel, error_zeidel, success_zeidel, q)

    # Шаг 5. Сравниваем результаты, если оба метода выполнили критерий остановки.
    if success_yakobi and success_zeidel:
        max_diff = 0.0
        for i in range(n):
            diff = abs(x_yakobi[i] - x_zeidel[i])
            if diff > max_diff:
                max_diff = diff
        print(f"\nМаксимальная разность решений двух методов: {max_diff:.9e}")

        if iterations_yakobi > iterations_zeidel:
            difference = iterations_yakobi - iterations_zeidel
            print(f"Для этой системы метод Зейделя потребовал на {difference} итераций меньше.")
        elif iterations_zeidel > iterations_yakobi:
            difference = iterations_zeidel - iterations_yakobi
            print(f"Для этой системы метод Якоби потребовал на {difference} итераций меньше.")
        else:
            print("Оба метода потребовали одинаковое количество итераций.")
    else:
        print("\nСравнение числа итераций не выполняется: не оба метода завершились успешно.")

    # Шаг 6. Проверяем подстановкой в исходную систему: r = A*x - b.
    print("\nПроверка полученных приближений по исходной СЛАУ:")
    if x_yakobi is not None:
        residual_yakobi = max_nevyazka(A, b, x_yakobi)
        if residual_yakobi is None:
            print("Якоби: при вычислении невязки получено недопустимое значение.")
        else:
            print(f"Максимальная абсолютная невязка (Якоби): {residual_yakobi:.9e}")

    if x_zeidel is not None:
        residual_zeidel = max_nevyazka(A, b, x_zeidel)
        if residual_zeidel is None:
            print("Зейдель: при вычислении невязки получено недопустимое значение.")
        else:
            print(f"Максимальная абсолютная невязка (Зейдель): {residual_zeidel:.9e}")


# При запуске файла выполнение начинается с вызова main().
if __name__ == "__main__":
    main()
