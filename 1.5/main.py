import math


def multiply_matrices(left, right):
    rows = len(left)
    columns = len(right[0])
    common_size = len(right)
    result = []

    for i in range(rows):
        result.append([0.0] * columns)
        for j in range(columns):
            for k in range(common_size):
                result[i][j] = result[i][j] + left[i][k] * right[k][j]

    return result


def transpose_matrix(A):
    rows = len(A)
    columns = len(A[0])
    result = []

    for j in range(columns):
        row = []
        for i in range(rows):
            row.append(A[i][j])
        result.append(row)

    return result


def qr_decomposition(A):
    n = len(A)

    # Шаг 1. R = A, Q = I. Исходная матрица не изменяется.
    R = []
    for row in A:
        R.append(row.copy())

    Q = []
    for i in range(n):
        Q.append([0.0] * n)
        Q[i][i] = 1.0

    for k in range(n - 1):
        # Шаг 2. Норма части столбца от R[k][k] до R[n-1][k].
        sum_squares = 0.0
        column_is_zero = True
        for i in range(k, n):
            sum_squares = sum_squares + R[i][k] * R[i][k]
            if R[i][k] != 0:
                column_is_zero = False

        if column_is_zero:
            continue

        norm_x = math.sqrt(sum_squares)
        if norm_x == 0 or not math.isfinite(norm_x):
            raise ValueError("Масштаб столбца не позволяет вычислить его норму.")

        if R[k][k] >= 0:
            sign = 1
        else:
            sign = -1

        # Шаг 3. Формула (1.30): v = x + sign(x_1) * ||x|| * e_1.
        v = [0.0] * n
        v[k] = R[k][k] + sign * norm_x
        for i in range(k + 1, n):
            v[i] = R[i][k]

        v_norm_squared = 0.0
        for i in range(n):
            v_norm_squared = v_norm_squared + v[i] * v[i]

        if v_norm_squared == 0 or not math.isfinite(v_norm_squared):
            raise ValueError("Невозможно безопасно разделить на квадрат нормы v.")

        # Шаг 4. Формула (1.29): H = I - 2 * v * v^T / (v^T * v).
        H = []
        for i in range(n):
            H.append([0.0] * n)
            H[i][i] = 1.0

        for i in range(n):
            for j in range(n):
                H[i][j] = H[i][j] - 2 * v[i] * v[j] / v_norm_squared

        # Шаг 5. Отражение обнуляет элементы столбца ниже диагонали.
        R = multiply_matrices(H, R)
        Q = multiply_matrices(Q, H)

    return Q, R


def eigenvalues_from_blocks(A, tolerance):
    n = len(A)
    eigenvalues = []
    block_sizes = []

    # Ниже первой поддиагонали не должно быть значительных элементов.
    for i in range(2, n):
        for j in range(i - 1):
            if abs(A[i][j]) > tolerance:
                return [], []

    i = 0
    while i < n:
        if i + 1 < n and abs(A[i + 1][i]) > tolerance:
            # Два соседних больших элемента означают блок как минимум 3x3.
            if i + 2 < n and abs(A[i + 2][i + 1]) > tolerance:
                return [], []

            a = A[i][i]
            b = A[i][i + 1]
            c = A[i + 1][i]
            d = A[i + 1][i + 1]

            # lambda^2 - (a + d) * lambda + a*d - b*c = 0.
            D = (a - d) * (a - d) + 4 * b * c
            if D >= 0:
                root_D = math.sqrt(D)
                lambda1 = (a + d + root_D) / 2
                lambda2 = (a + d - root_D) / 2
            else:
                real_part = (a + d) / 2
                imag_part = math.sqrt(-D) / 2
                lambda1 = complex(real_part, imag_part)
                lambda2 = complex(real_part, -imag_part)

            eigenvalues.append(lambda1)
            eigenvalues.append(lambda2)
            block_sizes.append(2)
            i = i + 2
        else:
            eigenvalues.append(A[i][i])
            block_sizes.append(1)
            i = i + 1

    return eigenvalues, block_sizes


def qr_algorithm(A, epsilon, max_iterations=10000):
    n = len(A)
    current_A = []
    for row in A:
        current_A.append(row.copy())

    matrix_scale = 0.0
    for i in range(n):
        for j in range(n):
            matrix_scale = max(matrix_scale, abs(A[i][j]))

    # Для малости элементов учитываем масштаб исходной матрицы.
    element_tolerance = epsilon * matrix_scale
    previous_values = []
    previous_blocks = []
    iterations = 0

    while iterations < max_iterations:
        # A_new = R*Q = Q^T*A*Q: подобие сохраняет собственные значения.
        Q, R = qr_decomposition(current_A)
        current_A = multiply_matrices(R, Q)
        iterations = iterations + 1

        for i in range(n):
            for j in range(n):
                if not math.isfinite(current_A[i][j]):
                    raise ValueError("При QR-итерациях возникло переполнение.")

        # Выделяем сразу все блоки; неполный набор корней не принимаем.
        eigenvalues, block_sizes = eigenvalues_from_blocks(current_A, element_tolerance)
        if len(eigenvalues) != n:
            previous_values = []
            previous_blocks = []
            continue

        # Сравниваем корни тех же блоков на двух соседних итерациях.
        if block_sizes == previous_blocks:
            values_are_stable = True
            for i in range(n):
                if not math.isfinite(abs(eigenvalues[i])):
                    raise ValueError("Не удалось вычислить конечные корни блоков.")
                difference = abs(eigenvalues[i] - previous_values[i])
                if difference > epsilon:
                    values_are_stable = False

            if values_are_stable:
                return current_A, eigenvalues, iterations, True

        previous_values = eigenvalues
        previous_blocks = block_sizes

    # Предварительные корни не выдаём как результат сошедшегося алгоритма.
    return current_A, [], iterations, False


def main():
    A = [
        [5, -5, -6],
        [-1, -8, -5],
        [2, 7, -3]
    ]
    epsilon = 0.000001
    n = len(A)

    # Проверяем исходные данные до начала матричных вычислений.
    if n < 2:
        print("Ошибка: размерность матрицы должна быть не меньше 2.")
        return

    for row in A:
        if len(row) != n:
            print("Ошибка: матрица должна быть квадратной.")
            return
        for value in row:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                print("Ошибка: элементы должны быть конечными вещественными числами.")
                return

    if not isinstance(epsilon, (int, float)):
        print("Ошибка: epsilon должна быть положительным конечным числом.")
        return
    if not math.isfinite(epsilon) or epsilon <= 0:
        print("Ошибка: epsilon должна быть положительным конечным числом.")
        return

    print("Исходная матрица A:")
    for row in A:
        for value in row:
            print(f"{value:16.8f}", end=" ")
        print()
    print(f"\nЗаданная точность epsilon = {epsilon:g}")
    print("Порог малости элементов = epsilon * максимальный модуль элемента A.")
    print("Порог изменения собственных значений = epsilon.")

    # Сначала отдельно выполняем и проверяем QR-разложение исходной A.
    Q, R = qr_decomposition(A)

    print("\nОртогональная матрица Q:")
    for row in Q:
        for value in row:
            print(f"{value:16.8f}", end=" ")
        print()

    print("\nВерхняя треугольная матрица R:")
    for row in R:
        for value in row:
            print(f"{value:16.8f}", end=" ")
        print()

    QR = multiply_matrices(Q, R)
    Q_T = transpose_matrix(Q)
    Q_T_Q = multiply_matrices(Q_T, Q)
    qr_error = 0.0
    orthogonal_error = 0.0
    matrix_scale = 0.0

    for i in range(n):
        for j in range(n):
            matrix_scale = max(matrix_scale, abs(A[i][j]))
            qr_error = max(qr_error, abs(QR[i][j] - A[i][j]))
            if i == j:
                expected = 1.0
            else:
                expected = 0.0
            orthogonal_error = max(orthogonal_error, abs(Q_T_Q[i][j] - expected))

    print(f"\nПроверка A ~ Q*R: максимальная ошибка = {qr_error:.3e}")
    print(f"Проверка Q^T*Q ~ I: максимальная ошибка = {orthogonal_error:.3e}")
    if qr_error > epsilon * matrix_scale or orthogonal_error > epsilon:
        print("Проверки QR-разложения не пройдены. Вычисления остановлены.")
        return
    print("Обе проверки QR-разложения пройдены.")

    print("\nНачинаем QR-алгоритм без сдвигов.")
    final_A, eigenvalues, iterations, converged = qr_algorithm(A, epsilon)

    print("\nМатрица после QR-итераций:")
    for row in final_A:
        for value in row:
            print(f"{value:16.8f}", end=" ")
        print()
    print(f"\nКоличество итераций: {iterations}")

    if not converged or len(eigenvalues) != n:
        print("Сходимость за отведённое число итераций не достигнута.")
        print("QR-алгоритм без сдвигов может не сходиться для данной матрицы.")
        print("Подтверждённый полный набор собственных значений не получен.")
        return

    print("\nВсе блоки выделены, собственные значения стабилизировались.")
    print("Найденные собственные значения:")
    for i in range(n):
        value = eigenvalues[i]
        if value.imag == 0:
            print(f"lambda{i + 1} = {value.real:.10f}")
        else:
            print(f"lambda{i + 1} = {value.real:.10f} {value.imag:+.10f}i")
            print(f"  Действительная часть: {value.real:.10f}")
            print(f"  Мнимая часть:         {value.imag:.10f}")

    trace_A = 0.0
    values_sum = 0.0
    values_product = 1.0
    for i in range(n):
        trace_A = trace_A + A[i][i]
        values_sum = values_sum + eigenvalues[i]
        values_product = values_product * eigenvalues[i]

    sum_error = abs(values_sum - trace_A)
    print(f"\nСлед исходной матрицы: {trace_A:.10f}")
    print(f"Сумма собственных значений: {values_sum.real:.10f} {values_sum.imag:+.10f}i")
    print(f"Ошибка проверки суммы: {sum_error:.3e}")
    if sum_error <= n * epsilon * matrix_scale:
        print("Проверка суммы собственных значений пройдена.")
    else:
        print("Проверка суммы не пройдена: результат требует уточнения.")

    # Эти контрольные числа используются только для заданного варианта.
    variant_A = [[5, -5, -6], [-1, -8, -5], [2, 7, -3]]
    if A == variant_A:
        control_values = [
            4.8315686776,
            complex(-5.4157843388, 5.8311875246),
            complex(-5.4157843388, -5.8311875246)
        ]
        print("\nСравнение с контрольными значениями варианта:")
        for control in control_values:
            nearest_error = abs(eigenvalues[0] - control)
            for value in eigenvalues:
                nearest_error = min(nearest_error, abs(value - control))
            print(f"  {control}: отклонение = {nearest_error:.3e}")

        print(f"Произведение собственных значений: {values_product.real:.10f}")
        print(f"Отклонение произведения от det(A) = 306: {abs(values_product - 306):.3e}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OverflowError) as error:
        print("Ошибка вычислений:", error)
