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


def calculate_error(A):
    # t(A) = sqrt(сумма A[i][j]^2 при i < j).
    n = len(A)
    sum_squares = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            sum_squares = sum_squares + A[i][j] ** 2

    error = math.sqrt(sum_squares)
    return error


def method_rotation(A, epsilon, max_iterations=10000):
    n = len(A)

    # Сохраняем исходную матрицу и создаём V = I.
    current_A = []
    for row in A:
        current_A.append(row.copy())

    eigenvectors = []
    for i in range(n):
        eigenvectors.append([0.0] * n)
        eigenvectors[i][i] = 1.0

    error = calculate_error(current_A)
    errors = [error]
    iterations = 0

    while error > epsilon and iterations < max_iterations:
        # Шаг 1. Находим максимальный по модулю элемент выше диагонали.
        max_element = 0.0
        i = 0
        j = 1

        for row in range(n):
            for column in range(row + 1, n):
                if abs(current_A[row][column]) > max_element:
                    max_element = abs(current_A[row][column])
                    i = row
                    j = column

        # Шаг 2. Вычисляем угол по формуле из методички.
        difference = current_A[i][i] - current_A[j][j]
        if difference == 0:
            angle = math.pi / 4
        else:
            angle = 0.5 * math.atan(2 * current_A[i][j] / difference)

        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        # Шаг 3. Строим единичную матрицу и заменяем четыре элемента.
        rotation = []
        for row in range(n):
            rotation.append([0.0] * n)
            rotation[row][row] = 1.0

        rotation[i][i] = cos_angle
        rotation[j][j] = cos_angle
        rotation[i][j] = -sin_angle
        rotation[j][i] = sin_angle

        # Шаг 4. A_new = U^T * A_current * U.
        rotation_T = transpose_matrix(rotation)
        temp = multiply_matrices(rotation_T, current_A)
        new_A = multiply_matrices(temp, rotation)
        current_A = new_A

        # Шаг 5. V = V * U: накапливаем произведение всех вращений.
        eigenvectors = multiply_matrices(eigenvectors, rotation)

        # Шаг 6. Считаем выполненное вращение и новое значение t(A).
        iterations = iterations + 1
        error = calculate_error(current_A)
        errors.append(error)

    # Собственные значения берём из диагонали, векторы уже в столбцах V.
    eigenvalues = []
    for i in range(n):
        eigenvalues.append(current_A[i][i])

    return eigenvalues, eigenvectors, current_A, iterations, errors


def main():
    # Для другого варианта достаточно изменить A и epsilon.
    A = [
        [5, 5, 3],
        [5, -4, 1],
        [3, 1, 2]
    ]
    epsilon = 0.000001
    n = len(A)
    max_iterations = 10000

    if n < 2:
        print("Ошибка: размерность матрицы должна быть не меньше 2.")
        return
    if not math.isfinite(epsilon) or epsilon <= 0:
        print("Ошибка: epsilon должна быть положительным конечным числом.")
        return

    for i in range(n):
        if len(A[i]) != n:
            print("Ошибка: матрица A должна быть квадратной.")
            return

    for i in range(n):
        for j in range(i + 1, n):
            if not math.isclose(A[i][j], A[j][i], rel_tol=1e-12, abs_tol=1e-12):
                print("Ошибка: матрица A должна быть симметрической.")
                return

    print("Исходная симметрическая матрица A:")
    for row in A:
        print(row)
    print("Заданная точность epsilon =", epsilon)
    print("Размерность матрицы n =", n)

    eigenvalues, eigenvectors, final_A, iterations, errors = method_rotation(
        A, epsilon, max_iterations
    )

    if errors[-1] > epsilon:
        print("\nЛимит итераций исчерпан: заданная точность не достигнута.")
        print("Количество выполненных вращений:", iterations)
        print(f"Последнее значение t(A): {errors[-1]:.9e}")
        return

    print("\nИтоговая преобразованная матрица:")
    for i in range(n):
        for j in range(n):
            print(f"{final_A[i][j]:17.9e}", end=" ")
        print()

    print("\nСобственные значения и соответствующие собственные векторы:")
    for j in range(n):
        print(f"lambda{j + 1} = {eigenvalues[j]:.10f}")
        print(f"v{j + 1} (столбец {j + 1} матрицы V):")
        for i in range(n):
            print(f"  {eigenvectors[i][j]: .10f}")

    print("\nКоличество выполненных вращений:", iterations)
    print(f"Начальное значение t(A): {errors[0]:.9e}")
    print(f"Итоговое значение t(A):  {errors[-1]:.9e}")
    print("\nИтерация | Показатель t(A)")
    for k in range(len(errors)):
        print(f"{k:8d} | {errors[k]:.9e}")

    # Проверяем каждую пару с помощью исходной матрицы: A*v = lambda*v.
    print("\nПроверка A*v ≈ lambda*v (максимальная разность компонентов):")
    for j in range(n):
        vector = []
        for i in range(n):
            vector.append(eigenvectors[i][j])

        max_difference = 0.0
        for i in range(n):
            left_part = 0.0
            for k in range(n):
                left_part = left_part + A[i][k] * vector[k]
            right_part = eigenvalues[j] * vector[i]

            difference = abs(left_part - right_part)
            if difference > max_difference:
                max_difference = difference

        print(f"lambda{j + 1}, v{j + 1}: {max_difference:.9e}")

    # Проверяем ортогональность: V^T * V ≈ I.
    vectors_T = transpose_matrix(eigenvectors)
    product = multiply_matrices(vectors_T, eigenvectors)
    max_orthogonal_error = 0.0

    for i in range(n):
        for j in range(n):
            if i == j:
                expected = 1.0
            else:
                expected = 0.0

            difference = abs(product[i][j] - expected)
            if difference > max_orthogonal_error:
                max_orthogonal_error = difference

    print("\nМаксимальное отклонение V^T * V от I:")
    print(f"{max_orthogonal_error:.9e}")

    print("\nАнализ изменения t(A):")
    print("t(A) характеризует величину элементов выше главной диагонали.")
    decreases = True
    for k in range(1, len(errors)):
        if errors[k] > errors[k - 1]:
            decreases = False

    if iterations == 0:
        print("Начальное t(A) уже не превышает epsilon: вращения не требуются.")
    elif decreases:
        print("После каждого вращения t(A) не увеличивался.")
    else:
        print("В истории есть увеличение t(A); значения приведены в таблице.")

    print(f"Условие t(A) <= {epsilon:g} достигнуто за {iterations} вращений.")
    if errors[0] > 0:
        if errors[-1] == 0:
            print("Итоговый показатель t(A) равен нулю в машинной арифметике.")
        else:
            reduction = errors[0] / errors[-1]
            print(f"Показатель t(A) уменьшился в {reduction:.6e} раз.")

    print("\nСравнение для разных значений точности:")
    print("Точность     | Количество итераций | Итоговый t(A)")
    for test_epsilon in [0.01, 0.0001, 0.000001]:
        test_values, test_vectors, test_A, test_iterations, test_errors = (
            method_rotation(A, test_epsilon, max_iterations)
        )
        print(
            f"{test_epsilon:12g} | {test_iterations:19d} | {test_errors[-1]:.9e}"
        )
        if test_errors[-1] > test_epsilon:
            print("Для этой точности лимит итераций исчерпан.")

    print("Меньшая epsilon может потребовать больше вращений.")
    print("Для разных epsilon число вращений иногда совпадает.")


# При запуске файла выполнение начинается с вызова main().
if __name__ == "__main__":
    main()
