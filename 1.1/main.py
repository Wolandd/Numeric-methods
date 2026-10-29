def lu_decomposition(A):
    n = len(A)
    if n == 0:
        raise ValueError("Матрица A должна быть непустой.")
    for row in A:
        if len(row) != n:
            raise ValueError("Матрица A должна быть квадратной.")

    # Каждую строку копируем отдельно, чтобы сохранить исходную матрицу A.
    U = []
    for row in A:
        U.append(row.copy())

    L = []
    for i in range(n):
        L.append([0.0] * n)
        L[i][i] = 1.0

    # permutation[i] — номер исходной строки, оказавшейся на месте i.
    permutation = []
    for i in range(n):
        permutation.append(i)

    swap_count = 0
    eps = 1e-12  # Абсолютный порог для ведущего элемента.

    for k in range(n - 1):
        # Шаг 1. Выбираем максимальный по модулю элемент столбца k.
        pivot_row = k
        for i in range(k + 1, n):
            if abs(U[i][k]) > abs(U[pivot_row][k]):
                pivot_row = i

        if abs(U[pivot_row][k]) <= eps:
            raise ValueError(
                "LU-разложение невозможно в этой реализации: "
                "ведущий элемент нулевой или слишком мал."
            )

        if pivot_row != k:
            saved_row = U[k]
            U[k] = U[pivot_row]
            U[pivot_row] = saved_row

            saved_number = permutation[k]
            permutation[k] = permutation[pivot_row]
            permutation[pivot_row] = saved_number

            # В L переставляем только уже найденные коэффициенты.
            for j in range(k):
                saved_value = L[k][j]
                L[k][j] = L[pivot_row][j]
                L[pivot_row][j] = saved_value

            swap_count = swap_count + 1

        # Шаг 2. Прямой ход Гаусса: исключаем элементы ниже ведущего.
        for i in range(k + 1, n):
            multiplier = U[i][k] / U[k][k]
            L[i][k] = multiplier

            for j in range(k + 1, n):
                U[i][j] = U[i][j] - multiplier * U[k][j]

            U[i][k] = 0.0

    if abs(U[n - 1][n - 1]) <= eps:
        raise ValueError(
            "LU-разложение невозможно в этой реализации: "
            "последний диагональный элемент нулевой или слишком мал."
        )

    return L, U, permutation, swap_count


def solve_with_lu(L, U, permutation, b):
    n = len(L)

    permuted_b = [0.0] * n
    for i in range(n):
        permuted_b[i] = b[permutation[i]]

    # Прямая подстановка: L * z = P * b. На диагонали L стоят единицы.
    z = [0.0] * n
    for i in range(n):
        sum_of_previous_terms = 0.0
        for j in range(i):
            sum_of_previous_terms = sum_of_previous_terms + L[i][j] * z[j]

        z[i] = permuted_b[i] - sum_of_previous_terms

    # Обратная подстановка: U * x = z. Используем уже найденные x справа.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_of_known_terms = 0.0
        for j in range(i + 1, n):
            sum_of_known_terms = sum_of_known_terms + U[i][j] * x[j]

        x[i] = (z[i] - sum_of_known_terms) / U[i][i]

    return z, x


def inverse_from_lu(L, U, permutation):
    n = len(L)
    inverse_matrix = []
    for i in range(n):
        inverse_matrix.append([0.0] * n)

    # Столбец обратной матрицы — решение системы A * x = e_column.
    for column in range(n):
        unit_vector = [0.0] * n
        unit_vector[column] = 1.0

        z, column_solution = solve_with_lu(L, U, permutation, unit_vector)
        for i in range(n):
            inverse_matrix[i][column] = column_solution[i]

    return inverse_matrix


def print_matrix(name, matrix):
    print(name)
    for row in matrix:
        for value in row:
            print(f"{value:12.6f}", end=" ")
        print()
    print()


def main():
    # 1. Исходные данные.
    A = [
        [9, -5, -6, 3],
        [1, -7, 1, 0],
        [3, -4, 9, 0],
        [6, -1, 9, 8]
    ]
    b = [-8, 38, 47, -8]
    n = len(A)

    print_matrix("Исходная матрица A:", A)
    print("Вектор правых частей b:", b)
    print()

    # 2. LU-разложение вычисляем один раз: P * A = L * U.
    try:
        L, U, permutation, swap_count = lu_decomposition(A)
    except ValueError as error:
        print(error)
        return

    # 3. Матрицы разложения и перестановки строк.
    print_matrix("Матрица L:", L)
    print_matrix("Матрица U:", U)
    print("Перестановки строк (номера в Python начинаются с 0):", permutation)
    print("Число перестановок строк:", swap_count)
    print()

    # 4. Решаем L * z = P * b, затем U * x = z.
    z, x = solve_with_lu(L, U, permutation, b)
    print("Промежуточный вектор z (вектор y из задания):")
    for i in range(n):
        print(f"z{i + 1} = {z[i]:.6f}")
    print()

    print("Решение СЛАУ:")
    for i in range(n):
        print(f"x{i + 1} = {x[i]:.6f}")
    print()

    # 5. Определитель: произведение диагонали U с учётом перестановок.
    determinant = 1
    for i in range(n):
        determinant = determinant * U[i][i]
    if swap_count % 2 == 1:
        determinant = -determinant

    print(f"Определитель det(A) = {determinant:.6f}")
    print()

    # 6. Обратная матрица: решаем системы с единичными правыми частями.
    inverse_matrix = inverse_from_lu(L, U, permutation)
    print_matrix("Обратная матрица A_inverse:", inverse_matrix)

    # 7. Проверки результатов без округления вычисленных значений.
    # Проверка P * A = L * U: сравниваем соответствующие элементы.
    max_lu_error = 0.0
    for i in range(n):
        for j in range(n):
            lu_element = 0.0
            for k in range(n):
                lu_element = lu_element + L[i][k] * U[k][j]

            pa_element = A[permutation[i]][j]
            difference = abs(pa_element - lu_element)
            if difference > max_lu_error:
                max_lu_error = difference

    # Проверка A * x = b: вычисляем невязку r = A * x - b.
    residual = [0.0] * n
    max_residual = 0.0
    for i in range(n):
        ax_element = 0.0
        for j in range(n):
            ax_element = ax_element + A[i][j] * x[j]

        residual[i] = ax_element - b[i]
        if abs(residual[i]) > max_residual:
            max_residual = abs(residual[i])

    # Проверка A * A_inverse = I: на диагонали ожидаем 1, вне её — 0.
    max_inverse_error = 0.0
    for i in range(n):
        for j in range(n):
            product_element = 0.0
            for k in range(n):
                product_element = product_element + A[i][k] * inverse_matrix[k][j]

            if i == j:
                identity_element = 1.0
            else:
                identity_element = 0.0

            difference = abs(product_element - identity_element)
            if difference > max_inverse_error:
                max_inverse_error = difference

    print("Результаты проверок:")
    print(f"Максимальное отклонение P * A от L * U: {max_lu_error:.3e}")
    print("Невязка r = A * x - b:")
    for i in range(n):
        print(f"r{i + 1} = {residual[i]:.3e}")
    print(f"Максимальная абсолютная невязка: {max_residual:.3e}")
    print(f"Максимальное отклонение A * A_inverse от I: {max_inverse_error:.3e}")

    check_tolerance = 1e-9
    if (max_lu_error <= check_tolerance
            and max_residual <= check_tolerance
            and max_inverse_error <= check_tolerance):
        print(f"Все проверки пройдены с допуском {check_tolerance:.0e}.")
    else:
        print(f"Есть отклонение, превышающее допуск {check_tolerance:.0e}.")


# При запуске файла выполнение алгоритма начинается с main().
if __name__ == "__main__":
    main()
