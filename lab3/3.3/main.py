def solve_gauss(A, b):
    n = len(b)
    matrix = []
    for i in range(n):
        matrix.append(A[i] + [b[i]])

    # Прямой ход: выбираем ведущую строку и исключаем элементы под диагональю.
    for k in range(n):
        pivot_row = k
        for i in range(k + 1, n):
            if abs(matrix[i][k]) > abs(matrix[pivot_row][k]):
                pivot_row = i
        if abs(matrix[pivot_row][k]) <= 1e-12:
            raise ValueError("Нормальная система вырождена или ведущий элемент слишком мал.")
        matrix[k], matrix[pivot_row] = matrix[pivot_row], matrix[k]

        for i in range(k + 1, n):
            multiplier = matrix[i][k] / matrix[k][k]
            for j in range(k + 1, n + 1):
                matrix[i][j] = matrix[i][j] - multiplier * matrix[k][j]
            matrix[i][k] = 0.0

    # Обратная подстановка по уже найденным коэффициентам.
    coefficients = [0.0] * n
    for i in range(n - 1, -1, -1):
        right_part = matrix[i][n]
        for j in range(i + 1, n):
            right_part = right_part - matrix[i][j] * coefficients[j]
        coefficients[i] = right_part / matrix[i][i]
    return coefficients


def fit_polynomial(x, y, degree):
    A = []
    b = []
    # Формула (3.17): строка k соответствует условию для коэффициента a_k.
    for k in range(degree + 1):
        row = [0.0] * (degree + 1)
        right_part = 0.0
        for j in range(len(x)):
            for i in range(degree + 1):
                row[i] = row[i] + x[j] ** (k + i)
            right_part = right_part + y[j] * x[j] ** k
        A.append(row)
        b.append(right_part)

    coefficients = solve_gauss(A, b)
    max_residual = 0.0
    for k in range(len(b)):
        left_part = 0.0
        for i in range(len(coefficients)):
            left_part = left_part + A[k][i] * coefficients[i]
        max_residual = max(max_residual, abs(left_part - b[k]))
    return coefficients, max_residual


def polynomial_value(coefficients, x):
    value = 0.0
    for i in range(len(coefficients)):
        value = value + coefficients[i] * x ** i
    return value


def plot_results(x, y, linear, quadratic):
    from matplotlib import pyplot as plt

    plot_x = []
    linear_y = []
    quadratic_y = []
    for i in range(101):
        point = x[0] + (x[-1] - x[0]) * i / 100
        plot_x.append(point)
        linear_y.append(polynomial_value(linear, point))
        quadratic_y.append(polynomial_value(quadratic, point))

    plt.scatter(x, y, color="black", label="Исходные точки", zorder=3)
    plt.plot(plot_x, linear_y, label="Многочлен 1-й степени")
    plt.plot(plot_x, quadratic_y, label="Многочлен 2-й степени")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Приближение методом наименьших квадратов")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    x = [-0.9, 0.0, 0.9, 1.8, 2.7, 3.6]
    y = [-0.36892, 0.0, 0.36892, 0.85408, 1.7856, 6.3138]

    if len(x) != len(y) or len(set(x)) < 3:
        print("Ошибка: нужны парные значения x, y и хотя бы три различных x.")
        return

    print("Вариант №3. Метод наименьших квадратов")
    polynomials = []
    errors = []
    for degree in (1, 2):
        try:
            coefficients, residual = fit_polynomial(x, y, degree)
        except ValueError as error:
            print(error)
            return

        # Формула (3.15): суммируем квадраты отклонений в исходных узлах.
        squared_error = 0.0
        for j in range(len(x)):
            difference = polynomial_value(coefficients, x[j]) - y[j]
            squared_error = squared_error + difference ** 2
        polynomials.append(coefficients)
        errors.append(squared_error)

        formula = f"{coefficients[0]:.9f}"
        for i in range(1, len(coefficients)):
            formula = formula + f" {coefficients[i]:+.9f}*x^{i}"
        print(f"\nP{degree}(x) = {formula}")
        print(f"Сумма квадратов ошибок: {squared_error:.9f}")
        print(f"Максимальная невязка нормальной системы: {residual:.3e}")

    if errors[1] <= errors[0] + 1e-12 * max(1, errors[0]):
        print("\nПроверка E₂ ≤ E₁ пройдена.")
    else:
        print("\nПроверка E₂ ≤ E₁ не пройдена.")
    plot_results(x, y, polynomials[0], polynomials[1])


if __name__ == "__main__":
    main()
