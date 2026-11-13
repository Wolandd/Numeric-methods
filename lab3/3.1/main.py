from math import pi, tan


def lagrange_value(nodes, values, x):
    result = 0.0
    for i in range(len(nodes)):
        # Базисный многочлен: в числителе стоит x - x_j.
        basis = 1.0
        for j in range(len(nodes)):
            if j != i:
                basis = basis * (x - nodes[j]) / (nodes[i] - nodes[j])
        result = result + values[i] * basis
    return result


def divided_differences(nodes, values):
    # Первый столбец таблицы состоит из значений функции.
    column = values.copy()
    coefficients = [column[0]]
    for order in range(1, len(nodes)):
        next_column = []
        for i in range(len(nodes) - order):
            difference = (column[i + 1] - column[i]) / (nodes[i + order] - nodes[i])
            next_column.append(difference)
        column = next_column
        coefficients.append(column[0])
    return coefficients


def newton_value(nodes, coefficients, x):
    result = coefficients[0]
    product = 1.0
    for k in range(1, len(coefficients)):
        product = product * (x - nodes[k - 1])
        result = result + coefficients[k] * product
    return result


def main():
    nodes_a = [0, pi / 8, 2 * pi / 8, 3 * pi / 8]
    nodes_b = [0, pi / 8, pi / 3, 3 * pi / 8]
    x_star = 3 * pi / 16

    exact_value = tan(x_star)
    print("Вариант №3. Интерполяция функции tan(x)")
    print(f"X* = {x_star:.10f}, tan(X*) = {exact_value:.10f}")
    errors = []

    for name, nodes in [("а", nodes_a), ("б", nodes_b)]:
        if len(set(nodes)) != len(nodes):
            print(f"Ошибка в наборе {name}: узлы должны быть различными.")
            return

        values = []
        print(f"\nНабор {name}")
        print(" i |          X_i |          Y_i")
        for i in range(len(nodes)):
            values.append(tan(nodes[i]))
            print(f"{i:2d} | {nodes[i]:12.9f} | {values[i]:12.9f}")

        coefficients = divided_differences(nodes, values)
        lagrange = lagrange_value(nodes, values, x_star)
        newton = newton_value(nodes, coefficients, x_star)
        lagrange_error = abs(exact_value - lagrange)
        newton_error = abs(exact_value - newton)
        errors.append(lagrange_error)

        print("Лагранж: L(x) = сумма Y_i * l_i(x) по строкам таблицы.")
        print("l_i(x) = произведение (x - X_j) / (X_i - X_j) по j != i.")
        print("Ньютон: P(x) = c0 + c1*(x-X0) + c2*(x-X0)*(x-X1) + ...")
        for k in range(len(coefficients)):
            print(f"c{k} = {coefficients[k]:.10f}")
        print(f"L(X*) = {lagrange:.10f}, абсолютная ошибка = {lagrange_error:.3e}")
        print(f"P(X*) = {newton:.10f}, абсолютная ошибка = {newton_error:.3e}")

        max_node_error = 0.0
        for i in range(len(nodes)):
            lagrange_at_node = lagrange_value(nodes, values, nodes[i])
            newton_at_node = newton_value(nodes, coefficients, nodes[i])
            max_node_error = max(max_node_error, abs(lagrange_at_node - values[i]),
                                 abs(newton_at_node - values[i]))
        print(f"Максимальное отклонение в узлах: {max_node_error:.3e}")
        print(f"Расхождение двух форм в X*: {abs(lagrange - newton):.3e}")

    print(f"\nОшибки интерполяции в X*: набор а — {errors[0]:.3e}, набор б — {errors[1]:.3e}.")


if __name__ == "__main__":
    main()
