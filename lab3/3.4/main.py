def differentiate(x, y, x_star):
    if len(x) != len(y) or x_star not in x:
        raise ValueError("Нужны парные значения x, y и точка X* из таблицы.")
    i = x.index(x_star)
    if i == 0 or i == len(x) - 1:
        raise ValueError("Точке X* нужны соседние узлы слева и справа.")

    left_step = x[i] - x[i - 1]
    right_step = x[i + 1] - x[i]
    if left_step <= 0 or right_step <= 0:
        raise ValueError("Три выбранных узла должны строго возрастать.")

    # Наклоны и вторая разделённая разность квадратичного интерполянта.
    left_slope = (y[i] - y[i - 1]) / left_step
    right_slope = (y[i + 1] - y[i]) / right_step
    difference = (right_slope - left_slope) / (x[i + 1] - x[i - 1])

    # Формулы (3.20) и (3.21): производные интерполяционного многочлена.
    first = left_slope + difference * (2 * x_star - x[i - 1] - x[i])
    second = 2 * difference
    return first, second, i


def main():
    x = [1.0, 1.5, 2.0, 2.5, 3.0]
    y = [0.0, 0.40547, 0.69315, 0.91629, 1.0986]
    x_star = 2.0

    try:
        first, second, i = differentiate(x, y, x_star)
    except ValueError as error:
        print(error)
        return

    print("Вариант №3. Численное дифференцирование")
    print(f"Использованные узлы: {x[i - 1]}, {x[i]}, {x[i + 1]}")
    print(f"f'({x_star:g}) ≈ {first:.8f}")
    print(f"f''({x_star:g}) ≈ {second:.8f}")

    # Для равномерных узлов сверяем ответ с центральными разностями.
    h = x[i] - x[i - 1]
    if abs(x[i + 1] - x[i] - h) <= 1e-12 * h:
        first_check = (y[i + 1] - y[i - 1]) / (2 * h)
        second_check = (y[i + 1] - 2 * y[i] + y[i - 1]) / h ** 2
        print(f"Расхождение с центральными разностями: {abs(first - first_check):.3e}, "
              f"{abs(second - second_check):.3e}")


if __name__ == "__main__":
    main()
