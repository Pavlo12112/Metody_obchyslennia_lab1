import requests
import numpy as np
import matplotlib.pyplot as plt
import urllib3

# Вимкнення попередження про SSL-сертифікат
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================
# 1. Отримання даних з Open-Elevation API
# ============================================================

url = (
    "https://api.open-elevation.com/api/v1/lookup?"
    "locations="
    "48.164214,24.536044|"
    "48.164983,24.534836|"
    "48.165605,24.534068|"
    "48.166228,24.532915|"
    "48.166777,24.531927|"
    "48.167326,24.530884|"
    "48.167011,24.530061|"
    "48.166053,24.528039|"
    "48.166655,24.526064|"
    "48.166497,24.523574|"
    "48.166128,24.520214|"
    "48.165416,24.517170|"
    "48.164546,24.514640|"
    "48.163412,24.512980|"
    "48.162331,24.511715|"
    "48.162015,24.509462|"
    "48.162147,24.506932|"
    "48.161751,24.504244|"
    "48.161197,24.501793|"
    "48.160580,24.500537|"
    "48.160250,24.500106"
)

response = requests.get(url, verify=False)
response.raise_for_status()

data = response.json()
results = data["results"]


# ============================================================
# 2. Виведення отриманих GPS-даних
# ============================================================

n = len(results)

print("=" * 80)
print("ЛАБОРАТОРНА РОБОТА №1")
print("ІНТЕРПОЛЯЦІЯ КУБІЧНИМИ СПЛАЙНАМИ")
print("=" * 80)

print("\nКІЛЬКІСТЬ ВУЗЛІВ:", n)

print("\nТабуляція вузлів:")
print("№ | Latitude | Longitude | Elevation (m)")
print("-" * 60)

for i, point in enumerate(results):
    print(
        f"{i:2d} | "
        f"{point['latitude']:.6f} | "
        f"{point['longitude']:.6f} | "
        f"{point['elevation']:.2f}"
    )


# ============================================================
# 3. Формування координат та висот
# ============================================================

coords = [
    (point["latitude"], point["longitude"])
    for point in results
]

elevations = np.array(
    [point["elevation"] for point in results],
    dtype=float
)


# ============================================================
# 4. Функція Haversine
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    """
    Обчислення відстані між двома GPS-точками
    за формулою Haversine.
    """

    R = 6371000

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1)
        * np.cos(phi2)
        * np.sin(dlambda / 2) ** 2
    )

    return 2 * R * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )


# ============================================================
# 5. Обчислення кумулятивної відстані
# ============================================================

distances = [0.0]

for i in range(1, n):

    d = haversine(
        coords[i - 1][0],
        coords[i - 1][1],
        coords[i][0],
        coords[i][1]
    )

    distances.append(
        distances[-1] + d
    )

distances = np.array(distances)


print("\n" + "=" * 80)
print("ТАБУЛЯЦІЯ: ВІДСТАНЬ ТА ВИСОТА")
print("=" * 80)

print("№ | Distance (m) | Elevation (m)")
print("-" * 45)

for i in range(n):

    print(
        f"{i:2d} | "
        f"{distances[i]:12.2f} | "
        f"{elevations[i]:10.2f}"
    )


# ============================================================
# 6. Запис результатів у results.txt
# ============================================================

with open(
    "results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "ЛАБОРАТОРНА РОБОТА №1\n"
    )

    file.write(
        "Інтерполяція кубічними сплайнами\n\n"
    )

    file.write(
        "Табуляція GPS-вузлів\n"
    )

    file.write("=" * 70 + "\n")

    file.write(
        "№ | Latitude | Longitude | Elevation (m)\n"
    )

    for i, point in enumerate(results):

        file.write(
            f"{i:2d} | "
            f"{point['latitude']:.6f} | "
            f"{point['longitude']:.6f} | "
            f"{point['elevation']:.2f}\n"
        )

    file.write(
        "\nТабуляція відстані та висоти\n"
    )

    file.write("=" * 70 + "\n")

    file.write(
        "№ | Distance (m) | Elevation (m)\n"
    )

    for i in range(n):

        file.write(
            f"{i:2d} | "
            f"{distances[i]:12.2f} | "
            f"{elevations[i]:10.2f}\n"
        )


# ============================================================
# 7. Метод прогонки
# ============================================================

def thomas_algorithm(a, b, c, d):
    """
    Розв'язання трьохдіагональної системи
    методом прогонки.

    a - нижня діагональ
    b - головна діагональ
    c - верхня діагональ
    d - вектор правої частини
    """

    n = len(d)

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float)

    # Пряма прогонка
    for i in range(1, n):

        m = a[i] / b[i - 1]

        b[i] = (
            b[i] - m * c[i - 1]
        )

        d[i] = (
            d[i] - m * d[i - 1]
        )

    # Зворотна прогонка
    x = np.zeros(n)

    x[-1] = d[-1] / b[-1]

    for i in range(n - 2, -1, -1):

        x[i] = (
            d[i] - c[i] * x[i + 1]
        ) / b[i]

    return x


# ============================================================
# 8. Побудова кубічного сплайна
# ============================================================

def cubic_spline(x, y):

    """
    Побудова натурального кубічного сплайна.

    Повертає коефіцієнти:
    a, b, c, d
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    n = len(x) - 1

    h = np.diff(x)

    lower = np.zeros(n + 1)
    main = np.ones(n + 1)
    upper = np.zeros(n + 1)
    rhs = np.zeros(n + 1)

    # Натуральні крайові умови
    main[0] = 1.0
    rhs[0] = 0.0

    main[n] = 1.0
    rhs[n] = 0.0

    # Внутрішні вузли
    for i in range(1, n):

        lower[i] = h[i - 1]

        main[i] = 2 * (
            h[i - 1] + h[i]
        )

        upper[i] = h[i]

        rhs[i] = 3 * (
            (y[i + 1] - y[i]) / h[i]
            -
            (y[i] - y[i - 1]) / h[i - 1]
        )

    # Розв'язання системи методом прогонки
    c_full = thomas_algorithm(
        lower,
        main,
        upper,
        rhs
    )

    # Коефіцієнти a, b, d
    a = y[:-1]

    b = np.zeros(n)

    d = np.zeros(n)

    for i in range(n):

        b[i] = (
            (y[i + 1] - y[i]) / h[i]
            -
            h[i] * (
                2 * c_full[i]
                + c_full[i + 1]
            ) / 3
        )

        d[i] = (
            c_full[i + 1]
            - c_full[i]
        ) / (
            3 * h[i]
        )

    return (
        a,
        b,
        c_full[:-1],
        d
    ), (
        lower,
        main,
        upper,
        rhs,
        c_full
    )


# ============================================================
# 9. Обчислення значення сплайна
# ============================================================

def spline_value(
    x,
    coefficients,
    xx
):

    a, b, c, d = coefficients

    xx = np.asarray(xx)

    result = np.zeros_like(
        xx,
        dtype=float
    )

    for j, value in enumerate(xx):

        if value <= x[0]:

            i = 0

        elif value >= x[-1]:

            i = len(x) - 2

        else:

            i = np.searchsorted(
                x,
                value
            ) - 1

        dx = value - x[i]

        result[j] = (
            a[i]
            + b[i] * dx
            + c[i] * dx ** 2
            + d[i] * dx ** 3
        )

    return result


# ============================================================
# 10. Кубічний сплайн для всіх 21 вузлів
# ============================================================

coefficients, system_data = cubic_spline(
    distances,
    elevations
)

lower, main, upper, rhs, c_full = system_data


# ============================================================
# 11. Виведення системи та коефіцієнтів
# ============================================================

print("\n" + "=" * 80)
print("МЕТОД ПРОГОНКИ")
print("=" * 80)

print(
    "Систему трьохдіагонального вигляду "
    "розв'язано методом прогонки."
)

print("\nЗнайдені коефіцієнти c_i:")

for i in range(len(c_full)):

    print(
        f"c[{i:2d}] = {c_full[i]:.10f}"
    )


print("\n" + "=" * 80)
print("КОЕФІЦІЄНТИ КУБІЧНОГО СПЛАЙНА")
print("=" * 80)

print(
    "№ | a | b | c | d"
)

print("-" * 80)

for i in range(len(coefficients[0])):

    a = coefficients[0][i]
    b = coefficients[1][i]
    c = coefficients[2][i]
    d = coefficients[3][i]

    print(
        f"{i:2d} | "
        f"{a:12.6f} | "
        f"{b:12.6f} | "
        f"{c:12.6f} | "
        f"{d:12.8f}"
    )


# ============================================================
# 12. Перевірка правильності методу прогонки
# ============================================================

print("\n" + "=" * 80)
print("ПЕРЕВІРКА РОЗВ'ЯЗКУ СИСТЕМИ")
print("=" * 80)

# Побудова лівої частини системи A*c
left_side = np.zeros(
    len(c_full)
)

for i in range(len(c_full)):

    left_side[i] = (
        lower[i] * c_full[i - 1]
        if i > 0
        else 0
    )

    left_side[i] += (
        main[i] * c_full[i]
    )

    left_side[i] += (
        upper[i] * c_full[i + 1]
        if i < len(c_full) - 1
        else 0
    )

# Похибка
verification_error = (
    left_side - rhs
)

max_verification_error = np.max(
    np.abs(verification_error)
)

print(
    "\nМаксимальна похибка перевірки:"
)

print(
    f"{max_verification_error:.12e}"
)

if max_verification_error < 1e-10:

    print(
        "Результат: розв'язок системи "
        "є правильним у межах "
        "обчислювальної похибки."
    )

else:

    print(
        "Результат: необхідно перевірити "
        "розв'язок системи."
    )


# ============================================================
# 13. Гладкий профіль
# ============================================================

xx = np.linspace(
    distances[0],
    distances[-1],
    1000
)

yy = spline_value(
    distances,
    coefficients,
    xx
)


# ============================================================
# 14. Графік початкових точок та сплайна
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    distances,
    elevations,
    "o",
    label="GPS-точки"
)

plt.plot(
    xx,
    yy,
    label="Кубічний сплайн"
)

plt.xlabel(
    "Кумулятивна відстань, м"
)

plt.ylabel(
    "Висота, м"
)

plt.title(
    "Профіль висоти маршруту Заросляк — Говерла"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 15. Порівняння 10, 15 та 20 вузлів
# ============================================================

node_counts = [
    10,
    15,
    20
]

plt.figure(
    figsize=(12, 6)
)

for count in node_counts:

    indices = np.linspace(
        0,
        len(distances) - 1,
        count,
        dtype=int
    )

    x_nodes = distances[
        indices
    ]

    y_nodes = elevations[
        indices
    ]

    coeff, _ = cubic_spline(
        x_nodes,
        y_nodes
    )

    x_plot = np.linspace(
        x_nodes[0],
        x_nodes[-1],
        1000
    )

    y_plot = spline_value(
        x_nodes,
        coeff,
        x_plot
    )

    plt.plot(
        x_plot,
        y_plot,
        label=f"{count} вузлів"
    )


# Початкові GPS-точки один раз
plt.plot(
    distances,
    elevations,
    "o",
    label="Початкові GPS-точки"
)

plt.xlabel(
    "Кумулятивна відстань, м"
)

plt.ylabel(
    "Висота, м"
)

plt.title(
    "Вплив кількості вузлів "
    "на кубічну сплайн-інтерполяцію"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 16. Оцінка похибки для 10, 15 та 20 вузлів
# ============================================================

print("\n" + "=" * 80)
print("ОЦІНКА ТОЧНОСТІ ДЛЯ РІЗНОЇ КІЛЬКОСТІ ВУЗЛІВ")
print("=" * 80)

print(
    "Вузли | Максимальна похибка (м) | "
    "Середня абсолютна похибка (м)"
)

print("-" * 75)

error_results = {}

for count in node_counts:

    indices = np.linspace(
        0,
        len(distances) - 1,
        count,
        dtype=int
    )

    x_nodes = distances[
        indices
    ]

    y_nodes = elevations[
        indices
    ]

    coeff, _ = cubic_spline(
        x_nodes,
        y_nodes
    )

    y_interpolated = spline_value(
        x_nodes,
        coeff,
        distances
    )

    error = (
        y_interpolated
        - elevations
    )

    max_error = np.max(
        np.abs(error)
    )

    mean_error = np.mean(
        np.abs(error)
    )

    error_results[count] = (
        max_error,
        mean_error
    )

    print(
        f"{count:5d} | "
        f"{max_error:23.6f} | "
        f"{mean_error:29.6f}"
    )


# ============================================================
# 17. Запис оцінки похибки у results.txt
# ============================================================

with open(
    "results.txt",
    "a",
    encoding="utf-8"
) as file:

    file.write(
        "\n\nОЦІНКА ТОЧНОСТІ ДЛЯ 10, 15 ТА 20 ВУЗЛІВ\n"
    )

    file.write("=" * 70 + "\n")

    file.write(
        "Вузли | Максимальна похибка (м) | "
        "Середня абсолютна похибка (м)\n"
    )

    for count in node_counts:

        max_error, mean_error = (
            error_results[count]
        )

        file.write(
            f"{count:5d} | "
            f"{max_error:23.6f} | "
            f"{mean_error:29.6f}\n"
        )


# ============================================================
# 18. Графік похибки для 10, 15 та 20 вузлів
# ============================================================

plt.figure(
    figsize=(12, 6)
)

for count in node_counts:

    indices = np.linspace(
        0,
        len(distances) - 1,
        count,
        dtype=int
    )

    x_nodes = distances[
        indices
    ]

    y_nodes = elevations[
        indices
    ]

    coeff, _ = cubic_spline(
        x_nodes,
        y_nodes
    )

    y_interpolated = spline_value(
        x_nodes,
        coeff,
        distances
    )

    error = (
        y_interpolated
        - elevations
    )

    plt.plot(
        distances,
        error,
        label=f"{count} вузлів"
    )

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Кумулятивна відстань, м"
)

plt.ylabel(
    "Похибка, м"
)

plt.title(
    "Похибка кубічної сплайн-інтерполяції"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 19. Порівняння заданої та наближеної функції
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    distances,
    elevations,
    "o",
    label="Задані значення"
)

plt.plot(
    xx,
    yy,
    label="Наближене значення"
)

plt.xlabel(
    "Кумулятивна відстань, м"
)

plt.ylabel(
    "Висота, м"
)

plt.title(
    "Задана та наближена функція"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 20. Похибка у вузлах
# ============================================================

interpolated_at_nodes = spline_value(
    distances,
    coefficients,
    distances
)

error_at_nodes = (
    interpolated_at_nodes
    - elevations
)

print("\n" + "=" * 80)
print("ПОХИБКА У ВУЗЛАХ")
print("=" * 80)

print(
    "№ | Висота | Наближене значення | Похибка"
)

print("-" * 70)

for i in range(n):

    print(
        f"{i:2d} | "
        f"{elevations[i]:8.3f} | "
        f"{interpolated_at_nodes[i]:18.3f} | "
        f"{error_at_nodes[i]:10.6f}"
    )


# ============================================================
# 21. Графік похибки у вузлах
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    distances,
    error_at_nodes,
    "o-"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Кумулятивна відстань, м"
)

plt.ylabel(
    "Похибка, м"
)

plt.title(
    "Похибка інтерполяції у вузлах"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 22. Загальна довжина маршруту
# ============================================================

total_distance = distances[-1]

print("\n" + "=" * 80)
print("ХАРАКТЕРИСТИКИ МАРШРУТУ")
print("=" * 80)

print(
    f"Загальна довжина маршруту: "
    f"{total_distance:.2f} м"
)

print(
    f"Загальна довжина маршруту: "
    f"{total_distance / 1000:.3f} км"
)


# ============================================================
# 23. Загальний набір висоти
# ============================================================

total_ascent = sum(
    max(
        elevations[i]
        - elevations[i - 1],
        0
    )
    for i in range(1, n)
)

print(
    f"Сумарний набір висоти: "
    f"{total_ascent:.2f} м"
)


# ============================================================
# 24. Загальний спуск
# ============================================================

total_descent = sum(
    max(
        elevations[i - 1]
        - elevations[i],
        0
    )
    for i in range(1, n)
)

print(
    f"Сумарний спуск: "
    f"{total_descent:.2f} м"
)


# ============================================================
# 25. Аналіз градієнта через похідну сплайна
# ============================================================

grad_full = (
    np.gradient(
        yy,
        xx
    )
    * 100
)

max_ascent_gradient = np.max(
    grad_full
)

max_descent_gradient = np.min(
    grad_full
)

mean_gradient = np.mean(
    np.abs(grad_full)
)

print("\n" + "=" * 80)
print("АНАЛІЗ ГРАДІЄНТА")
print("=" * 80)

print(
    f"Максимальний підйом: "
    f"{max_ascent_gradient:.2f} %"
)

print(
    f"Максимальний спуск: "
    f"{max_descent_gradient:.2f} %"
)

print(
    f"Середній градієнт: "
    f"{mean_gradient:.2f} %"
)


# ============================================================
# 26. Ділянки з крутизною понад 15 %
# ============================================================

steep = np.abs(
    grad_full
) > 15

number_of_steep_points = np.sum(
    steep
)

print(
    f"\nКількість точок із "
    f"крутизною понад 15 %: "
    f"{number_of_steep_points}"
)

if np.any(steep):

    print(
        "Виявлено ділянки з крутизною "
        "понад 15 %."
    )

else:

    print(
        "Ділянок з крутизною понад 15 % "
        "не виявлено."
    )


# ============================================================
# 27. Механічна робота для маси 80 кг
# ============================================================

mass = 80
g = 9.81

energy = (
    mass
    * g
    * total_ascent
)

print("\n" + "=" * 80)
print("МЕХАНІЧНА РОБОТА")
print("=" * 80)

print(
    f"Маса: {mass} кг"
)

print(
    f"Прискорення вільного падіння: "
    f"{g} м/с²"
)

print(
    f"Механічна робота: "
    f"{energy:.2f} Дж"
)

print(
    f"Механічна робота: "
    f"{energy / 1000:.2f} кДж"
)

print(
    f"Енергія: "
    f"{energy / 4184:.2f} ккал"
)


# ============================================================
# 28. Запис характеристик маршруту у results.txt
# ============================================================

with open(
    "results.txt",
    "a",
    encoding="utf-8"
) as file:

    file.write(
        "\n\nХАРАКТЕРИСТИКИ МАРШРУТУ\n"
    )

    file.write("=" * 70 + "\n")

    file.write(
        f"Загальна довжина маршруту: "
        f"{total_distance:.2f} м\n"
    )

    file.write(
        f"Загальна довжина маршруту: "
        f"{total_distance / 1000:.3f} км\n"
    )

    file.write(
        f"Сумарний набір висоти: "
        f"{total_ascent:.2f} м\n"
    )

    file.write(
        f"Сумарний спуск: "
        f"{total_descent:.2f} м\n"
    )

    file.write(
        f"Максимальний підйом: "
        f"{max_ascent_gradient:.2f} %\n"
    )

    file.write(
        f"Максимальний спуск: "
        f"{max_descent_gradient:.2f} %\n"
    )

    file.write(
        f"Середній градієнт: "
        f"{mean_gradient:.2f} %\n"
    )

    file.write(
        f"Кількість точок із крутизною понад 15 %: "
        f"{number_of_steep_points}\n"
    )

    file.write(
        f"Механічна робота для 80 кг: "
        f"{energy:.2f} Дж\n"
    )

    file.write(
        f"Механічна робота для 80 кг: "
        f"{energy / 1000:.2f} кДж\n"
    )

    file.write(
        f"Енергія: "
        f"{energy / 4184:.2f} ккал\n"
    )


# ============================================================
# 29. Фінальна інформація
# ============================================================

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТ")
print("=" * 80)

print(
    "Усі основні обчислення лабораторної роботи виконано."
)

print(
    "Результати збережено у файл results.txt."
)

print(
    "\nПрограму виконано успішно."
)

print("=" * 80)