import numpy as np
import time

class SORMethod:
    '''
    Зазначаємо рівняння загального вигляду, для якого розв'язуємо завдання:
    A(x, y) * d^2(U)/d(x^2) + B(x, y) * d^2(U)/d(y^2) + C(x, y) * U = G(x, y)

    Параметри (як і в BlockThomas, щоб не пропустити):
    ----------
    1) x_params та y_params : dict
       - 'func' - функції A(x, y) або B(x, y)
       - 'interval' - (мінімальне, максимальне) значення для осей
       - 'boundary_cond' - граничні умови для лівого/верхнього та правого/нижнього країв
       - 'grid_points' - кількість точок на сітці
    2) C : callable
          Функція C(x, y)
    3) G : callable
          Функція G(x, y)
    4) internal_conditions : dict
        - 'area' - межі для області внутрішніх умов
        - 'value' - значення U в цій області

    Примітка: коефіцієнт релаксації w використовуватимемо 1.5, за потреби його можна налаштувати пізніше
    
    '''
    def __init__(self, x_params, y_params, C, G, internal_conditions=[]):
        # Ініціалізація меж і кроків сітки
        self.x_min, self.x_max = x_params['interval']
        self.y_min, self.y_max = y_params['interval']
        self.hx = (self.x_max - self.x_min) / (x_params['grid_points'] - 1)
        self.hy = (self.y_max - self.y_min) / (y_params['grid_points'] - 1)
        self.grid_size = (y_params['grid_points'], x_params['grid_points'])
        self.error_history = []

        # Координати вузлів сітки
        self.x_coords = self.x_min + self.hx * np.arange(self.grid_size[1])
        self.y_coords = self.y_min + self.hy * np.arange(self.grid_size[0])
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)

        # Ініціалізація функцій
        self.A_func = x_params['func']
        self.B_func = y_params['func']
        self.C_func = C
        self.G_func = G
        self.internal_conditions = internal_conditions
        
        # Ініціалізація коефіцієнтів і граничних умов
        self._initialize_coefficients()
        self._set_boundary_conditions(x_params['boundary_cond'], y_params['boundary_cond'])
        self._apply_internal_conditions()

    def solve(self, max_iterations, accuracy_threshold=1e-7, w=1.5):
        # Початкова ініціалізація розв'язку
        n, m = self.grid_size
        U = np.random.normal(size=self.grid_size)
        U_next = U.copy()
    
        last_check_time = time.time()
        for iteration in range(max_iterations):
            # Обчислення значень на сітці
            for i in range(n):
                for j in range(m):
                    # Визначення сусідніх значень
                    U_left = U_next[i, j - 1] if j > 0 else 0
                    U_right = U_next[i, j + 1] if j < m - 1 else 0
                    U_top = U_next[i - 1, j] if i > 0 else 0
                    U_bottom = U_next[i + 1, j] if i < n - 1 else 0

                    # Обчислення нового значення U[i, j] з урахуванням коефіцієнтів
                    U_next[i, j] = ((1 - w) * U[i, j] + (w / self.B[i, j]) * (self.D[i, j] - 
                                    self.AX[i, j] * U_left -
                                    self.CX[i, j] * U_right -
                                    self.AY[i, j] * U_top -
                                    self.CY[i, j] * U_bottom))
            
            # Перевірка точності
            delta_U = np.mean(np.abs(U - U_next) / (np.abs(U_next) + 1e-12))
            if delta_U < accuracy_threshold:
                return U
            if time.time() - last_check_time > 0.5:
                self.error_history.append(delta_U)
                last_check_time = time.time()
            U = U_next.copy()

        return U_next

    def _initialize_coefficients(self):
        # Ініціалізація коефіцієнтів для кожного вузла сітки
        self.AX = self.A_func(self.X, self.Y) / self.hx**2
        self.CX = self.A_func(self.X, self.Y) / self.hx**2
        self.AY = self.B_func(self.X, self.Y) / self.hy**2
        self.CY = self.B_func(self.X, self.Y) / self.hy**2
        self.B = self.C_func(self.X, self.Y) - 2 * self.AX - 2 * self.AY
        self.D = self.G_func(self.X, self.Y)
    
    def _set_boundary_conditions(self, x_cond, y_cond):
        # Ініціалізація граничних умов
        self.left_bound = np.array([(x_cond[0][0], x_cond[0][1], x_cond[0][2](y)) for y in self.y_coords]).reshape(-1, 3)
        self.right_bound = np.array([(x_cond[1][0], x_cond[1][1], x_cond[1][2](y)) for y in self.y_coords]).reshape(-1, 3)
        self.top_bound = np.array([(y_cond[0][0], y_cond[0][1], y_cond[0][2](x)) for x in self.x_coords]).reshape(-1, 3)
        self.bottom_bound = np.array([(y_cond[1][0], y_cond[1][1], y_cond[1][2](x)) for x in self.x_coords]).reshape(-1, 3)

        n, m = self.grid_size

        # Ліва межа
        a_left0, a_left1, f_left = self.left_bound[:, 0], self.left_bound[:, 1], self.left_bound[:, 2]
        self.AX[:, 0] = np.zeros(n)
        self.B[:, 0] = a_left0 - a_left1 / self.hx
        self.CX[:, 0] = a_left1 / self.hx
        self.D[:, 0] = f_left

        self.AY[:, 0] = np.zeros(n)
        self.CY[:, 0] = np.zeros(n)

        # Права межа
        b_right0, b_right1, f_right = self.right_bound[:, 0], self.right_bound[:, 1], self.right_bound[:, 2]
        self.AX[:, m - 1] = -b_right1 / self.hx
        self.B[:, m - 1] = b_right0 + b_right1 / self.hx
        self.CX[:, m - 1] = np.zeros(n)
        self.D[:, m - 1] = f_right

        self.AY[:, m - 1] = np.zeros(n)
        self.CY[:, m - 1] = np.zeros(n)

        # Верхня межа
        a_top0, a_top1, f_top = self.top_bound[:, 0], self.top_bound[:, 1], self.top_bound[:, 2]
        self.AY[0, :] = np.zeros(m)
        self.B[0, :] = a_top0 - a_top1 / self.hy
        self.CY[0, :] = a_top1 / self.hy
        self.D[0, :] = f_top

        self.AX[0, :] = np.zeros(m)
        self.CX[0, :] = np.zeros(m)

        # Нижня межа
        b_bottom0, b_bottom1, f_bottom = self.bottom_bound[:, 0], self.bottom_bound[:, 1], self.bottom_bound[:, 2]
        self.AY[n - 1, :] = -b_bottom1 / self.hy
        self.B[n - 1, :] = b_bottom0 + b_bottom1 / self.hy
        self.CY[n - 1, :] = np.zeros(m)
        self.D[n - 1, :] = f_bottom

        self.AX[n - 1, :] = np.zeros(m)
        self.CX[n - 1, :] = np.zeros(m)

    def _apply_internal_conditions(self):
        # Застосування внутрішніх умов
        for condition in self.internal_conditions:
            (x1, x2), (y1, y2) = condition['area']
            mask = ((self.X >= x1) & (self.X <= x2) & (self.Y >= y1) & (self.Y <= y2))

            self.AX[mask] = 0
            self.CX[mask] = 0
            self.AY[mask] = 0
            self.CY[mask] = 0

            self.B[mask] = 1
            self.D[mask] = condition['value']