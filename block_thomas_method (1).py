import numpy as np

class BlockThomas:
    '''
    Запишемо рівняння загального виду, для якого запрограмуємо розв'язок:
    A(x, y) * d^2(U)/d(x^2) + B(x, y) * d^2(U)/d(y^2) + C(x, y) * U = G(x, y)

    Параметри:
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
    '''
    def __init__(self, x_params, y_params, C, G, internal_conditions=[]):
        # Ініціалізація меж сітки та кроків
        self.x_min, self.x_max = x_params['interval']
        self.y_min, self.y_max = y_params['interval']
        self.hx = (self.x_max - self.x_min) / (x_params['grid_points'] - 1)
        self.hy = (self.y_max - self.y_min) / (y_params['grid_points'] - 1)
        self.lattice_size = (y_params['grid_points'], x_params['grid_points'])

        # Координати вузлів сітки
        n_rows, n_cols = self.lattice_size
        self.x = self.x_min + self.hx * np.arange(n_cols)
        self.y = self.y_min + self.hy * np.arange(n_rows)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Ініціалізація функцій
        self.A_func = x_params['func']
        self.B_func = y_params['func']
        self.C_func = C
        self.G_func = G
        self.internal_conditions = internal_conditions
        
        # Виклик методів ініціалізації
        self._initialize_coefficients()
        self._apply_boundary_conditions(x_params['boundary_cond'], y_params['boundary_cond'])
        self._apply_internal_conditions()
        self._initialize_block_coefficients()

    def solve(self):
        # Обчислення вперед і назад
        forward_matrices, solution_vectors = self._forward_sweep()
        result = self._backward_sweep(forward_matrices, solution_vectors)
        return result.squeeze()

    def _forward_sweep(self):
        # Прямий хід
        n, m = self.lattice_size
        forward_matrices = np.zeros((n, m, m))
        solution_vectors = np.zeros((n, m, 1))

        # Ініціалізація першого кроку
        forward_matrices[0] = -np.linalg.inv(self.Bb[0]) @ self.Cb[0]
        solution_vectors[0] = np.linalg.inv(self.Bb[0]) @ self.Db[0]
        for i in range(1, n):
            # Використання методу прогонки для обчислення коефіцієнтів
            forward_matrices[i] = -np.linalg.inv(self.Bb[i] + self.Ab[i] @ forward_matrices[i - 1]) @ self.Cb[i]
            solution_vectors[i] = (np.linalg.inv(self.Bb[i] + self.Ab[i] @ forward_matrices[i - 1]) @
                                   (self.Db[i] - self.Ab[i] @ solution_vectors[i - 1]))
        
        return forward_matrices, solution_vectors

    def _backward_sweep(self, forward_matrices, solution_vectors):
        # Зворотний хід
        n, m = self.lattice_size
        result = np.zeros((n, m, 1))

        # Ініціалізація кінцевого кроку
        result[n - 1] = solution_vectors[n - 1].copy()
        for i in range(n - 2, -1, -1):
            # Використання коефіцієнтів для зворотної прогонки
            result[i] = forward_matrices[i] @ result[i + 1] + solution_vectors[i]
        
        return result

    def _initialize_coefficients(self):
        # Ініціалізація коефіцієнтів сітки
        self.AX = self.A_func(self.X, self.Y) / self.hx**2
        self.CX = self.A_func(self.X, self.Y) / self.hx**2
        self.AY = self.B_func(self.X, self.Y) / self.hy**2
        self.CY = self.B_func(self.X, self.Y) / self.hy**2
        self.B = self.C_func(self.X, self.Y) - 2 * self.AX - 2 * self.AY
        self.D = self.G_func(self.X, self.Y)
    
    def _apply_boundary_conditions(self, x_boundary_cond, y_boundary_cond):
        # Застосування граничних умов
        self.left_cond = np.array([(x_boundary_cond[0][0], x_boundary_cond[0][1], x_boundary_cond[0][2](y)) for y in self.y]).reshape(-1, 3)
        self.right_cond = np.array([(x_boundary_cond[1][0], x_boundary_cond[1][1], x_boundary_cond[1][2](y)) for y in self.y]).reshape(-1, 3)
        self.top_cond = np.array([(y_boundary_cond[0][0], y_boundary_cond[0][1], y_boundary_cond[0][2](x)) for x in self.x]).reshape(-1, 3)
        self.bottom_cond = np.array([(y_boundary_cond[1][0], y_boundary_cond[1][1], y_boundary_cond[1][2](x)) for x in self.x]).reshape(-1, 3)

        n, m = self.lattice_size 

        # Обчислення граничних умов для лівої межі
        a_left0, a_left1, f_left = self.left_cond[:, 0], self.left_cond[:, 1], self.left_cond[:, 2]
        self.AX[:, 0] = np.zeros(n)
        self.B[:, 0] = a_left0 - a_left1 / self.hx
        self.CX[:, 0] = a_left1 / self.hx
        self.D[:, 0] = f_left

        self.AY[:, 0] = np.zeros(n)
        self.CY[:, 0] = np.zeros(n)

        # Обчислення граничних умов для правої межі
        b_right0, b_right1, f_right = self.right_cond[:, 0], self.right_cond[:, 1], self.right_cond[:, 2]
        self.AX[:, m - 1] = -b_right1 / self.hx
        self.B[:, m - 1] = b_right0 + b_right1 / self.hx
        self.CX[:, m - 1] = np.zeros(n)
        self.D[:, m - 1] = f_right

        self.AY[:, m - 1] = np.zeros(n)
        self.CY[:, m - 1] = np.zeros(n)

        # Обчислення граничних умов для верхньої межі
        a_top0, a_top1, f_top = self.top_cond[:, 0], self.top_cond[:, 1], self.top_cond[:, 2]
        self.AY[0, :] = np.zeros(m)
        self.B[0, :] = a_top0 - a_top1 / self.hy
        self.CY[0, :] = a_top1 / self.hy
        self.D[0, :] = f_top

        self.AX[0, :] = np.zeros(n)
        self.CX[0, :] = np.zeros(n)

        # Обчислення граничних умов для нижньої межі
        b_bottom0, b_bottom1, f_bottom = self.bottom_cond[:, 0], self.bottom_cond[:, 1], self.bottom_cond[:, 2]
        self.AY[n - 1, :] = -b_bottom1 / self.hy
        self.B[n - 1, :] = b_bottom0 + b_bottom1 / self.hy
        self.CY[n - 1, :] = np.zeros(m)
        self.D[n - 1, :] = f_bottom

        self.AX[n - 1, :] = np.zeros(n)
        self.CX[n - 1, :] = np.zeros(n)

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

    def _initialize_block_coefficients(self):
        # Ініціалізація блочних коефіцієнтів
        n, m = self.lattice_size
        self.Ab = np.zeros((n, m, m))
        self.Bb = np.zeros((n, m, m))
        self.Cb = np.zeros((n, m, m))
        self.Db = np.zeros((n, m, 1))

        for i in range(n):
            self.Ab[i] = np.diag(self.AY[i, :], k=0)
            self.Cb[i] = np.diag(self.CY[i, :], k=0)
            self.Bb[i] = (np.diag(self.AX[i, 1:], k=-1) + 
                          np.diag(self.B[i, :], k=0) + 
                          np.diag(self.CX[i, :-1], k=1))
            self.Db[i] = self.D[i].reshape(-1, 1)