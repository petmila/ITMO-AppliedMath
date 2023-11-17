import numpy as np
import json


class Linear_task:
    def __init__(self):
        self.func: np.array(int)
        self.goal: str
        self.constraints: list[dict()]
        self.constraints_matrix: np.array()
        self.b_array: np.array()
        self.original_size: int

        self.simplex_table = dict()
        self.optimal: np.array()
        self.basis: np.array()
        self.func_b = 0.0
    
    # считывание условий задачи из файла json
    def load_data_from_json(self, filename: str) -> None:
        with open(filename) as f:
            data = json.load(f)
        self.func = np.array(data[' f '].copy()).astype(np.float32)
        self.original_size = len(self.func)
        self.goal = data[' goal ']
        self.constraints = data[' constraints ']
        self.constraints_matrix = np.array([line[' coefs '] for line in self.constraints]).astype(np.float32)
        self.b_array = np.array([line[' b '] for line in self.constraints]).astype(np.float32)
        self.make_canonic_form()

    # переход к канонической форме задачи - добавление балансирующих переменных
    def make_canonic_form(self) -> None:
        for index, line in enumerate(self.constraints):
            if line[' type '] != ' eq ':
                if line[' type '] == ' gte ':
                    for item in self.constraints_matrix[index]:
                        item = -item
                self.constraints_matrix = np.append(self.constraints_matrix, 
                        [[1] if i == index else [0] for i in range(len(self.constraints_matrix))], axis= 1)
                self.func = np.append(self.func, 0)
        
    def simplex_method(self):
        self.set_start_values()

        while True:
            column = self.choose_solving_column()
            # остановка алгоритма
            if column == -1:
                break
            try:
                row, new_value = self.choose_solving_row(column)
            except ValueError:
                return "Система ограничений несовместна"
            self.recount_matrix(column, row)
            self.recount_optimal(column, new_value, row)
            self.recount_simplex_table(column, row)
            self.change_func_coefs(column, row)
            print("Значение функции:    ", np.dot(self.optimal, self.func.T) + self.func_b)

        return self.optimal[: self.original_size]

    def change_func_coefs(self, index_new_basis, index_old_basis):

        for i in range(len(self.constraints_matrix)):
            if self.constraints_matrix[i][index_new_basis] != 0:
                self.func_b += self.func[index_new_basis]*self.b_array[i]
                for j, value in enumerate(self.constraints_matrix[i]):
                    if j != index_new_basis:
                        self.func[j] += -self.func[index_new_basis]*value
                        
        self.func[index_new_basis] = 0
            
    def recount_simplex_table(self, index_new_basis, index_old_basis):
        self.simplex_table[index_new_basis] = dict.fromkeys(list(self.simplex_table[index_old_basis].keys()))
        del self.simplex_table[index_old_basis]
        
        for key, value in self.simplex_table.items():
            del value[index_new_basis]
            for i in range(len(self.constraints_matrix)):
                if self.constraints_matrix[i][key] != 0:
                    value[index_old_basis] = self.constraints_matrix[i][index_old_basis]/self.constraints_matrix[i][key]
        
        for key_ in self.simplex_table.keys():
            for key, value in self.simplex_table[key_].items():
                if key == index_old_basis:
                    continue
                for i in range(len(self.constraints_matrix)):
                    if self.constraints_matrix[i][key_] == 0:
                        continue
                    self.simplex_table[key_][key] = self.constraints_matrix[i][key]/self.constraints_matrix[i][key_]
    
    def recount_matrix(self, old_index, new_index):
        matrix = np.copy(self.constraints_matrix)
        basis_indices = list(self.simplex_table.keys())
        basis_indices.remove(new_index)
        b = np.copy(self.b_array)
        line_number = -1
        for row in range(len(self.constraints_matrix)):
            flag_right_line = True
            for k in basis_indices:
                if matrix[row][k] != 0:
                    flag_right_line = False
            if matrix[row][old_index] == 0:
                continue
            elif flag_right_line:
                line_number = row
                break

        for row in range(len(self.constraints_matrix)):
            if matrix[row][old_index] == 0:
                continue
            elif row == line_number:
                f = matrix[row][old_index]
                for j in range(len(self.constraints_matrix[row])):
                    matrix[row][j] /= f
                b[row] /= f
            else:
                f = matrix[row][old_index] / matrix[line_number][old_index]
                for j in range(len(self.constraints_matrix[row])):
                    matrix[row][j] -= f * matrix[line_number][j]
                b[row] -= f * b[line_number]
            
        self.constraints_matrix = matrix
        self.b_array = b

    def recount_optimal(self, index_new_basis, new_value, index_old_basis):
        for i in self.simplex_table.keys():
            for row in range(len(self.constraints_matrix)):
                if self.constraints_matrix[row][i] != 0:
                    self.optimal[i] = self.b_array[row]/self.constraints_matrix[row][i]

        self.optimal[index_new_basis] = new_value
        self.optimal[index_old_basis] = 0

    def set_start_values(self):
        self.optimal = [0 for i in range(len(self.func))]
        # определение базисных векторов
        self.simplex_table = dict.fromkeys([i for i in range(self.original_size, len(self.func))])

        for k in range(len(self.constraints_matrix)):
            for i in range(self.original_size):
                if self.constraints_matrix[k][i] == 0:
                    continue
                flag_basis = True
                for j in self.simplex_table.keys():
                    if self.constraints_matrix[k][j] != 0:
                        flag_basis = False
            if flag_basis:
                self.simplex_table[i] = None
        nonbasis = [i for i in range(len(self.func)) if i not in self.simplex_table]

        for i in range(len(self.func)):
            if i in self.simplex_table:
                for k in range(len(self.constraints_matrix)):
                    if self.constraints_matrix[k][i] == 0:
                        continue
                    row = {j : self.constraints_matrix[k][j]/self.constraints_matrix[k][i] for j in nonbasis}
                    self.simplex_table[i] = row.copy()
                    self.optimal[i] = self.b_array[k]
            else:
                self.optimal[i] = 0

        self.optimal = np.array(self.optimal)
        self.optimal_func = np.dot(self.optimal, self.func.T)

    def choose_solving_row(self, column):
        dict_ = {key: abs(self.optimal[key]/value[column]) for key, value in self.simplex_table.items()
                  if value[column] != 0 and self.optimal[key]/value[column] > 0}
        if len(list(dict_.values())) == 0:
            raise ValueError
        min_value = min(list(dict_.values()))
        row = list(dict_.keys())[list(dict_.values()).index(min_value)]

        return row, min_value

    def choose_solving_column(self):
        max_index = -1
        min_index = -1
        for i in list(self.simplex_table[list(self.simplex_table.keys())[0]].keys()):
            if self.goal == " max ":
                if self.func[i] >= self.func[max_index] and self.func[i] >= 0:
                    max_index = i
            elif self.goal == " min ":
                if self.func[i] < self.func[min_index] and self.func[i] <= 0:
                    min_index = i
        
        if self.goal == " max ":
            return max_index
        elif self.goal == " min ":
            return min_index