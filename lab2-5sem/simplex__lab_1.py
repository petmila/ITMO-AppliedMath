import json
import numpy as np
import pandas as pd
import warnings
import time
import random
warnings.filterwarnings("ignore")

m = 0
initial_gaol = ''
initial_coefs = []

class LinearTask:
  def load_data_from_json(self, filename: str):
    with open(filename) as f:
      input_data = json.load(f)
    func = Function(input_data[' f '])
    goal = input_data[' goal ']
    constraints = [Constraint(i[' coefs '], i[' b '], i[' type ']) for i in input_data[' constraints ']]
    self.task = LinearProgrammingTask(func, goal, constraints)

  def solve(self):
    task = self.task
    if not check_degeneracy(task):
      print('No solutions!')
    else:
      global initial_gaol, initial_coefs, m
      initial_gaol = task.goal
      initial_coefs = task.func.coefs
      if task.goal == 'max':
        task.func.coefs = [x * -1 for x in task.func.coefs]
        task.goal = 'min'
      m = len(task.constraints)
      basis = []
      to_canonical_form(task)
      constraints = make_extended_matrix(task)
      basis = find_basis_vectors(constraints)
      initial_basis_vectors_amount = len(basis)
      constraints = Jordan_Gauss_method(constraints)
      move_basis_vectors_back(constraints, initial_basis_vectors_amount)
      if check_no_solutions(constraints):
        print('No solutions!')
      else:
        return get_solution(constraints, task)

class Function:
  def __init__(self, coefs):
    self.coefs = coefs

class Constraint:
  def __init__(self, coefs, b, sign):
    self.coefs = coefs
    self.b = b
    self.sign = sign

class LinearProgrammingTask:
  def __init__(self, func, goal, constraints):
    self.func = func
    self.goal = goal
    self.constraints = constraints


"""**Приведение к канонической форме**"""

def add_new_var_to_constraints(task: LinearProgrammingTask, constraint_index: int, val: int):
  for i in range(len(task.constraints)):
    result_val = val if i == constraint_index else 0
    task.constraints[i].coefs.append(result_val)

def to_canonical_form(task: LinearProgrammingTask):
  task_constraints_count = len(task.constraints)
  for i in range(task_constraints_count):
    sign = task.constraints[i].sign
    if sign == 'eq':
      continue
    val = 1 if sign == 'lte' else -1
    add_new_var_to_constraints(task, i, val)
    if task.constraints[i].b < 0:
        task.constraints[i].coefs *= -1
        task.constraints[i].b *= -1

"""**Функции для обработки строк и столбцов расширенной матрицы**"""

def find_non_zero_element_index_in_column(column):
  for i, val in enumerate(column):
    if val != 0:
      return i
  return -1

def handle_column(constraints, row, column, ind):
  r1 = constraints[row].copy()
  r2 = constraints[row + ind].copy()
  constraints[row] = r2
  constraints[row + ind] = r1
  constraints[row] = constraints[row] / constraints[row][column]
  for i in range(row + 1, m):
    constraints[i] -= constraints[row] * constraints[i][column]

def delete_zero_rows(constraints):
  zero_rows_to_delete = []
  list_constraints = []
  for constraint in constraints:
    list_constraints.append(list(constraint))
    if find_non_zero_element_index_in_column(constraint) == -1:
      zero_rows_to_delete.append(list(constraint))
  for row in zero_rows_to_delete:
    list_constraints.remove(row)
  return np.array(list_constraints)

"""**Обратный и прямой ход метода Гаусса**"""

def reversed_Gauss_move(constraints):
  n = len(constraints[0])
  m = len(constraints)
  row = m - 1
  column = m - 1
  iteration = 0
  while iteration != m - 1:
    cur_column = column
    while constraints[row][cur_column] == 0:
      cur_column += 1
    constraints[row] = constraints[row] / constraints[row][cur_column]
    for i in range(0, row):
      constraints[i] -= constraints[row] * constraints[i][cur_column]
    row -= 1
    column -= 1
    iteration += 1
  return delete_zero_rows(constraints)

def straight_Gauss_move(constraints):
  n = len(constraints[0])
  m = len(constraints)
  iteration = 0
  row = 0
  column = 0
  while iteration != m - 1:
    ind = find_non_zero_element_index_in_column(constraints[row:, column])
    cur_column = column
    if ind == -1:
      while(ind == -1 and cur_column < n - 1):
        cur_column += 1
        ind = find_non_zero_element_index_in_column(constraints[row:, cur_column])
      if cur_column == n - 1 and ind == -1:
        constraints = constraints[:(row + 1), :]
        break
    handle_column(constraints, row, cur_column, ind)
    row += 1
    column += 1
    iteration += 1
  return delete_zero_rows(constraints)

def Jordan_Gauss_method(constraints_matrix):
  constraints = constraints_matrix
  constraints = straight_Gauss_move(constraints)
  constraints = reversed_Gauss_move(constraints)
  return constraints

"""**Проверки**"""

def check_for_basis_vector(vector):
  ones_count = 0
  zero_count = 0
  for el in vector:
    if el == 0:
      zero_count += 1
    elif el == 1 or el == -1:
      ones_count += 1
  return ones_count == 1 and zero_count == len(vector) - 1

def check_no_solutions(constraints):
  n = len(constraints[0])
  for constraint in constraints:
    if np.all(constraint[:n - 1] == 0) and constraint[n - 1] != 0:
      return True
  return False

"""**Преобразование матрицы в расширенную**"""

def move_basis_vectors_in_front(extended_matrix):
  first_basis_vector_index = len(extended_matrix[0]) - 2
  while first_basis_vector_index >= 0:
    if check_for_basis_vector(extended_matrix[:, first_basis_vector_index]):
      first_basis_vector_index -= 1
    else:
      break
  first_basis_vector_index += 1
  basis_vectors = extended_matrix[:, first_basis_vector_index:len(extended_matrix[0]) - 1].copy()
  non_basis_vectors = extended_matrix[:, :first_basis_vector_index].copy()
  extended_matrix[:, :len(extended_matrix[0]) - first_basis_vector_index - 1] = basis_vectors
  extended_matrix[:, len(extended_matrix[0]) - first_basis_vector_index - 1 : len(extended_matrix[0]) - 1] = non_basis_vectors

def move_basis_vectors_back(extended_matrix, count):
  initial_basis_vectors = extended_matrix[:, :count].copy()
  rest_vectors_amount = len(extended_matrix[0]) - 1 - count
  rest_vectors = extended_matrix[:, count: count + rest_vectors_amount].copy()
  extended_matrix[:, :rest_vectors_amount] = rest_vectors
  extended_matrix[:, rest_vectors_amount:len(extended_matrix[0]) - 1] = initial_basis_vectors

def make_extended_matrix(task: LinearProgrammingTask):
  m = len(task.constraints)
  constraints = []
  for constraint in task.constraints:
    extended_row = list(constraint.coefs)
    extended_row.append(constraint.b)
    extended_row = np.array(extended_row)
    constraints.append(extended_row)
  extended_matrix = np.float_(np.array(constraints))
  move_basis_vectors_in_front(extended_matrix)
  return extended_matrix

"""**Поиск в системе базисных векторов**"""

def find_basis_vectors(constraints):
  basis = {}
  n = len(constraints[0])
  for i in range(n):
    vector = constraints[:, i]
    if check_for_basis_vector(vector):
      row = find_non_zero_element_index_in_column(vector)
      basis[row] = i
    if len(basis) == m:
      break
  return basis

"""**Создание симплекс таблицы**"""

def create_simplex_table(constraints, basis, task):
  m = len(constraints)
  n = len(constraints[0]) - 1
  basis_nums = [0] * len(basis)
  for key in basis.keys():
    basis_nums[key] = f'x{basis[key] + 1}'
  basis_nums += ['f']
  initial_coefs = task.func.coefs + [0 for i in range(n - len(task.func.coefs))]
  b = constraints[:, n]
  table = {'Basis': basis_nums}
  for i in range(n):
    coefs = list(constraints[:, i]) + [-initial_coefs[i]]
    table[f'x{i + 1}'] = coefs
  table['b'] = list(b) + [0]
  table = pd.DataFrame(table)
  return table

"""**Симплекс метод**"""

def get_non_basis_vars(basis_vars, all_vars):
  non_basis_vars = []
  for var in all_vars:
    if var not in basis_vars:
      non_basis_vars.append(var)
  return non_basis_vars

def check_simplex_method_stop(table, n, m):
  basis_vars = table['Basis'][:m].to_list()
  basis_coefs = [table[basis_var][m] for basis_var in basis_vars]
  return all(x <= 0 for x in table.iloc[m, 1:n].to_list()) and all(coef == 0 for coef in basis_coefs)

def get_leading_row(table, n, m, new_basis_var):
  coefs = table[new_basis_var][:m].to_list()
  b = table['b'].to_list()
  devision_list = []
  size = len(coefs)
  for i in range(size):
    if coefs[i] == 0:
      devision_list.append(float('inf'))
    else:
      devision_list.append(b[i] / coefs[i])
  positive_devision_list = [x for x in devision_list if x >= 0]
  if len(positive_devision_list) == 0 or all(x == float('inf') for x in positive_devision_list):
    return -1
  min_positive_val = min([x for x in devision_list if x > 0])
  if min_positive_val != float('inf'):
    return devision_list.index(min_positive_val)
  return devision_list.index(0)

def check_constraints(table, task, m, n):
  vars = table.columns.values[1:n]
  basis = table['Basis'][:m].to_list()
  b = table['b'][:m].to_list()
  initial_vars_count = len(task.func.coefs)

  var_vals = {}
  for i in range(initial_vars_count):
    if vars[i] in basis:
      var_vals[vars[i]] = b[basis.index(vars[i])]
      if b[basis.index(vars[i])] < 0:
        return False
    else:
      var_vals[vars[i]] = 0
  for constraint in task.constraints:
    left_sum = (np.array(list(var_vals.values())) * np.array(constraint.coefs)[:initial_vars_count]).sum()
    if constraint.sign == 'lte' and left_sum <= constraint.b:
      continue
    elif constraint.sign == 'gte' and left_sum >= constraint.b:
      continue
    elif constraint.sign == 'eq' and left_sum == constraint.b:
      continue
    else:
      return False
  return True

def find_leading_column_in_functional_row_in_non_positive_vals_case(functional_row_variables, vars, table, n, m):
  negative_vals = functional_row_variables.copy()
  iteration = 1
  while True:
    if iteration >= 6 * n:
      return 'none', -1
    index = random.randint(0, len(negative_vals) - 1)
    if negative_vals[index] == 0:
      continue
    new_basis_var = vars[index]
    leading_row = get_leading_row(table, n, m, new_basis_var)
    if leading_row == -1:
      continue
    else:
      return new_basis_var, leading_row
    iteration += 1

def get_solution(constraints, task):
  m = len(constraints)
  n = len(constraints[0])
  basis = find_basis_vectors(constraints)
  table = create_simplex_table(constraints, basis, task)
  have_solution = simplex_method(table, m, n, task)
  if have_solution:
    basis_vars = table['Basis'][:m].to_list()

    b = table['b'][:m].to_list()
    for i in range(len(b) - 1):
      if b[i] < 0:
        b[i] *= -1

    vars_values = []
    for i in range(len(initial_coefs)):
      if f'x{i + 1}' in basis_vars:
        vars_values.append((f'x{i + 1}', b[basis_vars.index(f'x{i + 1}')]))
      else:
        vars_values.append((f'x{i + 1}', 0))

    f: float = -table["b"][m] if initial_gaol != task.goal else table["b"][m]
    vars = []
    for var in vars_values:
      vars.append(var[1])
    return f, vars
  else:
    print('The minimum value of function is -inf')

def simplex_method(table, m, n, task):
  while True:
    # print(table)
    time.sleep(0.5)
    vars = table.columns.values[1:n + 1]
    if check_simplex_method_stop(table, n, m) and check_constraints(table, task, m, n):
        break
    functional_row_variables = table.iloc[m, 1:n].to_list()
    max_elemnt_index = functional_row_variables.index(max(functional_row_variables))
    if functional_row_variables[max_elemnt_index] <= 2:
      new_basis_var, leading_row = find_leading_column_in_functional_row_in_non_positive_vals_case(functional_row_variables, vars, table, n, m)
    else:
      new_basis_var = vars[max_elemnt_index]
      leading_row = get_leading_row(table, n, m, new_basis_var)

    if new_basis_var == 'none' and leading_row == -1:
      break
    if leading_row == -1:
      return False
    leading_element = table[new_basis_var][leading_row]
    table['Basis'][leading_row] = new_basis_var
    table_part_to_change = table.iloc[:m+1, 1:n + 2].to_numpy()
    table_part_to_change[leading_row] = table_part_to_change[leading_row] * (1 / leading_element)
    for i in range(len(table_part_to_change)):
      if i != leading_row:
        coef_in_leading_column = table[new_basis_var][i]
        table_part_to_change[i] -= coef_in_leading_column * table_part_to_change[leading_row]
    for i in range(len(vars)):
      table[vars[i]] = table_part_to_change[:, i]
  return True

def check_all_zeroes_case(constraint, sign):
  all_zeroes = True
  size = len(constraint)
  if size == 0:
    return True
  left_part = constraint[:size - 1]
  right_part = constraint[size - 1]
  for el in left_part:
    if el != 0:
      all_zeroes = False
      break
  if all_zeroes:
    if sign == 'lte' and right_part < 0 or \
       sign == 'gte' and right_part > 0 or \
       sign == 'eq' and right_part != 0:
      return False
  return True

def check_constraints_correctness(constraint_rp1, constraint_rp2):
  sign1 = constraint_rp1[0]
  sign2 = constraint_rp2[0]
  val1 = constraint_rp1[1]
  val2 = constraint_rp2[1]
  if val1 > val2:
    return sign1 == 'lte' and (sign2 == 'gte' or sign2 == 'eq') or \
           sign1 == 'eq' and (sign2 == 'gte')
  elif val1 < val2:
    return sign2 == 'lte' and (sign1 == 'gte' or sign1 == 'eq') or \
           sign2 == 'eq' and (sign1 == 'gte')
  return True

def check_linear_dependency(constraint1, sign1, constraint2, sign2):
  linear_multiplier = float('-inf')
  size = len(constraint1)
  c1_left_part = constraint1[:size - 1]
  c2_left_part = constraint2[:size - 1]
  for i in range(size - 1):
    if c1_left_part[i] == 0 and c2_left_part[i] != 0 or \
       c1_left_part[i] != 0 and c2_left_part[i] == 0:
       linear_multiplier = float('-inf')
       break
    elif c1_left_part[i] != 0 and c1_left_part[i] != 0:
      linear_multiplier = c1_left_part[i] / c2_left_part[i]
  if linear_multiplier == float('-inf') and (np.any(c1_left_part) and np.any(c2_left_part)):
    return True
  for i in range(size - 1):
    el1 = c1_left_part[i]
    el2 = c2_left_part[i]
    if el1 != 0 and el2 != 0 and el1 / el2 != linear_multiplier:
      return True
  c1_right_part = constraint1[size - 1]
  c2_right_part = constraint2[size - 1] * linear_multiplier
  return check_constraints_correctness((sign1, c1_right_part), (sign2, c2_right_part))

def check_linear_dependency_for_all_constraints(all_constraints):
  constraints_pairs = [(0, 1), (1, 2), (0, 2)]
  for c_p in constraints_pairs:
    constraints1 = all_constraints[c_p[0]][0]
    constraints2 = all_constraints[c_p[1]][0]
    sign1 = all_constraints[c_p[0]][1]
    sign2 = all_constraints[c_p[1]][1]
    for c1 in constraints1:
      for c2 in constraints2:
        res = check_linear_dependency(c1, sign1, c2, sign2)
        if res == False:
          return False
  return True

def check_degeneracy(task):
  size = len(task.constraints[0].coefs)
  coefs_lte = []
  coefs_gte = []
  coefs_eq = []

  for constraint in task.constraints:
    c = np.zeros(size + 1)
    c[:size] = constraint.coefs
    b = np.zeros(size + 1)
    b[size] = constraint.b
    if constraint.sign == 'lte':
      coefs_lte.append(c + b)
    elif constraint.sign == 'gte':
      coefs_gte.append(c + b)
    elif constraint.sign == 'eq':
      coefs_eq.append(c + b)
  all_constraints = [(coefs_lte, 'lte'), (coefs_gte, 'gte'), (coefs_eq, 'eq')]
  for c in all_constraints:
    for coefs in c[0]:
      if len(coefs) == 0:
        continue
      if check_all_zeroes_case(coefs, c[1]) == False:
        return False
  return check_linear_dependency_for_all_constraints(all_constraints)