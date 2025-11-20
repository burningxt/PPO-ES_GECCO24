import cocoex as ex
import numpy as np

# Create a BBOB suite, restricting to function 1 and dimension 40
# suite = ex.Suite("bbob", "", "dimensions: 40 function_indices:1-24 instance_indices: 1")
suite = ex.Suite("bbob", "", "dimensions:40 function_indices:1-24 instance_indices:1")

print(len(suite))

# Get the single problem in this filtered suite
# this will return problem 0
problem = suite.get_problem(0)
print(type(problem))

# x = problem.initial_solution
# f
# print("feasible solution: ", x)

# Print the final target f-value
# print("Final target f-value:", problem.final_target_fvalue1)
# value = problem.final_target_fvalue1

# x_opt = problem._best_parameter()

# print(x)

# f_opt = problem(x)

# print(f_opt)

problem._best_parameter('print')
x_opt = np.loadtxt('._bbob_problem_best_parameter.txt')
f_opt = problem(x_opt)
print(f_opt)