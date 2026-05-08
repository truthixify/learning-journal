import numpy as np
from sympy import symbols, Matrix, sympify

def string_to_augmented_matrix(equations):
    # Split the input string into individual equations
    equation_list = equations.split('\n')
    equation_list = [x for x in equation_list if x != '']
    # Create a list to store the coefficients and constants
    coefficients = []
    
    ss = ''
    for c in equations:
        if c in 'abcdefghijklmnopqrstuvwxyz':
            if c not in ss:
                ss += c + ' '
    ss = ss[:-1]
    
    # Create symbols for variables x, y, z, etc.
    variables = symbols(ss)
    # Parse each equation and extract coefficients and constants
    for equation in equation_list:
        # Remove spaces and split into left and right sides
        sides = equation.replace(' ', '').split('=')
        
        # Parse the left side using SymPy's parser
        left_side = sympify(sides[0])
        
        # Extract coefficients for variables
        coefficients.append([left_side.coeff(variable) for variable in variables])
        
        # Append the constant term
        coefficients[-1].append(int(sides[1]))

    # Create a matrix from the coefficients
    augmented_matrix = Matrix(coefficients)
    augmented_matrix = np.array(augmented_matrix).astype("float64")

    A, B = augmented_matrix[:,:-1], augmented_matrix[:,-1].reshape(-1,1)
    
    return ss, A, B

def gaussian_elimination(A, B):
    """
    Solve the linear system Ax = B using Gaussian elimination
    with partial pivoting, then back substitution.
    """
    A = A.astype("float64").copy()
    B = B.astype("float64").copy()
    n = len(A)

    # Check if the system has a unique solution
    if np.isclose(np.linalg.det(A), 0):
        return "Singular system"

    # Build the augmented matrix [A | B]
    M = np.hstack((A, B))

    # --- Forward elimination: turn M into row echelon form ---
    for col in range(n):
        # If the current pivot is zero, swap with a row below that has a non-zero entry
        if np.isclose(M[col, col], 0):
            for search_row in range(col + 1, n):
                if not np.isclose(M[search_row, col], 0):
                    M[[col, search_row]] = M[[search_row, col]]
                    break

        # Scale the pivot row so the pivot becomes 1
        pivot = M[col, col]
        M[col] = M[col] / pivot

        # Eliminate all entries below the pivot
        for row_below in range(col + 1, n):
            factor = M[row_below, col]
            M[row_below] = M[row_below] - factor * M[col]

    # --- Back substitution: turn row echelon form into reduced row echelon form ---
    # At this point M looks like:
    #   [ 1  a  b | d ]
    #   [ 0  1  c | e ]
    #   [ 0  0  1 | f ]
    # We work from the bottom row upward, using each pivot row
    # to zero out all entries above its pivot.
    for pivot_row in reversed(range(n)):
        # The pivot column is the first non-zero column in this row.
        # Since the matrix is in row echelon form with 1s on the diagonal,
        # the pivot column equals the row index.
        pivot_col = pivot_row

        # Zero out every entry above the pivot in this column
        for row_above in range(pivot_row):
            factor = M[row_above, pivot_col]
            M[row_above] = M[row_above] - factor * M[pivot_row]

    # The last column now holds the solution
    return M[:, -1]

equations = """
3*x + 6*y + 6*w + 8*z = 1
5*x + 3*y + 6*w = -10
4*y - 5*w + 8*z = 8
4*w + 8*z = 9
"""

variables, A, B = string_to_augmented_matrix(equations)
solution = gaussian_elimination(A, B)

if isinstance(solution, str):
    print(solution)
else:
    for var, val in zip(variables.split(" "), solution):
        print(f"{var} = {val:.4f}")
