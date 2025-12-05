import numpy as np

def jacobi(A, b, x0=None, tol=1e-8, max_iter=100, verbose=False):
    """
    Solves the system of linear equations Ax = b using the Jacobi method.

    Args:
        A (numpy.ndarray): The NxN coefficient matrix.
        b (numpy.ndarray): The N-element constant vector.
        x0 (numpy.ndarray, optional): The initial guess for the solution vector x.
                                      If None, it defaults to a zero vector.
        tol (float): The tolerance for convergence. The method stops when the
                     L-infinity norm of the difference between successive
                     iterations is less than tol.
        max_iter (int): The maximum number of iterations.
        verbose (bool): If True, prints the iteration number and current solution.

    Returns:
        numpy.ndarray: The estimated solution vector x.

    Raises:
        ValueError: If the method does not converge within max_iter.
    """
    n = len(b)
    A = np.asarray(A)
    b = np.asarray(b)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.asarray(x0, dtype=float)

    x_new = np.zeros_like(x)

    for k in range(max_iter):
        for i in range(n):
            # sum(A[i,j] * x[j] for j != i)
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        if verbose:
            print(f"Jacobi Iteration {k+1}: x = {x_new}")

        # Check for convergence using L-infinity norm
        if np.linalg.norm(x_new - x, np.inf) < tol:
            print(f"Jacobi method converged after {k+1} iterations.")
            return x_new

        x = x_new.copy()

    raise ValueError(f"Jacobi method did not converge within {max_iter} iterations.")

def gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=100, verbose=False):
    """
    Solves the system of linear equations Ax = b using the Gauss-Seidel method.

    Args:
        A (numpy.ndarray): The NxN coefficient matrix.
        b (numpy.ndarray): The N-element constant vector.
        x0 (numpy.ndarray, optional): The initial guess for the solution vector x.
                                      If None, it defaults to a zero vector.
        tol (float): The tolerance for convergence. The method stops when the
                     L-infinity norm of the difference between successive
                     iterations is less than tol.
        max_iter (int): The maximum number of iterations.
        verbose (bool): If True, prints the iteration number and current solution.

    Returns:
        numpy.ndarray: The estimated solution vector x.

    Raises:
        ValueError: If the method does not converge within max_iter.
    """
    n = len(b)
    A = np.asarray(A)
    b = np.asarray(b)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.asarray(x0, dtype=float)

    for k in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            # sum(A[i,j] * x[j] for j != i)
            # Uses updated x values from the current iteration
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]

        if verbose:
            print(f"Gauss-Seidel Iteration {k+1}: x = {x}")

        # Check for convergence using L-infinity norm
        if np.linalg.norm(x - x_old, np.inf) < tol:
            print(f"Gauss-Seidel method converged after {k+1} iterations.")
            return x

    raise ValueError(f"Gauss-Seidel method did not converge within {max_iter} iterations.")

if __name__ == '__main__':
    # Example: Solve a system of linear equations
    # 4x + y - z = 13
    # 3x + 5y + z = 20
    # x + y + 3z = 10
    # The exact solution is x=3, y=2, z=1

    A_matrix = np.array([[4, 1, -1],
                         [3, 5, 1],
                         [1, 1, 3]], dtype=float)

    b_vector = np.array([13, 20, 10], dtype=float)

    print("--- Solving Ax = b using iterative methods ---")
    print("Matrix A:\n", A_matrix)
    print("Vector b:\n", b_vector)

    # --- Jacobi Method Example ---
    print("\nUsing Jacobi method:")
    try:
        solution_jacobi = jacobi(A_matrix, b_vector)
        print("Solution:", solution_jacobi)
        residue = np.dot(A_matrix, solution_jacobi) - b_vector
        print("Max residue:", np.max(np.abs(residue)))
    except ValueError as e:
        print(e)

    # --- Gauss-Seidel Method Example ---
    print("\nUsing Gauss-Seidel method:")
    try:
        solution_gs = gauss_seidel(A_matrix, b_vector)
        print("Solution:", solution_gs)
        residue = np.dot(A_matrix, solution_gs) - b_vector
        print("Max residue:", np.max(np.abs(residue)))
    except ValueError as e:
        print(e)