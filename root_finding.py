import math

def newton_raphson(f, df, x0, tol=1e-7, max_iter=100):
    """
    Finds a root of a function using the Newton-Raphson method.

    Args:
        f (callable): The function for which to find a root.
        df (callable): The derivative of the function f.
        x0 (float): The initial guess for the root.
        tol (float): The tolerance for convergence. The method stops when
                     the absolute value of f(x) is less than tol.
        max_iter (int): The maximum number of iterations.

    Returns:
        float: The estimated root of the function.

    Raises:
        ValueError: If the derivative is zero during an iteration or
                    if the method does not converge within max_iter.
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            print(f"Newton-Raphson converged after {i+1} iterations.")
            return x

        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivative is zero. Newton-Raphson method failed.")

        x = x - fx / dfx

    raise ValueError(f"Newton-Raphson did not converge within {max_iter} iterations.")

def secant(f, x0, x1, tol=1e-7, max_iter=100):
    """
    Finds a root of a function using the secant method.

    Args:
        f (callable): The function for which to find a root.
        x0 (float): The first initial guess for the root.
        x1 (float): The second initial guess for the root.
        tol (float): The tolerance for convergence. The method stops when
                     the absolute value of f(x) is less than tol.
        max_iter (int): The maximum number of iterations.

    Returns:
        float: The estimated root of the function.

    Raises:
        ValueError: If the method does not converge within max_iter.
    """
    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)

        if abs(fx1) < tol:
            print(f"Secant method converged after {i+1} iterations.")
            return x1

        if fx1 == fx0:
            raise ValueError("Secant method failed: f(x1) == f(x0).")

        # Secant approximation of the derivative
        df_approx = (fx1 - fx0) / (x1 - x0)

        # Newton-Raphson update with secant approximation
        x_next = x1 - fx1 / df_approx
        x0, x1 = x1, x_next  # Update for next iteration

    raise ValueError(f"Secant method did not converge within {max_iter} iterations.")



def bisection_method(f, a, b, tol=1e-7, max_iter=100):
    """
    Finds a root of a function using the bisection method.

    Args:
        f (callable): The function for which to find a root.
        a (float): The start of the interval.
        b (float): The end of the interval.
        tol (float): The tolerance for convergence. The method stops when
                     the interval size (b - a) is less than tol.
        max_iter (int): The maximum number of iterations.

    Returns:
        float: The estimated root of the function.

    Raises:
        ValueError: If f(a) and f(b) do not have opposite signs, or
                    if the method does not converge within max_iter.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs for the bisection method.")

    for i in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)

        if abs(fc) < 1e-15 or (b - a) / 2.0 < tol:
            print(f"Bisection method converged after {i+1} iterations.")
            return c

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    raise ValueError(f"Bisection method did not converge within {max_iter} iterations.")


if __name__ == '__main__':
    # Example usage: Find the root of f(x) = x^2 - 4
    # The actual roots are -2 and 2.

    # Define the function and its derivative
    def my_function(x):
        return x**2 - 4

    def my_derivative(x):
        return 2*x

    def my_function(x):
        return x**2 - 4

    print("--- Finding root of f(x) = x^2 - 4 ---")

    # --- Newton-Raphson Example ---
    print("\nUsing Newton-Raphson method:")
    try:
        # Start with an initial guess of 1.0
        initial_guess = 1.0
        root_nr = newton_raphson(my_function, my_derivative, initial_guess)
        print(f"Root found at: {root_nr}")
        print(f"f({root_nr}) = {my_function(root_nr)}")
    except ValueError as e:
        print(e)

    # --- Bisection Method Example ---
    print("\nUsing Bisection method:")
    try:
        # Use an interval [0, 5]
        # f(0) = -4, f(5) = 21. They have opposite signs.
        interval_start = 0.0
        interval_end = 5.0
        root_bisection = bisection_method(my_function, interval_start, interval_end)
        print(f"Root found at: {root_bisection}")
        print(f"f({root_bisection}) = {my_function(root_bisection)}")
    except ValueError as e:
        print(e)

    # --- Secant Method Example ---
    print("\nUsing Secant method:")
    try:
        # Need two initial guesses
        initial_guess_1 = 1.0
        initial_guess_2 = 1.5

        root_secant = secant(my_function, initial_guess_1, initial_guess_2)
        print(f"Root found at: {root_secant}")
        print(f"f({root_secant}) = {my_function(root_secant)}")
    except ValueError as e:
        print(e)
