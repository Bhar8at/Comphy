import numpy as np

def newton_cotes_integration(f, a, b, N, method="trapezoidal"):
    """
    Performs numerical integration using various composite Newton-Cotes methods.


    Args:
        f (callable): The function to integrate.
        a (float): The lower limit of integration.
        b (float): The upper limit of integration.
        N (int): The total number of subintervals. Must be compatible with the
                 chosen method (e.g., even for Simpson's 1/3).
        method (str): The Newton-Cotes method to use. Options are:
                      'trapezoidal', 'simpson13', 'simpson38', 'boole'.
                      

    Returns:
        float: The approximate value of the definite integral.

    Raises:
        ValueError: If an invalid method is chosen or if N is not compatible
                    with the selected method.
    """
    if a >= b:
        raise ValueError("The lower limit 'a' must be less than the upper limit 'b'.")
    if N <= 0:
        raise ValueError("Number of subintervals 'N' must be positive.")

    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)

    weights = {
            'trapezoidal': (1, [1], h/2),
            'simpson13':   (2, [1, 4], h/3),
            'simpson38':   (3, [1, 3, 3], 3 * h / 8),
            'boole':       (4, [7, 32, 12, 32], 2 * h / 45)
        }

    if method not in weights:
            raise ValueError(f"Method '{method}' is not recognized. Choose from {list(weights.keys())}.")

    n = weights[method][0]
    if N % n != 0:
            raise ValueError(f"For method '{method}', N must be a multiple of {n}.")

    integral = y[0] + y[-1]
    w = weights[method][1]
        
    for i in range(0, N+1):
            m = i % n
            integral += w[m] * y[i]

        # Final scaling factor
    integral *= weights[method][2]

    return integral

if __name__ == '__main__':
    # --- Example Usage ---
    # Let's integrate f(x) = x^4 - 2x + 1 from a=0 to b=2.
    # The exact analytical solution is [x^5/5 - x^2 + x] from 0 to 2
    # = (32/5 - 4 + 2) - 0 = 6.4 - 2 = 4.4

    def my_func(x):
        return x**4 - 2*x + 1

    a, b = 0, 2
    exact_value = 4.4
    print(f"Integrating f(x) = x^4 - 2x + 1 from {a} to {b}")
    print(f"Exact analytical solution: {exact_value}\n")


    # Using Trapezoidal rule
    N_trap = 200
    result_trap = newton_cotes_integration(my_func, a, b, N_trap, method='trapezoidal')
    print(f"Trapezoidal (N={N_trap}):     {result_trap:.8f}, Error: {abs(result_trap - exact_value):.2e}")

    # Using Simpson's 1/3 rule (default)
    # N must be even. Let's use a reasonably large N.
    N_simpson = 200
    result_simpson = newton_cotes_integration(my_func, a, b, N_simpson, method='simpson13')
    print(f"Simpson's 1/3 (N={N_simpson}): {result_simpson:.8f}, Error: {abs(result_simpson - exact_value):.2e}")

     # Using Simpson's 3/8 rule
    # N must be a multiple of 3.
    N_38 = 198
    result_38 = newton_cotes_integration(my_func, a, b, N_38, method='simpson38')
    print(f"Simpson's 3/8 (N={N_38}):   {result_38:.8f}, Error: {abs(result_38 - exact_value):.2e}")

    # Using Boole's rule
    # N must be a multiple of 4.
    N_boole = 200
    result_boole = newton_cotes_integration(my_func, a, b, N_boole, method='boole')
    print(f"Boole's Rule (N={N_boole}):      {result_boole:.8f}, Error: {abs(result_boole - exact_value):.2e}")


