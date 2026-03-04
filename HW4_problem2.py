"""
MAE 3403 – HW4 – Problem 2
Find intersection of:

Circle:
    (x - x1)^2 + (y - y1)^2 = R^2

Parabola:
    y = a*x^2 + b
"""
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def equations(x, params):
    """
    Function whose root corresponds to the intersection point.
    We eliminate y by using the parabola equation y = a*x^2 + b.
    Then plug into the circle equation:
      (y - y1)^2 + (x - x1)^2 - R^2 = 0
    """
    x1, y1, R, a, b = params
# y from parabola
    y = a * x**2 + b
# circle equation with y substituted
    return (y - y1)**2 + (x - x1)**2 - R**2

def find_intersections(x1, y1, R, a, b, guesses):
    """
    Use multiple initial guesses to find (numerically) distinct
    intersection points between the circle and parabola.
    """
    params = (x1, y1, R, a, b)
    roots = []

    for g in guesses:
        sol = fsolve(equations, g, args=(params,))
        x_root = sol[0]

# Compute y from parabola
        y_root = a * x_root**2 + b

# Check if this root is new (not already found)
        if not roots:
            roots.append((x_root, y_root))
        else:
            if all(np.hypot(x_root - xr, y_root - yr) > 1e-3
                   for xr, yr in roots):
                roots.append((x_root, y_root))

    return roots

def main():
    print("\nIntersection of Circle and Parabola\n")

# Default parameters (test case)
    x1_default = 1.0
    y1_default = 0.0
    R_default  = 4.0          # since equation is (y - y1)^2 + (x - x1)^2 = 16
    a_default  = 0.5          # width of parabola
    b_default  = 1.0          # vertical offset of parabola

# User input (with defaults)
    x1_in = input(f"Circle center x1? (default {x1_default}): ")
    x1 = x1_default if x1_in.strip() == "" else float(x1_in)

    y1_in = input(f"Circle center y1? (default {y1_default}): ")
    y1 = y1_default if y1_in.strip() == "" else float(y1_in)

    R_in = input(f"Circle radius R? (default {R_default}): ")
    R = R_default if R_in.strip() == "" else float(R_in)

    a_in = input(f"Parabola width a in y = a x^2 + b? (default {a_default}): ")
    a = a_default if a_in.strip() == "" else float(a_in)

    b_in = input(f"Parabola offset b in y = a x^2 + b? (default {b_default}): ")
    b = b_default if b_in.strip() == "" else float(b_in)

# Find intersections using several initial guesses
    guesses = [-8, -4, -1, 0, 1, 4, 8]
    intersections = find_intersections(x1, y1, R, a, b, guesses)

    if intersections:
        print("\nIntersection point(s):")
        for i, (xi, yi) in enumerate(intersections, start=1):
            print(f"  Point {i}: x = {xi:.4f}, y = {yi:.4f}")
    else:
        print("\nNo real intersection points found.")

# Prepare plot range
    x_vals = np.linspace(-10, 10, 400)

# Circle: parametric representation for plotting
    theta = np.linspace(0, 2*np.pi, 400)
    x_circle = x1 + R * np.cos(theta)
    y_circle = y1 + R * np.sin(theta)

# Parabola
    y_parab = a * x_vals**2 + b

# Plot
    plt.figure(figsize=(6, 6))
    plt.plot(x_circle, y_circle, 'b-', label='Circle')
    plt.plot(x_vals, y_parab, 'r-', label='Parabola')

# Plot intersection points, if any
    if intersections:
        xs = [p[0] for p in intersections]
        ys = [p[1] for p in intersections]
        plt.plot(xs, ys, 'ko', label='Intersection(s)')

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Circle and Parabola Intersection')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', 'box')

    plt.show()

if __name__ == "__main__":
    main()
