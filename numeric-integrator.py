import numpy as np
from scipy import integrate


def integrand(x: float, y: float, a1: float, a2: float, a3: float,
    a4: float, a5: float) -> float:
    dzdx = a1 + 2*a3*x + a4*y
    dzdy = a2 + a4*x + 2*a5*y
    return np.sqrt(1 + dzdx**2 + dzdy**2)


def trapezium_rule(x_min: float, x_max: float, y_min: float, y_max: float,
    a1: float, a2: float, a3: float, a4: float, a5: float,
    nx: int, ny: int) -> float:
    """Performs double integration using the trapezium rule."""
    # Create grid
    x = np.linspace(x_min, x_max, nx + 1)
    y = np.linspace(y_min, y_max, ny + 1)
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    # Compute sum with trapezoidal weights
    total = 0
    for i in range(nx + 1):
        for j in range(ny + 1):
            # Trapezoidal weights
            weight = 1.0
            if i == 0 or i == nx:
                weight *= 0.5
            if j == 0 or j == ny:
                weight *= 0.5

            total += weight * integrand(x[i], y[j], a1, a2, a3, a4, a5)
    return total * dx * dy


def scipy_integration(x_min: float, x_max: float, y_min: float, y_max: float,
    a1: float, a2: float, a3: float, a4: float, a5: float) -> tuple[float, float]:
    """Performs double integration using `scipy.dblquad`, which uses
    adaptive quadrature for improved accuracy (slower)."""
    result, error = integrate.dblquad(
        lambda y, x: integrand(x, y, a1, a2, a3, a4, a5),  # dblquad expects y first
        x_min, x_max,  # x limits
        y_min, y_max   # y limits (can be functions of x in general)
    )
    return result, error


def vectorised_trap(x_min: float, x_max: float, y_min: float, y_max: float,
    a1: float, a2: float, a3: float, a4: float, a5: float,
    nx: int, ny: int):
    """Vectorises the trapezium rule calculation"""
    x = np.linspace(x_min, x_max, nx + 1)
    y = np.linspace(y_min, y_max, ny + 1)
    X, Y = np.meshgrid(x, y)

    # Vectorized computation of the integrand
    dZdX = a1 + 2*a3*X + a4*Y
    dZdY = a2 + a4*X + 2*a5*Y
    Z = np.sqrt(1 + dZdX**2 + dZdY**2)

    # Trapezoidal integration using numpy
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    # Apply trapezoidal weights at boundaries
    Z[0, :] *= 0.5
    Z[-1, :] *= 0.5
    Z[:, 0] *= 0.5
    Z[:, -1] *= 0.5

    return np.sum(Z) * dx * dy

print("Enter coefficients of fit function: ")
print("  z = f(x, y) = a0 + a1*x + a2*y + a3*x^2 + a4*xy + a5*y^2")
a0 = float(input("  a0 = "))
a1 = float(input("  a1 = "))
a2 = float(input("  a2 = "))
a3 = float(input("  a3 = "))
a4 = float(input("  a4 = "))
a5 = float(input("  a5 = "))
print("\nYou've supplied the following:")
print(f"  z = f(x, y) = {a0:.6f} + {a1:.6f}x + {a2:.6f}y + {a3:.6f}x^2" +
      f" + {a4:.6f}xy + {a5:.6f}y^2")
print("\nThis gives the following partial derivatives:")
print(f"  dz/dx = {a1:.6f} + {2*a3:.6f}x + {a4:.6f}y")
print(f"  dz/dy = {a2:.6f} + {a4:.6f}x + {2*a5:.6f}y")
print("Enter integration limits: ")
x_min = float(input("  xlim = "))
x_max = float(input("  xmax = "))
y_min = float(input("  y_min = "))
y_max = float(input("  y_max = "))

# Define integration bounds
x_min, x_max = -10, 10
y_min, y_max = -8, 8

print("\nComputing the integral:")
print(f"A = ∫∫ sqrt(1 + (dz/dx)² + (dz/dy)²) dx dy")
print(f"Over region: [{x_min}, {x_max}] × [{y_min}, {y_max}]")
print("-" * 60)

# Method 1: Trapezoidal rule with different grid sizes
print("\nMethod 1: Trapezoidal Rule")
result1 = trapezium_rule(x_min, x_max, y_min, y_max, a1, a2, a3, a4, a5, 200, 150)
print(f"  Grid 201×151: A = {result1:.6f}")

result2 = trapezium_rule(x_min, x_max, y_min, y_max, a1, a2, a3, a4, a5, 400, 300)
print(f"  Grid 401×301: A = {result2:.6f}")

print(f"  Difference: {abs(result2 - result1):.6f}")
print(f"  Relative difference: {abs(result2 - result1)/result2 * 100:.6f}%")

# Method 2: Vectorised computation (same algorithm, faster implementation)
print("\nMethod 2: Vectorised Trapezoidal")
result3 = vectorised_trap(x_min, x_max, y_min, y_max, a1, a2, a3, a4, a5, 400, 300)
print(f"  Grid 401×301: A = {result3:.6f}")

# Method 3: SciPy's adaptive quadrature (most accurate but slower)
print("\nMethod 3: SciPy's dblquad (adaptive quadrature)")
print("  (This may take a moment...)")
result4, error = scipy_integration(x_min, x_max, y_min, y_max, a1, a2, a3, a4, a5)
print(f"  A = {result4:.6f} ± {error:.2e}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY:")
print(f"  The integral A ≈ {result2:.2f}")
print(f"  Rectangle area (40 × 30) = {40*30}")
print(f"  Ratio A/Rectangle = {result2/1200:.4f}")

# Optional: Visualize the integrand
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create a mesh for visualization
    x = np.linspace(x_min, x_max, 50)
    y = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x, y)

    Z = np.sqrt(1 + (-0.01*X + 0.001*Y)**2 + (-0.016*Y + 0.001*X)**2)

    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Integrand')
    ax1.set_title('3D Surface of Integrand')
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour Plot of Integrand')
    fig.colorbar(contour, ax=ax2)

    plt.tight_layout()
    plt.show()

    print("\nVisualization plotted successfully!")

except ImportError:
    print("\n(Matplotlib not available - skipping visualization)")
