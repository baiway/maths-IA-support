import numpy as np
from scipy import integrate


def integrand(x: float, y: float) -> float:
    """Computes sqrt(1 + (-0.01x + 0.001y)^2 + (-0.016y + 0.001x)^2)"""
    return np.sqrt(1 + (-0.01*x + 0.001*y)**2 + (-0.016*y + 0.001*x)**2)


def trapezium_rule(x_min: float, x_max: float, y_min: float, y_max: float,
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

            total += weight * integrand(x[i], y[j])
    return total * dx * dy


def scipy_integration(x_min: float, x_max: float, y_min: float, y_max: float):
    """Performs double integration using `scipy.dblquad`, which uses
    adaptive quadrature for improved accuracy (slower)."""
    result, error = integrate.dblquad(
        lambda y, x: integrand(x, y),  # dblquad expects y first
        x_min, x_max,  # x limits
        y_min, y_max   # y limits (can be functions of x in general)
    )
    return result, error


def vectorised_trap(x_min: float, x_max: float, y_min: float, y_max: float,
    nx: int, ny: int):
    """Vectorises the trapezium rule calculation"""
    x = np.linspace(x_min, x_max, nx + 1)
    y = np.linspace(y_min, y_max, ny + 1)
    X, Y = np.meshgrid(x, y)

    # Vectorized computation of the integrand
    Z = np.sqrt(1 + (-0.01*X + 0.001*Y)**2 + (-0.016*Y + 0.001*X)**2)

    # Trapezoidal integration using numpy
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    # Apply trapezoidal weights at boundaries
    Z[0, :] *= 0.5
    Z[-1, :] *= 0.5
    Z[:, 0] *= 0.5
    Z[:, -1] *= 0.5

    return np.sum(Z) * dx * dy


# Define integration bounds
x_min, x_max = -20, 20
y_min, y_max = -15, 15

print("Computing the integral:")
print(f"A = ∫∫ sqrt(1 + (-0.01x + 0.001y)² + (-0.016y + 0.001x)²) dx dy")
print(f"Over region: [{x_min}, {x_max}] × [{y_min}, {y_max}]")
print("-" * 60)

# Method 1: Trapezoidal rule with different grid sizes
print("\nMethod 1: Trapezoidal Rule")
result1 = trapezium_rule(x_min, x_max, y_min, y_max, 200, 150)
print(f"  Grid 201×151: A = {result1:.6f}")

result2 = trapezium_rule(x_min, x_max, y_min, y_max, 400, 300)
print(f"  Grid 401×301: A = {result2:.6f}")

print(f"  Difference: {abs(result2 - result1):.6f}")
print(f"  Relative difference: {abs(result2 - result1)/result2 * 100:.6f}%")

# Method 2: Vectorised computation (same algorithm, faster implementation)
print("\nMethod 2: Vectorised Trapezoidal")
result3 = vectorised_trap(x_min, x_max, y_min, y_max, 400, 300)
print(f"  Grid 401×301: A = {result3:.6f}")

# Method 3: SciPy's adaptive quadrature (most accurate but slower)
print("\nMethod 3: SciPy's dblquad (adaptive quadrature)")
print("  (This may take a moment...)")
result4, error = scipy_integration(x_min, x_max, y_min, y_max)
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
