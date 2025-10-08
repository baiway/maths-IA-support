import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import argparse

def fit(xy: tuple[float, float], a0: float, a1: float, a2: float, a3: float,
    a4: float, a5: float) -> float:
    """Defines a bivariate quadratic fit function
      z = f(x, y) = a0 + a1*x + a2*y + a3*x**2 + a4*x*y + a5*y**2
    """
    x, y = xy
    return a0 + a1*x + a2*y + a3*x**2 + a4*x*y + a5*y**2

parser = argparse.ArgumentParser(description="Fits a bivariate quadratic.")
parser.add_argument(dest="filename", type=str,
        help="Path to `.csv` file containing (x,y,z) data."
)
args = parser.parse_args()
filename = args.filename

x, y, z = np.loadtxt(filename, dtype=np.float64, delimiter=",", unpack=True)
popt, pcov = curve_fit(fit, (x, y), z)

# Unpack optimised parameters (and their errors) from fit
a0, a1, a2, a3, a4, a5 = popt
a0err, a1err, a2err, a3err, a4err, a5err = np.sqrt(np.diag(pcov))

print(f"{a0 = } ± {a0err}")
print(f"{a1 = } ± {a1err}")
print(f"{a2 = } ± {a2err}")
print(f"{a3 = } ± {a3err}")
print(f"{a4 = } ± {a4err}")
print(f"{a5 = } ± {a5err}")
print("-" * 60)
print("\nFit function is therefore:")
print(f" z = f(x, y) = {a0:.6f} + {a1:.6f}x + {a2:.6f}y + {a3:.6f}x^2" +
      f" + {a4:.6f}xy + {a5:.6f}y^2")

# Plot smooth fit using more points and parameters from `curve_fit`
xpoints = np.linspace(min(x), max(x), num=1000)
ypoints = np.linspace(min(y), max(y), num=1000)
X, Y = np.meshgrid(xpoints, ypoints)
Z = fit((X, Y), *popt)

# Create figure with two subplots
fig = plt.figure(figsize=(12, 5))

# 3D surface plot
ax1 = fig.add_subplot(111, projection="3d")
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, label="Fit")
ax1.scatter(x, y, z, color='red', label="Measurements")
ax1.set_xlabel('x / cm')
ax1.set_ylabel('y / cm')
ax1.set_zlabel('z / cm')

# Set aspect ratio based on data ranges for equal scaling
x_range = max(x) - min(x)
y_range = max(y) - min(y)
z_range = np.max(Z) - np.min(Z)
ax1.set_box_aspect([x_range, y_range, z_range])

plt.legend()
plt.tight_layout()
plt.show()
