# Running the Code

This guide explains how to run the Python programs in this repository.

## Prerequisites

You'll need Python 3 installed on your computer. If you don't have Python installed, follow the instructions here:
- [Installing Python](https://github.com/baiway/MScFE_python_refresher/blob/main/docs/installing-python.md)

## Setup

1. **Clone this repository** to your computer:
   ```sh
   git clone <add_link>
   ```

2. **Open a terminal** and navigate to the project folder:
   ```sh
   cd maths-IA-support
   ```

3. **Create a virtual environment** (recommended):
   ```sh
   python3 -m venv .venv
   ```

4. **Activate the virtual environment**:
   - On macOS/Linux:
     ```sh
     source .venv/bin/activate
     ```
   - On Windows:
     ```sh
     .venv\Scripts\activate
     ```

5. **Install required packages**:
   ```sh
   pip install numpy scipy matplotlib
   ```

## Running the Programs

### 2D Fit Program
This program performs a bivariate quadratic fit on the seat measurements:
```bash
python 2d-fit.py
```
This will display the fitted parameters and show a 3D visualization of the fit.

### Numerical Integration Program
This program calculates the surface area using numerical integration:
```bash
python numeric-integrator.py
```
This will compute the integral using three different methods and display the results along with a visualisation.

## Troubleshooting

- If you get "command not found" errors, make sure Python is installed correctly
- If you get import errors, make sure you've activated the virtual environment and installed the required packages
- If the plots don't display, you may need to install additional dependencies for matplotlib (this is platform-specific)
