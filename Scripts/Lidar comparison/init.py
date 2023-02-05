from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import calendar
import scipy.stats as st
from matplotlib.gridspec import GridSpec
import seaborn as sns



###### VARIABLES ######

Save_files = "Yes" # Choose Yes to save graphs in the Results folder, No to just print them in output cells
site = "TRE" # Change to calculate for either site
filter = "3h" # Time smoothing
polynomial_alpha = "constant" # Set monthly to have a polynomial fit for each month, constant to have a global polynomial fit
poly_deg = 2 # Polynomial fit degree

########################



# Relative access to upper folders (since the doc is stored on onedrive)
path = Path(os.getcwd())
maindir = path.parents[1]

if site == "NOY":
    cmap="Greens"
if site == "TRE":
    cmap = "Blues"
heights = [55,70,80,95,120,130,150,170,200] #hard-coded (not the most elegant, but effective)
N_Sectors = 12 # We could use more sectors (if we had more shear coefficients)
edges = np.linspace(360/N_Sectors/2,360-360/N_Sectors/2,N_Sectors)
z0 = 4


def linear_reg_alpha(U, ur, Z, zr):
    Y = [np.log(u/ur) for u in U]
    X = [np.log(z/zr) for z in Z]
    coefficients = np.polyfit(X, Y, 1)
    return coefficients[0]


def alpha(u, ur, Z, zr):
    a = np.log(u/ur)
    b = np.log(z/zr)
    return a/b


# Power law function to use later in vector operation
def powerlaw(v0,h0,h1,alpha):
    v1 = v0*(h1/h0)**alpha
    return v1


def prettyprintPolynomial(p):
    """ Small function to print nicely the polynomial p as we write it in maths, in ASCII text."""
    coefs = p.coef  # List of coefficient, sorted by increasing degrees
    res = ""  # The resulting string
    for i, a in enumerate(coefs):
        if int(a) == a:  # Remove the trailing .0
            a = int(a)
        if i == 0:  # First coefficient, no need for X
            if a > 0:
                res += "{a} + ".format(a=a)
            elif a < 0:  # Negative a is printed like (a)
                res += "({a}) + ".format(a=a)
            # a = 0 is not displayed 
        elif i == 1:  # Second coefficient, only X and not X**i
            if a == 1:  # a = 1 does not need to be displayed
                res += "\u0394T + "
            elif a > 0:
                res += "{a} * \u0394T + ".format(a=a)
            elif a < 0:
                res += "({a}) * \u0394T + ".format(a=a)
        else:
            if a == 1:
                res += "\u0394T**{i} + ".format(i=i)
            elif a > 0:
                res += "{a} * \u0394T**{i} + ".format(a=a, i=i)
            elif a < 0:
                res += "({a}) * \u0394T**{i} + ".format(a=a, i=i)
    return res[:-3] if res else ""