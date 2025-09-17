from decimal import Decimal, getcontext
from typing import List
import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

class HighPrecisionGaussInt:
    """
    High precision Gaussian quadrature integration using decimal arithmetic
    Equivalent to __float128 precision (~34 decimal digits)
    """
    
    def __init__(self, npoints: int, precision: int = 40):
        # Set precision higher than __float128 (~34 digits) for intermediate calculations
        getcontext().prec = precision
        
        self.lroots: List[Decimal] = []
        self.weight: List[Decimal] = []
        self.lcoef: List[List[Decimal]] = []
        self.precision = precision
        
        # High precision constants
        self.PI = Decimal(str(math.pi)).quantize(Decimal(10) ** -(precision-5))
        # More precise pi calculation
        self.PI = self._calculate_pi_high_precision()
        
        self.init(npoints)
    
    def _calculate_pi_high_precision(self) -> Decimal:
        """Calculate pi to high precision using Machin's formula"""
        # pi/4 = 4*arctan(1/5) - arctan(1/239)
        getcontext().prec = self.precision + 10  # Extra precision for intermediate calc
        
        def arctan_series(x: Decimal, terms: int = None) -> Decimal:
            if terms is None:
                terms = self.precision + 5
            
            x_squared = x * x
            result = x
            term = x
            
            for n in range(1, terms):
                term *= -x_squared
                result += term / (2 * n + 1)
            
            return result
        
        one_fifth = Decimal(1) / Decimal(5)
        one_239 = Decimal(1) / Decimal(239)
        
        pi_quarter = 4 * arctan_series(one_fifth) - arctan_series(one_239)
        pi = 4 * pi_quarter
        
        getcontext().prec = self.precision  # Reset precision
        return pi
    
    def _cos_high_precision(self, x: Decimal) -> Decimal:
        """Calculate cosine using Taylor series"""
        # Reduce x to [0, 2*pi)
        two_pi = 2 * self.PI
        x = x % two_pi
        
        # Use Taylor series: cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...
        result = Decimal(1)
        term = Decimal(1)
        x_squared = x * x
        
        for n in range(1, self.precision + 5):
            term *= -x_squared / (Decimal(2*n-1) * Decimal(2*n))
            result += term
            
            # Stop if term becomes negligible
            if abs(term) < Decimal(10) ** -(self.precision + 2):
                break
        
        return result
    
    def _abs_decimal(self, x: Decimal) -> Decimal:
        """Absolute value for Decimal"""
        return x if x >= 0 else -x
    
    def lege_eval(self, n: int, x: Decimal) -> Decimal:
        """Evaluate Legendre polynomial at x using Horner's method"""
        s = self.lcoef[n][n]
        for i in range(n, 0, -1):
            s = s * x + self.lcoef[n][i - 1]
        return s
    
    def lege_diff(self, n: int, x: Decimal) -> Decimal:
        """Evaluate derivative of Legendre polynomial at x"""
        n_dec = Decimal(n)
        return n_dec * (x * self.lege_eval(n, x) - self.lege_eval(n - 1, x)) / (x * x - Decimal(1))
    
    def init(self, npoints: int):
        """
        Calculates abscissas and weights to high precision
        for n-point quadrature rule
        """
        # Initialize arrays
        self.lroots = [Decimal(0)] * npoints
        self.weight = [Decimal(0)] * npoints
        self.lcoef = [[Decimal(0) for _ in range(npoints + 1)] for _ in range(npoints + 1)]
        
        # Initialize Legendre polynomial coefficients
        self.lcoef[0][0] = Decimal(1)
        self.lcoef[1][1] = Decimal(1)
        
        # Generate Legendre polynomial coefficients using recurrence relation
        for n in range(2, npoints + 1):
            n_dec = Decimal(n)
            n_minus_1 = Decimal(n - 1)
            
            self.lcoef[n][0] = -n_minus_1 * self.lcoef[n - 2][0] / n_dec
            
            for i in range(1, n + 1):
                two_n_minus_1 = Decimal(2 * n - 1)
                self.lcoef[n][i] = ((two_n_minus_1 * self.lcoef[n - 1][i - 1] - 
                                   n_minus_1 * self.lcoef[n - 2][i]) / n_dec)
        
        # Find roots using Newton-Raphson method
        eps = Decimal(10) ** -(self.precision - 5)  # High precision tolerance
        
        for i in range(1, npoints + 1):
            # Initial guess using asymptotic formula
            i_dec = Decimal(i)
            npoints_dec = Decimal(npoints)
            
            # x = cos(π * (i - 0.25) / (npoints + 0.5))
            angle = self.PI * (i_dec - Decimal('0.25')) / (npoints_dec + Decimal('0.5'))
            x = self._cos_high_precision(angle)
            
            # Newton-Raphson iteration
            max_iterations = 100
            for iteration in range(max_iterations):
                x1 = x
                
                # Newton-Raphson step: x_new = x - f(x)/f'(x)
                f_val = self.lege_eval(npoints, x)
                f_prime = self.lege_diff(npoints, x)
                
                if f_prime == 0:
                    break
                    
                x = x - f_val / f_prime
                
                # Check convergence
                if self._abs_decimal(x - x1) <= eps:
                    break
            
            # Store root
            self.lroots[i - 1] = x
            
            # Calculate weight
            x1 = self.lege_diff(npoints, x)
            self.weight[i - 1] = Decimal(2) / ((Decimal(1) - x * x) * x1 * x1)

    def PrintWA(self):
        # Print results with high precision
        print(f"==== {len(self.weight)} ====")
        for i in range(len(self.weight)):
            print(f"Weight: {self.weight[i]}")
            print(f"Root:   {self.lroots[i]}")
            print()
    
    def integ(self, f, a: float, b: float) -> Decimal:
        """
        Integrate function f from a to b using Gaussian quadrature
        """
        a_dec = Decimal(str(a))
        b_dec = Decimal(str(b))
        
        c1 = (b_dec - a_dec) / Decimal(2)
        c2 = (b_dec + a_dec) / Decimal(2)
        sum_val = Decimal(0)
        
        for i in range(len(self.weight)):
            # Convert to float for function evaluation, then back to Decimal
            x_eval = float(c1 * self.lroots[i] + c2)
            f_val = Decimal(str(f(x_eval)))
            sum_val += self.weight[i] * f_val
        
        return c1 * sum_val
    
def integ_trap(n, f, a: float, b: float) -> Decimal:
    """Integrate function f from a to b using trapezoid rule"""
    a_dec = Decimal(str(a))
    b_dec = Decimal(str(b))

    h = (b_dec-a_dec)/Decimal(n - 1) # width of trapezoid
    # print(h)
    x_dec = np.array([a_dec+Decimal(i-1)*h for i in range(1, n + 1)])
    f_dec = np.array([Decimal(str(f(float(y)))) for y in x_dec])
    # print(x)

    # set up the alternating weights for simpson's rule
    w = np.array([h for i in range(1, n+1)])
    w[0] = h/Decimal(2)
    w[-1] = h/Decimal(2)

    return np.sum(f_dec * w)
    
def integ_simp(n, f, a: float, b: float) -> Decimal:
    """"Integrate function f from a to b using Simpson's rule"""
    # N must be odd - if I get an even number, add 1 to it instead
    n = oddify_n(n)


    n_interval = n-1 # number of intervals to add up
    a_dec = Decimal(str(a))
    b_dec = Decimal(str(b))

    sum_val = Decimal(0)
    h_dec = (b_dec-a_dec)/Decimal(n_interval) # width of interval
    h = float(h_dec)

    x_dec = np.array([a_dec+Decimal(i-1)*h_dec for i in range(1, n + 1)]) # points at which to take value of function
    f_dec = np.array([Decimal(str(f(float(y)))) for y in x_dec])

    # set up the alternating weights for simpson's rule
    w = np.array([h_dec/Decimal(3/2) for i in range(1, n+1)])
    w[0] = h_dec/Decimal(3)
    w[-1] = h_dec/Decimal(3)
    w[1:-1:2] = h_dec/Decimal(3/4)
    # print(w)

    return np.sum(f_dec*w)

def oddify_n(n):
    """if n is even, return it plus one"""
    if n % 2==0:
        return n+1
    return n
    

def comp_integ(function_name, f, f_integ, a: float, b: float, n_list, g_limit) -> Decimal:
    """"Compare the three integration methods for a function f and choices for number of points n_list"""
    print("Make classes")
    # make list of class objects for doing gauss integral
    n_gauss = [HighPrecisionGaussInt(n, precision=40) for n in n_list if n<g_limit]
    print("Done")

    # initialize arrays to fill with results for each method
    trap_res = np.array([Decimal(0) for i in range(len(n_list))])
    simp_res = np.array([Decimal(0) for i in range(len(n_list))])
    gauss_res = np.array([Decimal(0) for i in range(len(n_gauss))])

    # calculate actual integral value
    act_res = Decimal(str(f_integ(a, b)))

    # print(n_list)

    # go through and do calculations for each n (or up to g_limit for gaussian integral)
    for i in range(len(n_list)):
        # print(f"n={n_list[i]}")
        if n_list[i] < g_limit:
            gauss_res[i] = n_gauss[i].integ(f, a, b)
        trap_res[i] = integ_trap(n_list[i], f, a, b)
        simp_res[i] = integ_simp(n_list[i], f, a, b)
        
    # calculate relative errors for each method
    trap_err = np.abs((trap_res-act_res)/act_res)
    simp_err = np.abs((simp_res-act_res)/act_res)
    gauss_err = np.abs((gauss_res-act_res)/act_res)

    # Because I chose to add 1 to each even n for the simpson method, I need to make an array with the appropriate values to plot against
    simp_n = [oddify_n(n) for n in n_list]

    # have concatenated list of n's for gaussian method
    gauss_n = [n for n in n_list if n<g_limit]

    # make figure
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(12,10)

    # plot all three on the same axis versus h logscale
    axs.plot(n_list, trap_err, label="Trap. Method")
    axs.plot(simp_n, simp_err, label="Simpson's Method")
    axs.plot(gauss_n, gauss_err, label="Gauss. Method")
    axs.legend()
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_ylabel(f"Error {function_name}")
    axs.set_xlabel("N")

    # plt.show()
    plt.savefig(f"Errors.png")


def neg_exp(t):
    return np.exp(-t)

def neg_exp_integ(a, b):
    return np.exp(-a) - np.exp(-b)

def quadr(x: float):
    return 0.9*x**2 + 1.0*x + 28.0

def n_poly(coef: np.ndarray, x: float):
    """Return n polynomial with coefficients coef. Highest order coefficients are at front"""
    n = coef.size - 1 # order of polynomial
    poly_sum = 0.
    for i in range(n+1):
        poly_sum += coef[i]*x**(n-i)
        print(f"coef {i} * x^{n-i}")

    return poly_sum


# Example usage and testing
if __name__ == "__main__":
    import sys
    # # Create high precision integrator
    # if len(sys.argv)==1: order=10
    # else:
    #     order=int(sys.argv[1])
    # print(f"Creating {order}-point Gaussian quadrature with high precision...")
    # gauss_hp = HighPrecisionGaussInt(order, precision=40)
    # gauss_hp.PrintWA()
    
    # print(gauss_hp.integ(neg_exp, 0, 1))
    # print(gauss_hp.integ_trap(neg_exp, 0, 1))
    # print(gauss_hp.integ_simp(neg_exp, 0, 1))

    # # print(n_poly(np.array([2,3,4,1]), 10))
    # print(gauss_hp.integ(quadr, 0, 1))
    # print(gauss_hp.integ_trap(quadr, 0, 1))
    # print(gauss_hp.integ_simp(quadr, 0, 1))
    n_vals = [i for i in range(2,1000,4)]
    comp_integ("$\int e^{-t}$", neg_exp, neg_exp_integ, 0., 1., n_vals, g_limit=60)
    
