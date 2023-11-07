import numpy as np
from scipy.integrate import quad
# quad can calculate for the numerical integration
# use like this quad(func,a,d)  func is the function, a is the lower input / d is the upper input

# define the number h
h = 0.75

# define conditional probability function
def Pr_1(x):
    if x < 0:
        return h*x + h
    else:
        return -h*x + h

def Pr_0(x):
    return 1 - Pr_1(x)

# define the density function
# because x has uniform distrubiton on [-1,1] so the density is 1/(1-(-1)) = 0.5
def p_uniform(a,b):
    return 1 / (b-a)

def integrand(x):
    return min(Pr_1(x), Pr_0(x)) * p_uniform(-1,1)

bayes_error, error = quad(integrand, -1, 1)

print("Bayes Error = ", bayes_error)
