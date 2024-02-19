import math

def erci(n, m):
    return math.factorial(n) // (math.factorial(m) * math.factorial(n - m))

def calculaterh(l, theta):
    if l % 2 != 0:
        k = l // 2 + 1
    else:
        k = l // 2

    rh = 0
    for i in range(k, l + 1):
         rh += erci(l, i) * (theta**i) * ((1 - theta)**(l - i))
    return rh

l = 15
theta = 0.35
result = calculaterh(l,theta)
print(f"risk: {result:.4f}")