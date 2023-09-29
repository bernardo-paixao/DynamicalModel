from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def ode(t,x):

    a1 = 3e5
    a2 = 0.2
    a3 = 4e-7
    a4 = 0.6
    a5 = 8
    a6 = 90

    A = np.array([x[0],x[1],x[2]])
    B = np.array([x[3],x[4],x[5]])
    C = np.array([x[6],x[7],x[8]])

    dAdt = a1 - a2*A-a3*A*C
    dBdt = a3*A*C - a4*B
    dCdt = a3*A*C - a5*C + a6*B

    return np.concatenate((dAdt, dBdt, dCdt))

x0 = [2e6, 2e6, 1e6, 0, 0, 0, 90, 80, 70]

t = np.linspace(0,15,1000)
sol = solve_ivp(ode,[t[0],t[-1]],x0,method='RK45',t_eval=t)
var = sol.y
print(np.shape(var))
plt.semilogy(t, var.T)
plt.xlabel('t')
plt.legend(['A', 'B','C'], shadow=True)
plt.title('Lotka-Volterra System')
plt.show()