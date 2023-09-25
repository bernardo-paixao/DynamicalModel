# Authors : Bernardo Vasconcellos
# Description : This code models the displacement of a ponctual mass supported by a damper an a spring, subjected to a one dof displacement forcing. 

import numpy as np
import matplotlib.pyplot as plt 
 
m = 200 # Vehicule mass [Kg]
k = 5000 # String rate [N/m]
c = 100 # Damper rate [N.s/m]
g = 9.81 # Gravity [m/s**2]

t = np.linspace(0,40,2000)
dt = t[1]-t[0]
sw = 0.5 + 0.5*np.tanh(0.5*(t-20))
sm = np.zeros_like(t)
sm[0] = -m*g/k

dsw = np.zeros_like(t)
for i in range(len(t)):
    if i==0:
        dsw[i] = (sw[i+1]-sw[i])/dt
    elif i<len(t)-1:
        dsw[i] = (sw[i+1]-sw[i-1])/(2*dt)
    else:
        dsw[i] = (sw[i]-sw[i-1])/dt

f = k*sw + c*dsw - m*g

for i in range(len(t)-1):
    if i==0:
        sm[i+1] = -m*g/k #(-k*sm[i]+f[i])/(m/dt**2+c/(2*dt))
    else:
        sm[i+1] = ((2*m/dt**2-k)*sm[i]+(c/(2*dt)-m/dt**2)*sm[i-1]+f[i])/(m/dt**2+c/(2*dt))


fig, ax = plt.subplots()
ax.plot(t,sw,'k')
ax.plot(t,dsw,'b')
ax.plot(t,sm+1,'r')
fig.savefig("test.png")