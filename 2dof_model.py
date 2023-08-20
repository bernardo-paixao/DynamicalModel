import numpy as np
import matplotlib.pyplot as plt 
 
m = 200 # Vehicule mass [Kg]
L = 1.5 # Distance between axes [m]
hcg = 1 # CG height [m]
fwr = 0.5 # Front weight ratio 
kr = 4000 # Rear string rate [N/m]
kf = 5000 # Front string rate [N/m]
cr = 200 # Rear damper rate [N.s/m]
cf = 200 # Front damper rate [N.s/m]
g = 9.81 # Gravity [m/s**2]

I = 1/12*m*L**2 # Vehicule approx moment of inertia [kg.m**2]
dr = L*(1-fwr)
df = L*fwr

t = np.linspace(0,50,2000)
dt = t[1]-t[0]
swf = (0.5 + 0.5*np.tanh(0.5*(t-20)))*0.2
swr = (0.5 + 0.5*np.tanh(0.5*(t-30)))*0.2
sm = np.zeros_like(t)
beta = np.zeros_like(t)
sm[0] = -m*g/(kr+kf)

dswf = np.zeros_like(t)
dswr = np.zeros_like(t)
for i in range(len(t)):
    if i==0:
        dswf[i] = (swf[i+1]-swf[i])/dt
        dswf[i] = (swr[i+1]-swr[i])/dt
    elif i<len(t)-1:
        dswf[i] = (swf[i+1]-swf[i-1])/(2*dt)
        dswr[i] = (swr[i+1]-swr[i-1])/(2*dt)
    else:
        dswf[i] = (swf[i]-swf[i-1])/dt
        dswr[i] = (swr[i]-swr[i-1])/dt

f = kf*swf + cf*dswf + kr*swr + cr*dswr - m*g
h = kf*df*swf + cf*df*dswf - kr*dr*swr - cr*dr*dswr

for i in range(len(t)-1):
    if i==0:
        sm[i+1] = -m*g/(kr+kf) #(-k*sm[i]+f[i])/(m/dt**2+c/(2*dt))
    else:
        A = np.array([[m/dt**2+(cf+cr)/(2*dt), (cf*df-cr*dr)/(2*dt)*np.cos(beta[i])],
             [(cf*df-cr*dr)/(2*dt)*np.cos(beta[i]), I/dt**2+(cf*df**2-cr*dr**2)/(2*dt)*np.cos(beta[i])**2]])
        b1 = np.array([(2*m/dt**2-(kf+kr)), -(kf*df-kr*dr)*np.cos(beta[i])])*sm[i]
        b2 = np.array([(-m/dt**2+(cf+cr)/(2*dt)), (cf*df-cr*dr)/(2*dt)*np.cos(beta[i])])*sm[i-1]
        b3 = np.array([(kr*dr-kf*df)*np.sin(beta[i]), I/dt**2*2*beta[i]-(kf*df**2+kr*dr**2)*np.sin(beta[i])*np.cos(beta[i])])
        b4 = np.array([-(cr*dr-cf*df)/(2*dt)*np.cos(beta[i]), -I/dt**2+(cf*df**2+cr*dr**2)/(2*dt)*np.cos(beta[i])**2])*beta[i-1]
        b5 = np.array([f[i], h[i]*np.cos(beta[i])])
        B = b1 + b2 + b3 + b4 + b5
        sm[i+1], beta[i+1] = np.linalg.solve(A,B)

fig, ax = plt.subplots()
ax.plot(t,swr,'k')
ax.plot(t,swf,'b')
ax.plot(t,sm+1,'r')
ax.plot(t,beta,'g')
# ax.set_ylim((-10,10))
fig.savefig("test.png")