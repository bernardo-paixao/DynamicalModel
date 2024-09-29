import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
 

def ode(t,X):

    m = 200 # Vehicule mass [Kg]
    L = 1.5 # Distance between axes [m]
    W = 1.0 # Width of the car [m]
    fwr = 0.5 # Front weight ratio 
    kr = 4000 # Rear string rate [N/m]
    kf = 4000 # Front string rate [N/m]
    cr = 1000 # Rear damper rate [N.s/m]
    cf = 1000 # Front damper rate [N.s/m]
    g = 9.81 # Gravity [m/s**2]

    I = 1/12*m*L**2 # Vehicle approx moment of inertia [kg.m**2]

    # Reference frame of the CG projection on the track
    Xrl = np.array([-(1-fwr)*L, -W/2, 0]) # Position of the rear left contact patch
    Xrr = np.array([-(1-fwr)*L, W/2, 0]) # Position of the rear right contact patch
    Xfl = np.array([fwr*L, -W/2, 0]) # Position of the front left contact patch
    Xfr = np.array([fwr*L, W/2, 0]) # Position of the front right contact patch

    Arl = np.array([-(1-fwr)*L, -W/2, 0]) # Position of the rear left contact
    Arr = np.array([-(1-fwr)*L, W/2, 0]) # Position of the rear right contact patch
    Afl = np.array([fwr*L, -W/2, 0]) # Position of the front left contact patch
    Afr = np.array([fwr*L, W/2, 0]) # Position of the front right contact patch

    Kr = np.array([0, 0, kr])
    Kf = np.array([0, 0, kf])
    Cr = np.array([0, 0, cr])
    Cf = np.array([0, 0, cf])

    Fbreak = np.array([0, 0, 0])
    Fgas = np.array([1000*np.sin(2*np.pi*t/10), 0, 0])
    Xtrack = np.array([0, 0, 0])
    Xfollow = np.array([1, 1, 0])
    dXtrack = np.array([0, 0, 0])
    
    G = np.array([0, 0, -g])

    dXcg = X[0:3]
    Xcg = X[3:6]
    dTcg = X[6:9]
    Tcg = X[9:12]

    Rx = np.array([[1, 0, 0],[0, np.cos(Tcg[0]), -np.sin(Tcg[0])],[0, np.sin(Tcg[0]), np.cos(Tcg[0])]])
    Ry = np.array([[np.cos(Tcg[1]), 0, np.sin(Tcg[1])],[0, 1, 0],[-np.sin(Tcg[1]), 0, np.cos(Tcg[1])]])
    Rz = np.array([[np.cos(Tcg[2]), -np.sin(Tcg[2]), 0],[np.sin(Tcg[2]), np.cos(Tcg[2]), 0],[0, 0, 1]])
    
    dRx = np.array([[1, 0, 0],[0, -np.sin(Tcg[0])*dTcg[0], -np.cos(Tcg[0])*dTcg[0]],[0, np.cos(Tcg[0])*dTcg[0], -np.sin(Tcg[0])*dTcg[0]]])
    dRy = np.array([[-np.sin(Tcg[1])*dTcg[1], 0, np.cos(Tcg[1])*dTcg[1]],[0, 1, 0],[-np.cos(Tcg[1])*dTcg[1], 0, -np.sin(Tcg[1])*dTcg[1]]])
    dRz = np.array([[-np.sin(Tcg[2])*dTcg[2], -np.cos(Tcg[2])*dTcg[2], 0],[np.cos(Tcg[2])*dTcg[2], -np.sin(Tcg[2])*dTcg[2], 0],[0, 0, 1]])

    aArl = Xcg + Rx@Ry@Rz@Arl
    aArr = Xcg + Rx@Ry@Rz@Arr
    aAfl = Xcg + Rx@Ry@Rz@Afl
    aAfr = Xcg + Rx@Ry@Rz@Afr
    
    dArl = dXcg + dRx@Ry@Rz@Arl + Rx@dRy@Rz@Arl + Rx@Ry@dRz@Arl
    dArr = dXcg + dRx@Ry@Rz@Arr + Rx@dRy@Rz@Arr + Rx@Ry@dRz@Arr
    dAfl = dXcg + dRx@Ry@Rz@Afl + Rx@dRy@Rz@Afl + Rx@Ry@dRz@Afl
    dAfr = dXcg + dRx@Ry@Rz@Afr + Rx@dRy@Rz@Afr + Rx@Ry@dRz@Afr

    aXrl = Xrl+Xcg*Xfollow+Xtrack*np.sin(2*np.pi*t+np.pi/6)
    aXrr = Xrr+Xcg*Xfollow+Xtrack*np.sin(2*np.pi*t+np.pi/3)
    aXfl = Xfl+Xcg*Xfollow+Xtrack*np.sin(2*np.pi*t+np.pi/2)
    aXfr = Xfr+Xcg*Xfollow+Xtrack*np.sin(2*np.pi*t+np.pi)

    dXrl = dXcg*Xfollow+dXtrack*np.pi*np.cos(2*np.pi*t+np.pi/6)
    dXrr = dXcg*Xfollow+dXtrack*np.pi*np.cos(2*np.pi*t+np.pi/3)
    dXfl = dXcg*Xfollow+dXtrack*np.pi*np.cos(2*np.pi*t+np.pi/2)
    dXfr = dXcg*Xfollow+dXtrack*np.pi*np.cos(2*np.pi*t+np.pi)

    Zcg = np.array([0,0,Xcg[2]])

    d2Xcg = (Kr*((Xrr-aArr)+(Xrl-aArl)) + Kf*((aXfr-aAfr)+(aXfl-aAfl)) + Cr*((dXrr-dArr)+(dXrl-dArl)) + Cf*((dXfr-dAfr)+(dXfl-dAfl))+ Fbreak + Fgas + G*m)/m
    d2Tcg = (np.cross(Arl,Kr*(Xrl-aArl)) + np.cross(Arr,Kr*(Xrr-aArr)) + np.cross(Afl,Kf*(Xfl-aAfl)) + np.cross(Afr,Kf*(Xfr-aAfr))
            + np.cross(Arl,Cr*(dXrl-dArl)) + np.cross(Arr,Cr*(dXrr-dArr)) + np.cross(Afl,Cf*(dXfl-dAfl)) + np.cross(Afr,Cf*(dXfr-dAfr)) + np.cross(Zcg,Fbreak) + np.cross(Zcg,Fgas))/I

    return np.concatenate((d2Xcg, dXcg, d2Tcg, dTcg))


if __name__ == '__main__':

    x0 = np.zeros(12)
    # init_sol = solve_ivp(ode,[0,10],x0,method='RK45',args=ztrack)
    # x0[5] = init_sol.y[5,-1] # Initial cg height

    x0[11] = 0
    sol = solve_ivp(ode,[0,10],x0,method='RK45')

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,ncols=1)
    ax0.plot(sol.t,sol.y[3,:],label='CG x') # CG height (z) position
    ax0.plot(sol.t,sol.y[0,:],label='Cg Vx') # CG Vz velocity
    ax1.plot(sol.t,sol.y[5,:],label='CG z') # CG height (z) position
    ax1.plot(sol.t,sol.y[2,:],label='Cg Vz') # CG Vz velocity
    ax2.plot(sol.t,sol.y[9,:],label='Row') 
    ax2.plot(sol.t,sol.y[10,:],label='Pitch')
    ax2.plot(sol.t,sol.y[11,:],label='Yaw') 
    ax0.legend()
    ax1.legend()
    ax2.legend()
    plt.show()