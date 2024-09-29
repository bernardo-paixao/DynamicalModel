import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
 

def ode(t,X,case=None):

    m = 200 # Vehicule mass [Kg]
    mt = 10 # Unsprung mass (1/4) [Kg]
    L = 1.5 # Distance between axes [m]
    W = 1.0 # Width of the car [m]
    fwr = 0.5 # Front weight ratio 
    kr = 4000 # Rear spring rate [N/m]
    kf = 4000 # Front spring rate [N/m]
    cr = 1000 # Rear damper rate [N.s/m]
    cf = 1000 # Front damper rate [N.s/m]
    kt = 5000 # Tire spring rate [N/m]
    ct = 100 # Tire dampening rate [N.s/m]
    g = 9.81 # Gravity [m/s**2]

    I = 1/12*m*L**2 # Vehicle approx moment of inertia [kg.m**2]

    # Suspension anchoring points on the cg's reference frame
    Arl = np.array([-(1-fwr)*L, -W/2, 0]) # Position of the rear left anchor
    Arr = np.array([-(1-fwr)*L, W/2, 0]) # Position of the rear right contact anchor
    Afl = np.array([fwr*L, -W/2, 0]) # Position of the front left contact anchor
    Afr = np.array([fwr*L, W/2, 0]) # Position of the front right contact anchor

    # Spring, dampeners and tires constants
    stiff = 0
    Kr = np.array([stiff, stiff, kr])
    Kf = np.array([stiff, stiff, kf])
    Cr = np.array([stiff, stiff, cr])
    Cf = np.array([stiff, stiff, cf])

    Ktire = np.array([10*kt,10*kt, kt])
    Ctire = np.array([ct, ct, ct])

    # Breaking, accelarating and track inputs
    if case=='gas':
        Fgas = np.array([300*np.sin(2*np.pi*t/25), 0, 0],) # Gas force in the cg's reference frame
    else:
        Fgas = np.array([0, 0, 0],) # Gas force in the cg's reference frame

    Fbreak = np.array([0, 0, 0]) # Break force in the cg's reference frame
    Xtrack = np.array([0, 0, 0]) # Can only have z component
    dXtrack = np.array([0, 0, 0]) # Can only have z component
    Xfollow = np.array([1, 1, 0])
    G = np.array([0, 0, -g])

    # Recovering state variables
    dXcg = X[0:3]
    Xcg = X[3:6]
    dTcg = X[6:9]
    Tcg = X[9:12]
    dSrr = X[12:15]
    Srr = X[15:18]
    dSrl = X[18:21]
    Srl = X[21:24]
    dSfr = X[24:27]
    Sfr = X[27:30]
    dSfl = X[30:33]
    Sfl = X[33:36]


    # Rotation matrices from CG's reference frame to inertial reference frame
    Rx = np.array([[1, 0, 0],[0, np.cos(Tcg[0]), -np.sin(Tcg[0])],[0, np.sin(Tcg[0]), np.cos(Tcg[0])]])
    Ry = np.array([[np.cos(Tcg[1]), 0, np.sin(Tcg[1])],[0, 1, 0],[-np.sin(Tcg[1]), 0, np.cos(Tcg[1])]])
    Rz = np.array([[np.cos(Tcg[2]), -np.sin(Tcg[2]), 0],[np.sin(Tcg[2]), np.cos(Tcg[2]), 0],[0, 0, 1]])
    
    dRx = np.array([[1, 0, 0],[0, -np.sin(Tcg[0])*dTcg[0], -np.cos(Tcg[0])*dTcg[0]],[0, np.cos(Tcg[0])*dTcg[0], -np.sin(Tcg[0])*dTcg[0]]])
    dRy = np.array([[-np.sin(Tcg[1])*dTcg[1], 0, np.cos(Tcg[1])*dTcg[1]],[0, 1, 0],[-np.cos(Tcg[1])*dTcg[1], 0, -np.sin(Tcg[1])*dTcg[1]]])
    dRz = np.array([[-np.sin(Tcg[2])*dTcg[2], -np.cos(Tcg[2])*dTcg[2], 0],[np.cos(Tcg[2])*dTcg[2], -np.sin(Tcg[2])*dTcg[2], 0],[0, 0, 1]])

    # Anchor positions and velocities in the interial reference frame
    aArl = Xcg + Rx@Ry@Rz@Arl
    aArr = Xcg + Rx@Ry@Rz@Arr
    aAfl = Xcg + Rx@Ry@Rz@Afl
    aAfr = Xcg + Rx@Ry@Rz@Afr
    
    dArl = dXcg + dRx@Ry@Rz@Arl + Rx@dRy@Rz@Arl + Rx@Ry@dRz@Arl
    dArr = dXcg + dRx@Ry@Rz@Arr + Rx@dRy@Rz@Arr + Rx@Ry@dRz@Arr
    dAfl = dXcg + dRx@Ry@Rz@Afl + Rx@dRy@Rz@Afl + Rx@Ry@dRz@Afl
    dAfr = dXcg + dRx@Ry@Rz@Afr + Rx@dRy@Rz@Afr + Rx@Ry@dRz@Afr

    # Contact pach positions and velocities in the interial reference frame
    # The contact pach is assumed to be the projection of the anchor position on to the track
    aXrl = (Xcg+aArl)*Xfollow+Xtrack
    aXrr = (Xcg+aArr)*Xfollow+Xtrack
    aXfl = (Xcg+aAfl)*Xfollow+Xtrack
    aXfr = (Xcg+aAfr)*Xfollow+Xtrack

    dXrl = (dXcg+dArl)*Xfollow+dXtrack
    dXrr = (dXcg+dArr)*Xfollow+dXtrack
    dXfl = (dXcg+dAfl)*Xfollow+dXtrack
    dXfr = (dXcg+dAfr)*Xfollow+dXtrack

    Zcg = np.array([0,0,Xcg[2]])

    
    d2Srr = (Ktire*(aXrr-Srr) + Kr*(aArr-Srr) + Ctire*(dXrr-dSrr) + Cr*(dArr-dSrr) + Rx@Ry@Fbreak/4 + Rx@Ry@Fgas/4 + G*mt)/mt
    d2Srl = (Ktire*(aXrl-Srl) + Kr*(aArl-Srl) + Ctire*(dXrl-dSrl) + Cr*(dArl-dSrl) + Rx@Ry@Fbreak/4 + Rx@Ry@Fgas/4 + G*mt)/mt
    d2Sfr = (Ktire*(aXfr-Sfr) + Kf*(aAfr-Sfr) + Ctire*(dXfr-dSfr) + Cf*(dAfr-dSfr) + Rx@Ry@Fbreak/4 + Rx@Ry@Fgas/4 + G*mt)/mt
    d2Sfl = (Ktire*(aXfl-Sfl) + Kf*(aAfl-Sfl) + Ctire*(dXfl-dSfl) + Cf*(dAfl-dSfl) + Rx@Ry@Fbreak/4 + Rx@Ry@Fgas/4 + G*mt)/mt

    d2Xcg = (Kr*((Srr-aArr)+(Srl-aArl)) + Kf*((Sfr-aAfr)+(Sfl-aAfl)) + Cr*((dSrr-dArr)+(dSrl-dArl)) + Cf*((dSfr-dAfr)+(dSfl-dAfl)) + Rx@Ry@Fbreak + Rx@Ry@Fgas + G*m)/m

    d2Tcg = (np.cross(Arl,Kr*(Srl-aArl)) + np.cross(Arr,Kr*(Srr-aArr)) + np.cross(Afl,Kf*(Sfl-aAfl)) + np.cross(Afr,Kf*(Sfr-aAfr))
            + np.cross(Arl,Cr*(dSrl-dArl)) + np.cross(Arr,Cr*(dSrr-dArr)) + np.cross(Afl,Cf*(dSfl-dAfl)) + np.cross(Afr,Cf*(dSfr-dAfr)) + np.cross(Zcg,Rx@Ry@Fbreak) + np.cross(Zcg,Rx@Ry@Fgas))/I

    return np.concatenate((d2Xcg, dXcg, d2Tcg, dTcg, d2Srr, dSrr, d2Srl, dSrl, d2Sfr, dSfr, d2Sfl, dSfl))


if __name__ == '__main__':

    x0 = np.zeros(36)
    init_sol = solve_ivp(ode,[0,10],x0,method='RK45')
    x0 = init_sol.y[:,-1] # Initial cg height

    sol = solve_ivp(ode,[0,25],x0,method='RK45',args=['gas'])
    
    for i in range(36):
        sol.y[i,:] = sol.y[i,:]-x0[i]

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,ncols=1)
    fig.suptitle('CG variables', fontsize=12)
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

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4,ncols=1)
    fig.suptitle('Wheels variables', fontsize=12)
    ax0.plot(sol.t,sol.y[17,:],label='RR z') # CG height (z) position
    ax0.plot(sol.t,sol.y[14,:],label='RR Vz') # CG Vz velocity
    ax1.plot(sol.t,sol.y[23,:],label='RL z') # CG height (z) position
    ax1.plot(sol.t,sol.y[20,:],label='RL Vz') # CG Vz velocity
    ax2.plot(sol.t,sol.y[29,:],label='FR z') 
    ax2.plot(sol.t,sol.y[26,:],label='FR Vz')
    ax3.plot(sol.t,sol.y[35,:],label='FL z')
    ax3.plot(sol.t,sol.y[32,:],label='FL Vz')
    ax0.legend()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()