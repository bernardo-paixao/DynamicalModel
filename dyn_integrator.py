import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import time

def damper_spring_tyres():
    rigid_K = 1e4
    K_tyre = np.array([[1e3, 0, 0],
                      [0, 1e3, 0],
                      [0, 0, 200]])
    
    K_FR = np.array([[rigid_K, 0, 0],
                      [0, rigid_K, 0],
                      [0, 0, 200]])
    
    K_FL = np.array([[rigid_K, 0, 0],
                      [0, rigid_K, 0],
                      [0, 0, 200]])
    
    K_RR = np.array([[rigid_K, 0, 0],
                      [0, rigid_K, 0],
                      [0, 0, 200]])
    
    K_RL = np.array([[rigid_K, 0, 0],
                      [0, rigid_K, 0],
                      [0, 0, 200]])

    C_FR = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 200]])
    
    C_FL = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 200]])
    
    C_RR = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 200]])
    
    C_RL = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 200]])
    return K_tyre, K_FR, K_FL, K_RR, K_RR, C_FR, C_FL, C_RR, C_RL

def transf_gMc(gamma,beta,alpha):
    gMc = np.array([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma)],
                    [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)],
                    [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]])
    return gMc

def gMc_dot(g,b,a,dg,db,da):
    dgMc = np.array([[da*np.sin(a)*(-np.cos(b))-np.cos(a)*db*np.sin(b) , da*(-(np.sin(a)*np.sin(b)*np.sin(g) + np.cos(a)*np.cos(g))) + np.cos(a)*db*np.cos(b)*np.sin(g) + dg*(np.cos(a)*np.sin(b)*np.cos(g) + np.sin(a)*np.sin(g)) , da*(np.cos(a)*np.sin(g)-np.sin(a)*np.sin(b)*np.cos(g)) + np.cos(a)*db*np.cos(b)*np.cos(g) + dg*(np.sin(a)*np.cos(g) - np.cos(a)*np.sin(b)*np.sin(g))],
                     [da*np.cos(a)*np.cos(b) - np.sin(a)*db*np.sin(b) , da*(np.cos(a)*np.sin(b)*np.sin(g)-np.sin(a)*np.cos(g)) + np.sin(a)*db*np.cos(b)*np.sin(g) + dg*(np.sin(a)*np.sin(b)*np.cos(g) - np.cos(a)*np.sin(g)) , da*(np.cos(a)*np.sin(b)*np.cos(g) + np.sin(a)*np.sin(g)) + np.sin(a)*db*np.cos(b)*np.cos(g) - dg*(np.sin(a)*np.sin(b)*np.sin(g) + np.cos(a)*np.cos(g))],
                     [db*np.cos(b) , np.cos(b)*dg*np.cos(g) - db*np.sin(b)*np.sin(g) , db*np.sin(b)*(-np.cos(g)) - np.cos(b)* dg*np.sin(g)]])
    return dgMc

def transf_tMg(xu,yu,zu):
    tMg = np.array([[xu[0],yu[0],zu[0]],
                  [xu[1],yu[1],zu[1]],
                  [xu[2],yu[2],zu[2]]])
    return tMg

def tMg_dot(dxu,dyu,dzu):
    dtMg = np.array([[dxu[0],dyu[0],dzu[0]],
                  [dxu[1],dyu[1],dzu[1]],
                  [dxu[2],dyu[2],dzu[2]]])
    return dtMg

def vector_interpolated(fullvector,time):
        vecx = CubicSpline(time, fullvector[0,:])
        vecy = CubicSpline(time, fullvector[1,:])
        vecz = CubicSpline(time, fullvector[2,:])
        return [vecx,vecy,vecz]

def wheel_pos_reftrack(wheelbase,weight_dist_rear,front_axle_width,rear_axle_width,Sgeo):
    fr = np.array([wheelbase*(1-weight_dist_rear),-front_axle_width/2,0])+Sgeo
    fl = np.array([wheelbase*(1-weight_dist_rear),front_axle_width/2,0])+Sgeo
    rr = np.array([-wheelbase*weight_dist_rear,-rear_axle_width/2,0])+Sgeo
    rl = np.array([-wheelbase*weight_dist_rear,rear_axle_width/2,0])+Sgeo
    return fr, fl, rr, rl

def Sanchor_track(tMg,gMc,cSanchor, tScg):
    return tMg@gMc@cSanchor + tScg

def ODE(t,X,track_inputs,car_inputs,time):
    # The time derivates are computed in the Track (inertial) reference frame!
    id_t = np.argmin(np.abs(time-t))

    m_CG = 200 # kg
    m_FR = 10 # kg
    m_FL = 10 # kg
    m_RR = 10 # kg
    m_RL = 10 # kg

    xu = track_inputs["tan_track"][:,id_t]
    yu = track_inputs["level_track"][:,id_t]
    zu = track_inputs["normal_track"][:,id_t]
    dxu = track_inputs["dtan_track"][:,id_t]
    dyu = track_inputs["dlevel_track"][:,id_t]
    dzu = track_inputs["dnormal_track"][:,id_t]

    func_Sgeo = track_inputs["S_geo"]
    Sgeo = np.array([func_Sgeo[0](t),func_Sgeo[1](t),func_Sgeo[2](t)])
    S_FRtrack, S_FLtrack, S_RRtrack, S_RLtrack = wheel_pos_reftrack(car_inputs["Wheelbase"],car_inputs["Weight_dist"], car_inputs["Rear_axle_width"], car_inputs["Front_axle_width"],Sgeo)
    # print(np.shape(S_FRtrack))
    # print(f"t={t}, Sgeox={Sgeo[0]}, S_FR={S_FRtrack[0]}")
    cSfranchor = car_inputs["S_FRanchor"]
    cSflanchor = car_inputs["S_FLanchor"]
    cSrranchor = car_inputs["S_RRanchor"]
    cSrlanchor = car_inputs["S_RLanchor"]

    K_tyre, K_FR, K_FL, K_RR, K_RL, C_FR, C_FL, C_RR, C_RL = damper_spring_tyres()

    # Car Inertia Moment
    I_CG = np.array([[100, 0, 0],
                    [0, 100, 0],
                    [0, 0, 100]])
    
    
    g = np.array([0,0,9.81])
    S0tyre = np.array([0,0,9.81*(m_FR+0.25*m_CG)/K_tyre[-1,-1]])
    S0damper = np.array([0,0,9.81*(m_FR+0.25*m_CG)/K_FL[-1,-1]])

    dSfrdt = np.array([X[0],X[1],X[2]])
    Sfr = np.array([X[3],X[4],X[5]])
    dSfldt = np.array([X[6],X[7],X[8]])
    Sfl = np.array([X[9],X[10],X[11]])
    dSrrdt = np.array([X[12],X[13],X[14]])
    Srr = np.array([X[15],X[16],X[17]])
    dSrldt = np.array([X[18],X[19],X[20]])
    Srl = np.array([X[21],X[22],X[23]])
    dScgdt = np.array([X[24],X[25],X[26]])
    Scg = np.array([X[27],X[28],X[29]])
    dthetadt = np.array([X[30],X[31],X[32]])
    theta = np.array([X[33],X[34],X[35]])
    
    gMc = transf_gMc(theta[0],theta[1],theta[2])
    tMg = transf_tMg(xu,yu,zu)
    dot_gMc = gMc_dot(theta[0],theta[1],theta[2],dthetadt[0],dthetadt[1],dthetadt[2])
    dot_tMg = tMg_dot(dxu,dyu,dzu)
    
    Sfranchor = Sanchor_track(tMg,gMc,cSfranchor,Scg)
    Sflanchor = Sanchor_track(tMg,gMc,cSflanchor,Scg)
    Srranchor = Sanchor_track(tMg,gMc,cSrranchor,Scg)
    Srlanchor = Sanchor_track(tMg,gMc,cSrlanchor,Scg)
  
    Ffranchor = K_FR@(Sfr-Sfranchor+S0damper) + C_FR@(dSfrdt-(tMg@dot_gMc@cSfranchor+dot_tMg@gMc@cSfranchor+dScgdt)) #+ np.array([1,1,0])*K_tyre@(S_FRtrack-Sfr+S0tyre)
    Fflanchor = K_FL@(Sfl-Sflanchor+S0damper) + C_FL@(dSfldt-(tMg@dot_gMc@cSflanchor+dot_tMg@gMc@cSflanchor+dScgdt)) #+ np.array([1,1,0])*K_tyre@(S_FLtrack-Sfl+S0tyre)
    Frranchor = K_RR@(Srr-Srranchor+S0damper) + C_RR@(dSrrdt-(tMg@dot_gMc@cSrranchor+dot_tMg@gMc@cSrranchor+dScgdt)) #+ np.array([1,1,0])*K_tyre@(S_RRtrack-Srr+S0tyre)
    Frlanchor = K_RL@(Srl-Srlanchor+S0damper) + C_RL@(dSrldt-(tMg@dot_gMc@cSrlanchor+dot_tMg@gMc@cSrlanchor+dScgdt)) #+ np.array([1,1,0])*K_tyre@(S_RLtrack-Srl+S0tyre)

    ddSfrdt2 = (K_tyre@(S_FRtrack-Sfr+S0tyre)- Ffranchor)/m_FR -g
    ddSfldt2 = (K_tyre@(S_FLtrack-Sfl+S0tyre)- Fflanchor)/m_FL -g
    ddSrrdt2 = (K_tyre@(S_RRtrack-Srr+S0tyre)- Frranchor)/m_RR -g
    ddSrldt2 = (K_tyre@(S_RLtrack-Srl+S0tyre)- Frlanchor)/m_RL -g

    ddScgdt2 = (Ffranchor+Fflanchor+Frranchor+Frlanchor)/m_CG - g

    Tfranchor = np.cross(Scg-Sfranchor,Ffranchor) + np.cross(Sfranchor-S_FRtrack,K_tyre@(S_FRtrack-Sfr))
    Tflanchor = np.cross(Scg-Sflanchor,Fflanchor) + np.cross(Sflanchor-S_FLtrack,K_tyre@(S_FLtrack-Sfl))
    Trranchor = np.cross(Scg-Srranchor,Frranchor) + np.cross(Srranchor-S_RRtrack,K_tyre@(S_RRtrack-Srr))
    Trlanchor = np.cross(Scg-Srlanchor,Frlanchor) + np.cross(Srlanchor-S_RLtrack,K_tyre@(S_RLtrack-Srl))
    
    ddthetadt2 = np.linalg.inv(I_CG)@(Tfranchor+Tflanchor+Trranchor+Trlanchor) # This equation is wrong because the rotations need to be consider in the Geometrique reference frame

    return np.concatenate((ddSfrdt2,dSfrdt,ddSfldt2,dSfldt,ddSrrdt2,dSrrdt,ddSrldt2,dSrldt,ddScgdt2,dScgdt,ddthetadt2,dthetadt))

    
if __name__ == '__main__':
    
    t = np.linspace(0,5,120)

    # Vehicule geometry
    wheelbase = 1.5 # m
    front_axle_width = 1 # m
    rear_axle_width = 1 # m
    weight_dist_rear = 0.5 # 0 to 1

    Sgeo = np.array([0.1*t**2/2,0*t,0*t])
    tan = np.array([np.ones(len(t)),0*t,0*t])
    level = np.array([0*t,np.ones(len(t)),0*t])
    normal = np.array([0*t,0*t,np.ones(len(t))])
    dtan = np.array([0*t,0*t,0*t])
    dlevel = np.array([0*t,0*t,0*t])
    dnormal = np.array([0*t,0*t,0*t])

    fr_refcar = np.array([wheelbase*(1-weight_dist_rear),-front_axle_width/2,0])
    fl_refcar = np.array([wheelbase*(1-weight_dist_rear),front_axle_width/2,0])
    rr_refcar = np.array([-wheelbase*weight_dist_rear,-rear_axle_width/2,0])
    rl_refcar = np.array([-wheelbase*weight_dist_rear,rear_axle_width/2,0])

    CarInputs = {"S_FRanchor":fr_refcar,
                 "S_FLanchor":fl_refcar,
                 "S_RRanchor":rr_refcar,
                 "S_RLanchor":rl_refcar,
                 "Wheelbase": wheelbase,
                 "Weight_dist": weight_dist_rear,
                 "Rear_axle_width":rear_axle_width,
                 "Front_axle_width":front_axle_width}
    
    TrackInputs = {"tan_track":tan,
                   "level_track":level,
                   "normal_track":normal,
                   "dtan_track":dtan,
                   "dlevel_track":dlevel,
                   "dnormal_track":dnormal,
                   "S_geo":vector_interpolated(Sgeo,t)}
    
    x0 = np.zeros(36)
    x0[3:5+1] = fr_refcar+np.array([0,0,0.1])
    x0[9:11+1] = fl_refcar+np.array([0,0,0.1])
    x0[15:17+1] = rr_refcar+np.array([0,0,0.1])
    x0[21:23+1] = rl_refcar+np.array([0,0,0.1])
    x0[27:29+1] = np.array([0,0,0.5])

    # get the start time
    st = time.time()

    var = ODE(0, x0, TrackInputs, CarInputs, t)
    print(f'ddSfrdt2 = {var[0:3]}')
    print(f'dSfrdt = {var[3:6]}')
    print(f'ddSfldt2 = {var[6:9]}')
    print(f'dSfldt = {var[9:12]}')
    print(f'ddSrrdt2 = {var[12:15]}')
    print(f'dSrrdt = {var[15:18]}')
    print(f'ddSrldt2 = {var[18:21]}')
    print(f'dSrldt = {var[21:24]}')
    print(f'ddScgdt2 = {var[24:27]}')
    print(f'dScgdt = {var[27:30]}')
    print(f'ddthetadt2 = {var[30:33]}')
    print(f'dthetadt = {var[33:36]}')


    sol = solve_ivp(ODE,[t[0],t[-1]],x0,method='RK45',t_eval=t,args=(TrackInputs, CarInputs, t))

    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', np.round(elapsed_time,2), 'seconds')

    var2 = sol.y

    fig, ax = plt.subplots()
    ax.plot(t,var2[29,:]) # CG height (z) position
    ax.plot(t,var2[26,:]) # CG Vz velocity
    ax.plot(t,var2[27,:]) # CG x position
    ax.plot(t,var2[24,:]) # CG Vx velocity
    ax.plot(t,var2[28,:]) # CG y position
    ax.plot(t,var2[25,:]) # CG Vy velocity

    ax.set_xlabel('t')
    ax.set_ylabel('Displacement (m) / Velocity (m/s)')
    ax.legend(['z','Vz','x','Vx','y','Vy'], shadow=True)
    ax.set_title('CG position and velocity')

    fig2, ax2 = plt.subplots()
    ax2.plot(t,var2[5,:]) # Front Right wheel z position
    ax2.plot(t,var2[11,:]) # Front Left wheel z position
    ax2.plot(t,var2[17,:]) # Rear Right wheel z position
    ax2.plot(t,var2[23,:]) # Rear Left wheel z position

    ax2.set_xlabel('t')
    ax2.set_ylabel('Displacement (m) / Velocity (m/s)')
    ax2.legend(['FR z','FL z','RR z','RL z'], shadow=True)
    ax2.set_title('Wheels position and velocity')

    fig3, ax3 = plt.subplots()
    ax3.plot(t,var2[33,:]) # Row
    ax3.plot(t,var2[34,:]) # Pich
    ax3.plot(t,var2[35,:]) # Yaw

    ax3.set_xlabel('t')
    ax3.set_ylabel('Angle (rad)')
    ax3.legend(['Row','Pich','Yaw'], shadow=True)
    ax3.set_title('Pitch row and yaw')

    plt.show()
    