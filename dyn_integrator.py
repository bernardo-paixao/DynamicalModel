import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def damper_spring_tyres():
    K_tyre = np.array([1e8, 0, 0],
                      [0, 1e8, 0],
                      [0, 0, 200])
    
    K_FR = np.array([1e8, 0, 0],
                      [0, 1e8, 0],
                      [0, 0, 200])
    
    K_FL = np.array([1e8, 0, 0],
                      [0, 1e8, 0],
                      [0, 0, 200])
    
    K_RR = np.array([1e8, 0, 0],
                      [0, 1e8, 0],
                      [0, 0, 200])
    
    K_RL = np.array([1e8, 0, 0],
                      [0, 1e8, 0],
                      [0, 0, 200])

    C_FR = np.array([0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 200])
    
    C_FL = np.array([0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 200])
    
    C_RR = np.array([0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 200])
    
    C_RL = np.array([0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 200])
    return K_tyre, K_FR, K_FL, K_RR, K_RR, C_FR, C_FL, C_RR, C_RL

def transf_gMc(gamma,beta,alpha):
    gMc = np.array([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma)],
                    [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)],
                    [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]])
    return gMc

def gMc_dot(g,b,a,dg,db,da):
    dgMc = np.array([[da*np.sin(a)*(-np.cos(b))-np.cos(a)*db*sin(b) , da*(-(np.sin(a)*np.sin(b)*np.sin(g) + np.cos(a)*np.cos(g))) + np.cos(a)*db*np.cos(b)*np.sin(g) + dg*(np.cos(a)*np.sin(b)*np.cos(g) + np.sin(a)*np.sin(g)) , da*(np.cos(a)*np.sin(g)-np.sin(a)*np.sin(b)*np.cos(g)) + np.cos(a)*db*np.cos(b)*np.cos(g) + dg*(np.sin(a)*np.cos(g) - np.cos(a)*np.sin(b)*np.sin(g))],
                     [da*np.cos(a)*np.cos(b) - np.sin(a)*db*np.sin(b) , da*(np.cos(a)*np.sin(b)*np.sin(g)-np.sin(a)*np.cos(g)) + np.sin(a)*db*np.cos(b)*sin(g) + dg*(np.sin(a)*np.sin(b)*np.cos(g) - np.cos(a)*np.sin(g)) , da*(np.cos(a)*np.sin(b)*np.cos(g) + np.sin(a)*np.sin(g)) + np.sin(a)*db*np.cos(b)*np.cos(g) - dg*(np.sin(a)*np.sin(b)*np.sin(g) + np.cos(a)*np.cos(g))],
                     [db*np.cos(b) , np.cos(b)*dg*np.cos(g) - db*np.sin(b)*np.sin(g) , db*np.sin(b)*(-np.cos(g)) - np.cos(b)* dg*np.sin(g)]])

def transf_tMg(xu,yu,zu):
    tMg = np.array([[xu[0],yu[0],zu[0]],
                  [xu[1],yu[1],zu[1]],
                  [xu[2],yu[2],zu[2]]])
    return tMg

def ODE(t,X,track_inputs,car_inputs):
    # The time derivates are computed in the Track (inertial) reference frame!

    m_CG = 200 # kg
    m_FR = 10 # kg
    m_FL = 10 # kg
    m_RR = 10 # kg
    m_RL = 10 # kg

    xu = track_inputs["tan_track"]
    yu = track_inputs["level_track"]
    zu = track_inputs["normal_track"]
    S_FRtrack = track_inputs["S_FRtrack"]
    S_FLtrack = track_inputs["S_FLtrack"]
    S_RRtrack = track_inputs["S_RRtrack"]
    S_RLtrack = track_inputs["S_RLtrack"]

    cSfranchor = car_inputs["S_FRanchor"]
    cSflanchor = car_inputs["S_FLanchor"]
    cSrranchor = car_inputs["S_RRanchor"]
    cSrlanchor = car_inputs["S_RLanchor"]

    K_tyre, K_FR, K_FL, K_RR, K_RR, C_FR, C_FL, C_RR, C_RL = damper_spring_tyres()

    I_CG = np.array([100, 0, 0],
                    [0, 100, 0],
                    [0, 0, 100])
    
    g = np.array([0,0,9.81])

    dSfrdt = X[0]
    Sfr = X[1]
    dSfldt = X[2]
    Sfl = X[3]
    dSrrdt = X[4]
    Srr = X[5]
    dSrldt = X[6]
    Srl = X[7]
    dScgdt = X[8]
    Scg = X[9]
    dthetadt = X[10]
    theta = X[11]
    
    gMc = transf_gMc(theta[0],theta[1],theta[2])
    tMg = transf_tMg(xu,yu,zu)

    ddSfrdt2 = (K_tyre@(S_FRtrack-Sfr)- K_FR@(Sfr-tMg@gMc@cSfranchor+Scg) - C_FR@(dSfrdt-))/m_FR -g

    x3_dot = (K_tyre@(S_FLtrack-X[3])- K_FL@(X[3]-X[17]) - C_FL@(X[2]-X[16]))/m_FL -g

    x5_dot = (K_tyre@(S_RRtrack-X[5])- K_RR@(X[5]-X[15]) - C_RR@(X[4]-X[14]))/m_RR -g

    x7_dot = (K_tyre@(S_RLtrack-X[7])- K_RL@(X[7]-X[13]) - C_RL@(X[6]-X[12]))/m_RL -g

    x9_dot = (K_FR@(X[1]-X[19]) + C_FR@(X[0]-X[18]) + K_FL@(X[3]-X[17]) + C_FL@(X[2]-X[16]) + K_RR@(X[5]-X[15]) + C_RR@(X[4]-X[14]) + K_RL@(X[7]-X[13]) + C_RL@(X[6]-X[12]))/m_CG - g

    x11_dot = np.linalg.inv(I_CG)@(np.cross((X[9]-X[19]),K_FR@(X[1]-X[19])+C_FR@(X[0]-X[18])) + np.cross((X[9]-X[17]),K_FL@(X[3]-X[17])+C_FL@(X[2]-X[16])) + np.cross((X[9]-X[13]), K_RL@(X[7]-X[13])+C_RL@(X[6]-X[12])) + np.cross((X[9]-X[15]),K_RR@(X[5]-X[15])+C_RR@(X[4]-X[14])))

