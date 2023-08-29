# Authors : Bernardo Vasconcellos
# Description : At the moment this code generates arbitrary track coordinates and velocity data. Givem that and some geometry information about the vehicule, it computes the postion of the wheels at each moment. 

import numpy as np
import matplotlib.pyplot as plt 

def main():
    # Vehicule geometry
    wheelbase = 1.5 # m
    front_axle_width = 1 # m
    rear_axle_width = 1 # m
    weight_dist_rear = 0.6 # 0 to 1

    # Track coordinates
    Ns = 200
    x = np.linspace(0,100,Ns)
    dx = x[1]-x[0]
    y = 2*(0.5 + 0.5*np.tanh(0.5*(x-25)))*np.sin(np.pi*(x-25)/25)
    z = np.zeros(Ns)
    Strack = np.array([x,y,z])
    DStrack = (Strack[:,1]-Strack[:,0])/np.linalg.norm(Strack[:,1]-Strack[:,0])

    # Define time and velocity (magnitude) data
    Nt = 500
    t = np.linspace(0,10,Nt)
    dt = t[1]-t[0]
    Vel_cgeo = 50/3.6*np.ones(Nt)

    # Compute cgeo projection position
    Scgeo = np.zeros((3,Nt))
    for j in range(Nt-1):
        if j==0:
            Scgeo[:,j+1] = Scgeo[:,j] + Vel_cgeo[j]*DStrack*dt
        else:
            id_closest = np.argmin(np.linalg.norm(Strack-Scgeo[:,j][:,np.newaxis],axis=0))
            if id_closest==Ns-1:
                Scgeo[:,j+1] = Scgeo[:,j]
            else :
                if np.linalg.norm(Strack[:,id_closest+1]-Scgeo[:,j])<=np.linalg.norm(Strack[:,id_closest-1]-Scgeo[:,j]):
                    DStrack = (Strack[:,id_closest+1]-Scgeo[:,j])/np.linalg.norm(Strack[:,id_closest+1]-Scgeo[:,j])
                else:
                    DStrack = (Strack[:,id_closest]-Scgeo[:,j])/np.linalg.norm(Strack[:,id_closest]-Scgeo[:,j])
                Scgeo[:,j+1] = Scgeo[:,j] + Vel_cgeo[j]*DStrack*dt

    tan_Scgeo = np.diff(Scgeo,axis=1)/np.linalg.norm(np.diff(Scgeo,axis=1),axis=0)
    tan_Scgeo = np.concatenate((tan_Scgeo,tan_Scgeo[:,-1][:,np.newaxis]),axis=1) 
    normal_Scgeo = np.array([-tan_Scgeo[1,:],tan_Scgeo[0,:],tan_Scgeo[2,:]]) #  Does not takes z in acccount!
    
    # Coordinates of the tire contact path in the car reference frame
    fr_refcar = np.array([wheelbase*(1-weight_dist_rear),-front_axle_width/2,0])
    fl_refcar = np.array([wheelbase*(1-weight_dist_rear),front_axle_width/2,0])
    rr_refcar = np.array([-wheelbase*weight_dist_rear,-rear_axle_width/2,0])
    rl_refcar = np.array([-wheelbase*weight_dist_rear,rear_axle_width/2,0])

    # Coordinates of the tire contact path in the track reference frame (does not takes z in acccount)
    fr_reftrack = np.array([tan_Scgeo[0,:]*fr_refcar[0]+normal_Scgeo[0,:]*fr_refcar[1]+Scgeo[0,:], tan_Scgeo[1,:]*fr_refcar[0]+normal_Scgeo[1,:]*fr_refcar[1]+Scgeo[1,:], normal_Scgeo[2,:]+Scgeo[2,:]])
    fl_reftrack = np.array([tan_Scgeo[0,:]*fl_refcar[0]+normal_Scgeo[0,:]*fl_refcar[1]+Scgeo[0,:], tan_Scgeo[1,:]*fl_refcar[0]+normal_Scgeo[1,:]*fl_refcar[1]+Scgeo[1,:], normal_Scgeo[2,:]+Scgeo[2,:]])
    rl_reftrack = np.array([tan_Scgeo[0,:]*rl_refcar[0]+normal_Scgeo[0,:]*rl_refcar[1]+Scgeo[0,:], tan_Scgeo[1,:]*rl_refcar[0]+normal_Scgeo[1,:]*rl_refcar[1]+Scgeo[1,:], normal_Scgeo[2,:]+Scgeo[2,:]])
    rr_reftrack = np.array([tan_Scgeo[0,:]*rr_refcar[0]+normal_Scgeo[0,:]*rr_refcar[1]+Scgeo[0,:], tan_Scgeo[1,:]*rr_refcar[0]+normal_Scgeo[1,:]*rr_refcar[1]+Scgeo[1,:], normal_Scgeo[2,:]+Scgeo[2,:]])

    # Plot track and vehicule trajectory 
    fig, ax = plt.subplots()
    #ax.scatter(Strack[0,:],Strack[1,:],c='g')
    ax.scatter(Scgeo[0,:],Scgeo[1,:],c='b',marker='d')
    ax.quiver(*Scgeo[0:2,:], tan_Scgeo[0,:], tan_Scgeo[1,:],angles='xy', color='k')
    selected = [0, 100, 200, 300, 400]
    ax.scatter(fr_reftrack[0,selected],fr_reftrack[1,selected],c='g',marker='s') 
    ax.scatter(fl_reftrack[0,selected],fl_reftrack[1,selected],c='r',marker='s')
    ax.scatter(rl_reftrack[0,selected],rl_reftrack[1,selected],c='y',marker='s')
    ax.scatter(rr_reftrack[0,selected],rr_reftrack[1,selected],c='m',marker='s')
    # ax.quiver(*Scgeo[0:2,:], normal_Scgeo[0,:], normal_Scgeo[1,:], angles='xy', color='r') # There is a bug with quiver, the vector direction is plotted incorrectly
    fig.savefig('track.png')
    plt.show()

if __name__ == '__main__':
    main()