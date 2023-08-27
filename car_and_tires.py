# Authors : Bernardo Vasconcellos
# Description : This code models the displacement of a ponctual mass supported by a damper an a spring, subjected to a one dof displacement forcing. 

import numpy as np
import matplotlib.pyplot as plt 

# Vehicule geometry
wheelbase = 1.5 # m
front_axle_width = 1 # m
rear_axle_width = 1 # m
weight_dist_rear = 0.6 # 0 to 1

# Track coordinates
Ns = 200
x = np.linspace(0,100,Ns)
dx = x[1]-x[0]
y = 2*(0.5 + 0.5*np.tanh(0.5*(x-25)))*np.sin(2*np.pi*(x-25)/25)
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
normal_Scgeo = np.array([-tan_Scgeo[1,:],tan_Scgeo[0,:],tan_Scgeo[2,:]])
#Sfr = Scgeo + 


# Plot track and vehicule trajectory 
fig, ax = plt.subplots()
#ax.scatter(Strack[0,:],Strack[1,:],c='g')
ax.scatter(Scgeo[0,:],Scgeo[1,:],c='b',marker='d')
ax.quiver(*Scgeo[0:2,:], tan_Scgeo[0,:], tan_Scgeo[1,:],angles='xy', color='k')
ax.quiver(*Scgeo[0:2,:], normal_Scgeo[0,:], normal_Scgeo[1,:], angles='xy', color='r')
fig.savefig('track.png')
plt.show()