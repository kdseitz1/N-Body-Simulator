
#Computational Physics Final Project 2019
#Gabe Stash, John Lyons, Kevin Seitz
#WARNING, program compiles very slowly for larger values of N (N > 5). Time to run 
#approximately doubles when N is increased by 1. 7 bodies takes 30-60 seconds
#with a "good" CPU. 


from vpython import *
import numpy as np
from scipy.integrate import odeint
seednum = np.random.randint(1,546166688) # the current randomly generated seed
np.random.seed(seednum) #use see
print("Seed "+str(seednum)) #print out seed for user to see
N = int(input("Enter number of bodies to randomly generate: ")) #number of bodies
TN = float(input("Enter number of periods to generate over: ")) #recommend less than 10. Less generates faster and plays at a slower speed, but for a shorter time.
G = 4*np.pi**2 #G constant using units of AU, solar mass, and years for time
Ap = 35 #Halley's comet aphelion in AUs. Used as  a standard length in this program.
ec = .967 #Halley's comet eccentricity. Used to test. (change m1 to 1 and m2 and m3 to be very small)
a = Ap/(ec+1) #semimajor in AUs
b = a*np.sqrt(1-ec**2) #minor in AUs
T = np.sqrt(a**3) #period of body in years using the Halley's comet initial conditions. Used as a standard time measurement in this program.
body_colors = [color.red, color.green, color.blue,
              color.yellow, color.cyan, color.magenta] #list of colors to cycle through

#randomize arrays for initial conditions
M = np.random.rand(N)*2 #Mass, in Solar masses
px = (np.random.rand(N)*100)-50 #x  positions, in AU
py = np.random.rand(N)*100-50 #y
pz = np.random.rand(N)*100-50 #z pos
vx = np.random.rand(N)*1.5-.75 #x velocities, in AU/yr
vy = np.random.rand(N)*1.5-.75 #y velocities AU/yr
vz = np.random.rand(N)*1.5-.75 #z v in AU/yr
R = np.zeros(N)+1 #Generate range of animated sphere sizes, depending on the mass.
for i in range(N):
    if M[i] < 1:
        R[i] = 1 #min size
    elif M[i] > 5:
        R[i] = 5 #max size
    else:
        R[i] = M[i]
mtot = 0
for i in M:
    mtot += i #find total mass
conditions = np.concatenate((px,py,pz,vx,vy,vz))#initial conditions for diffeq

#diffeq solver. Imports boundary conditions (bc) and timestep (t)
def n_body(bc,t):    
    X = np.zeros(N,float)
    Y = np.zeros(N,float)
    Z = np.zeros(N,float)
    Vx = np.zeros(N,float)
    Vy = np.zeros(N,float)
    Vz = np.zeros(N,float)
    #apply initial conditions to arrays for x,y,z positions and velocities
    for i in range(N):
        X[i] = bc[i]
        Y[i] = bc[i+N]
        Z[i] = bc[i+2*N]
        Vx[i] = bc[i+3*N]
        Vy[i] = bc[i+4*N]
        Vz[i] = bc[i+5*N]
    #generate array of all the x equations of motion (x''(t)) to be solved by odeint
    ddx = np.zeros(np.size(M),float)  
    for i in range(np.size(M)):
        for j in range(np.size(M)):
            if (X[j]-X[i])**2+(Y[j]-Y[i])**2+(Z[j]-Z[i])**2 != 0:
                ddx[i]+= (G/M[i])*((M[i]*M[j]*(X[j]-X[i]))/((X[j]-X[i])**2+(Y[j]-Y[i])**2+(Z[j]-Z[i])**2)**(3/2))
                #formula is just the summation of all the gravitational accelerations
    #y EOM
    ddy = np.zeros(np.size(M),float)  
    for i in range(np.size(M)):
        for j in range(np.size(M)):
            if (X[j]-X[i])**2+(Y[j]-Y[i])**2+(Z[j]-Z[i])**2 != 0:
                ddy[i]+= (G/M[i])*((M[i]*M[j]*(Y[j]-Y[i]))/((X[j]-X[i])**2+(Y[j]-Y[i])**2+(Z[j]-Z[i])**2)**(3/2))
    #z EOM
    ddz = np.zeros(np.size(M),float)  
    for i in range(np.size(M)):
        for j in range(np.size(M)):
            if (X[j]-X[i])**2+(Y[j]-Y[i])**2+(Z[j]-Z[i])**2 != 0:
                ddz[i]+= (G/M[i])*((M[i]*M[j]*(Z[j]-Z[i]))/((X[j]-X[i])**2+(Y[j]-Y[i])**2+(Z[j]-Z[i])**2)**(3/2))
    DSolve = np.concatenate((Vx,Vy,Vz,ddx,ddy,ddz))
    return(DSolve) #scipy solve these
    #returns an array of (x1,x2,...,y1,y2,...z1,z2,...vx1,vx2,...,vy1,vy2,...,vz1,vz2,...) 
    #for each timestep (so 'step' number of the example array). 
    #rows are the individual steps. Columns are the components of the positions and velocities of each body.
    
step = 500 #number of datapoints to calculate
t1 = np.linspace(0, TN*T, step) #time step array

result = odeint(n_body,conditions,t1) #the 2D result array of all the calculated positions and velocities.

#just taking the intial conditions again, left in because can change zeros to ':' to plot
#1D arrays of initial positions for each body (for N bodies)
X0 = result[0,0:N]
Y0 = result[0,N:2*N]
Z0 = result[0,2*N:3*N]
#initial velocities
vx0 = result[0,3*N:4*N]
vy0 = result[0,4*N:5*N]
vz0 = result[0,5*N:6*N]
#2D arrays, rows are the positions at each timestep, columns are each N body (body 0,1,2...N-1)
X1 = result[:,0:N] 
Y1 = result[:,N:2*N]
Z1 = result[:,2*N:3*N]
#velocities, if needed
vx1 = result[:,3*N:4*N]
vy1 = result[:,4*N:5*N]
vz1 = result[:,5*N:6*N]
xcom = np.zeros(step)
ycom = np.zeros(step)
zcom = np.zeros(step)
#find the center of mass
for i in range(step):
    xcom[i] = np.dot(X1[i],M)/mtot
    ycom[i] = np.dot(Y1[i],M)/mtot
    zcom[i] = np.dot(Z1[i],M)/mtot
rcom = np.array([xcom,ycom,zcom]) #center of mass vector

scene = canvas(title='N-body Simulation',
     width=800, height=350, background=color.white) #make background white
Bodies = []
#create each body object, sphere for each, cycling through the color array.
for i in range(N):  
    Bodies.append(sphere(pos=vector(X0[i],Y0[i],Z0[i]), radius=R[i], color=body_colors[i%6],make_trail=True,trail_type="points",
              interval=1, retain=90))
#animate the bodies over the given period (loops back to beginning of animation)
COM = sphere(pos=vector(rcom[0,0],rcom[0,1],rcom[0,2]), radius=.5, color=color.black) #black point for center of mass

while(1):
    i = 1 #i is the timestep element (rows of position)
    while (i < step): #iterates through the positions at each timestep, i
        rate(30) # Number of frames/loops per second.
        #iterate through each gravitational body, j
        COM.pos.x = rcom[0,i]
        COM.pos.y = rcom[1,i]
        COM.pos.z = rcom[2,i]
        for j in range(N):  #j is the body element (columns of position)
            #show the position at each time, i
            Bodies[j].pos.x = X1[i,j] 
            Bodies[j].pos.y = Y1[i,j]
            Bodies[j].pos.z = Z1[i,j]
        scene.camera.follow(COM) #follow the center of mass of the system
        i = i + 1
print("End of program.")