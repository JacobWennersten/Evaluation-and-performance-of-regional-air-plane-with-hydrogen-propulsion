# MTF072 Computational Fluid Dynamics
# Task 3: laminar lid-driven cavity
# Template prepared by:
# Gonzalo Montero Villar
# Department of Mechanics and Maritime Sciences
# Division of Fluid Dynamics
# December 2020

#==============Packages needed=================
import matplotlib.pyplot as plt
import numpy as np
import math

plt.close('all')
#================= Inputs =====================

# Fluid properties and B. C. inputs

UWall = 1  # velocity of the upper wall
rho   = 1  # density
nu    = 1/1000 # kinematic viscosity

data_file = 'data_FOU_CD.txt' # data file where the given solution is stored

# Geometric inputs (fixed so that a fair comparison can be made)

mI = 11 # number of mesh points X direction. 
mJ = 11 # number of mesh points Y direction. 
xL =  1 # length in X direction
yL =  1 # length in Y direction

# Solver inputs

nIterations           = 10000   # maximum number of iterations
n_inner_iterations_gs = 30    # amount of inner iterations when solving 
                              # pressure correction with Gauss-Seidel
resTolerance = 0.00001 # convergence criteria for residuals
                      # each variable
alphaUV = 0.3        # under relaxation factor for U and V
alphaP  = 0.3        # under relaxation factor for P

# ================ Code =======================

# For all the matrices the first input makes reference to the x coordinate
# and the second input to the y coordinate, (i+1) is east and (j+1) north

# Allocate all needed variables
nI = mI + 1                      # number of nodes in the X direction. nodes 
                                  # added in the boundaries
nJ = mJ + 1                      # number of nodes in the Y direction. nodes 
                                  # added in the boundaries
coeffsUV   = np.zeros((nI,nJ,5)) # coefficients for the U and V equation
                                  # E, W, N, S and P
sourceUV   = np.zeros((nI,nJ,2)) # source coefficients for the U and V equation               
                                  # U and V
Spu        = np.zeros((nI,nJ))   # source coefficients for the pressure  

coeffsPp   = np.zeros((nI,nJ,5)) # coefficients for the pressure correction
                                  # equation E, W, N, S and P
sourcePp   = np.zeros((nI,nJ))           
   
Spp         = np.zeros((nI,nJ))                
              
                          
              
U          = np.zeros((nI,nJ))   # U velocity matrix
V          = np.zeros((nI,nJ))   # V velocity matrix
P          = np.zeros((nI,nJ))   # pressure matrix
Pp         = np.zeros((nI,nJ))   # pressure correction matrix



massFlows  = np.zeros((nI,nJ,4)) # mass flows at the faces
                                  # m_e, m_w, m_n and m_s

residuals  = np.zeros((3,1))     # U, V and conitnuity residuals

# Generate mesh and compute geometric variables

# Allocate all variables matrices
xCoords_N = np.zeros((nI,nJ)) # X coords of the nodes
yCoords_N = np.zeros((nI,nJ)) # Y coords of the nodes
xCoords_M = np.zeros((mI,mJ)) # X coords of the mesh points
yCoords_M = np.zeros((mI,mJ)) # Y coords of the mesh points
dxe_N     = np.zeros((nI,nJ)) # X distance to east node
dxw_N     = np.zeros((nI,nJ)) # X distance to west node
dyn_N     = np.zeros((nI,nJ)) # Y distance to north node
dys_N     = np.zeros((nI,nJ)) # Y distance to south node
dx_CV      = np.zeros((nI,nJ))    # X size of the node
dy_CV      = np.zeros((nI,nJ))    # Y size of the node
fxe        = np.zeros((nI,nJ))    #Non-equi east
fxw        = np.zeros((nI,nJ))    #Non-equi west
fyn        = np.zeros((nI,nJ))    #Non-equi north
fys        = np.zeros((nI,nJ))    #Non-equi south
b = 0      
dPp        = np.zeros((nI,nJ,4))
pGrad      = np.zeros((nI,nJ,2))
Pp_e       = np.zeros((nI,nJ))
Pp_w       = np.zeros((nI,nJ))
Pp_n       = np.zeros((nI,nJ))
Pp_s       = np.zeros((nI,nJ)) 


D          = np.zeros((nI,nJ,4))  #East and west, North and south
F          = np.zeros((nI,nJ,4))  #East and west, North and south

residuals_U = []
residuals_V = []
residuals_c = []

dx = xL/(mI - 1)
dy = yL/(mJ - 1)

# Fill the coordinates
for i in range(mI):
    for j in range(mJ):
        # For the mesh points
        xCoords_M[i,j] = i*dx
        yCoords_M[i,j] = j*dy

        # For the nodes
        if i > 0:
            xCoords_N[i,j] = 0.5*(xCoords_M[i,j] + xCoords_M[i-1,j])
        if i == mI-1 and j>0:
            yCoords_N[i+1,j] = 0.5*(yCoords_M[i,j] + yCoords_M[i,j-1])
        if j > 0:
            yCoords_N[i,j] = 0.5*(yCoords_M[i,j] + yCoords_M[i,j-1])
        if j == mJ-1 and i>0:
            xCoords_N[i,j+1] = 0.5*(xCoords_M[i,j] + xCoords_M[i-1,j])

        # Fill dx_CV and dy_CV
        if i > 0:
            dx_CV[i,j] = xCoords_M[i,j] - xCoords_M[i-1,j]
        if j > 0:
            dy_CV[i,j] = yCoords_M[i,j] - yCoords_M[i,j-1]

xCoords_N[-1,:] = xL
yCoords_N[:,-1] = yL


# Fill dxe, dxw, dyn and dys
for i in range(1,nI-1):
    for j in range(1,nJ-1):
        dxe_N[i,j] = xCoords_N[i+1,j] - xCoords_N[i,j]
        dxw_N[i,j] = xCoords_N[i,j] - xCoords_N[i-1,j]
        dyn_N[i,j] = yCoords_N[i,j+1] - yCoords_N[i,j]
        dys_N[i,j] = yCoords_N[i,j] - yCoords_N[i,j-1]
        
        
#Variables
for i in range(1,nI-1):     
    for j in range(1,nJ-1):                                                        
        fxe[i,j] = 0.5*dx_CV[i,j]/dxe_N[i,j]
        fxw[i,j] = 0.5*dx_CV[i,j]/dxw_N[i,j]
        fyn[i,j] = 0.5*dy_CV[i,j]/dyn_N[i,j]
        fys[i,j] = 0.5*dy_CV[i,j]/dys_N[i,j]
        
        U[i,-1] = 1     #Boundary condition

for i in range(1,nI-1):                                            
    for j in range(1,nJ-1):
        D[i,j,0] = nu*rho*dy_CV[i,j]/dxe_N[i,j]
        D[i,j,1] = nu*rho*dy_CV[i,j]/dxw_N[i,j]
        D[i,j,2] = nu*rho*dx_CV[i,j]/dyn_N[i,j]
        D[i,j,3] = nu*rho*dx_CV[i,j]/dys_N[i,j]

# Looping
for iter in range(nIterations):
    
    ## Compute coefficients for inner nodes
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            coeffsUV[i,j,0] = D[i,j,0] + max(0 , -F[i,j,0])   #aEu
            coeffsUV[i,j,1] = D[i,j,1] + max(F[i,j,1] , 0)    #aWu
            coeffsUV[i,j,2] = D[i,j,2] + max(0 , -F[i,j,2])   #aNu
            coeffsUV[i,j,3] = D[i,j,3] + max(F[i,j,3] , 0)    #aSu
            
            Spu[i,j] = - F[i,j,0] + F[i,j,1] - F[i,j,2] + F[i,j,3]                                      # Spu            
            sourceUV[i,j,0] = -((P[i+1,j]-P[i-1,j])/(dxe_N[i,j]+dxw_N[i,j]) + b)*dx_CV[i,j]*dy_CV[i,j]  #Source U      
            sourceUV[i,j,1] = -((P[i,j+1]-P[i,j-1])/(dyn_N[i,j]+dys_N[i,j]) + b)*dy_CV[i,j]*dx_CV[i,j]  #Source V                     
            
            #Correction terms
            sourceUV[i,j,0] = sourceUV[i,j,0] + max(Spu[i,j], 0)*U[i,j]                 #Correction soure U
            sourceUV[i,j,1] = sourceUV[i,j,1] + max(Spu[i,j], 0)*V[i,j]                 #Correction source V
                       
            Spu[i,j] = -max(-Spu[i,j],0)                                                #Correction Sp
            
            coeffsUV[i,j,4] = coeffsUV[i,j,0] + coeffsUV[i,j,1] + coeffsUV[i,j,2] + \
                              coeffsUV[i,j,3] - Spu[i,j]                                #Correction aPu
            
    ## Introduce implicit under-relaxation for U and V
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            coeffsUV[i,j,4] = coeffsUV[i,j,4] / alphaUV
            sourceUV[i,j,0] = sourceUV[i,j,0] + (1 - alphaUV) * coeffsUV[i,j,4] * U[i,j]
            sourceUV[i,j,1] = sourceUV[i,j,1] + (1 - alphaUV) * coeffsUV[i,j,4] * V[i,j]
        
    ## Solve for U and V using Gauss-Seidel                                                                                    
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            U[i,j] = (coeffsUV[i,j,0]*U[i+1,j] + coeffsUV[i,j,1]*U[i-1,j] + coeffsUV[i,j,2]*U[i,j+1]+ \
                      coeffsUV[i,j,3]*U[i,j-1] + sourceUV[i,j,0])/ coeffsUV[i,j,4]

            V[i,j] = (coeffsUV[i,j,0]*V[i+1,j] + coeffsUV[i,j,1]*V[i-1,j] + coeffsUV[i,j,2]*V[i,j+1]+ \
                      coeffsUV[i,j,3]*V[i,j-1] + sourceUV[i,j,1])/ coeffsUV[i,j,4]    
    
    ## Update face fluxes
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            F[i,j,0] = rho*((1-fxe[i,j])*U[i,j]+fxe[i,j]*U[i+1,j])*dy_CV[i,j]   #East     
            F[i,j,2] = rho*((1-fyn[i,j])*V[i,j]+fyn[i,j]*V[i,j+1])*dx_CV[i,j]   #North            

    ## Calculate at the faces using Rhie-Chow for the face velocities                                                         
    for i in range(1,nI-1):
        for j in range(1,nJ-1):  
            if i < nI-2:
                F[i,j,0] = F[i,j,0] + rho*dy_CV[i,j]*(-(1-fxe[i,j])*(-(P[i+1,j] - P[i-1,j]) /(dxw_N[i,j] + dxe_N[i,j])) * dx_CV[i,j]/coeffsUV[i,j,4] - \
                                        fxe[i,j] * (-(P[i+2,j] - P[i,j]) / (dxw_N[i+1,j] + dxe_N[i+1,j])) * dx_CV[i+1,j]/coeffsUV[i+1,j,4] + \
                                        ((1-fxe[i,j])*dx_CV[i,j] / coeffsUV[i,j,4] + fxe[i,j]*dx_CV[i+1,j]/coeffsUV[i+1,j,4]) * \
                                        (P[i,j]-P[i+1,j]) / dxe_N[i,j])     #East
            if j < nJ-2:
                F[i,j,2] = F[i,j,2] + rho*dy_CV[i,j]*(-(1-fyn[i,j])*(-(P[i,j+1] - P[i,j-1]) /(dys_N[i,j] + dyn_N[i,j])) * dy_CV[i,j]/coeffsUV[i,j,4] - \
                                        fyn[i,j] * (-(P[i,j+2] - P[i,j]) / (dys_N[i,j+1] + dyn_N[i,j+1])) * dy_CV[i,j+1]/coeffsUV[i,j+1,4] + \
                                        ((1-fyn[i,j])*dy_CV[i,j] / coeffsUV[i,j,4] + fyn[i,j]*dy_CV[i,j+1]/coeffsUV[i,j+1,4]) * \
                                        (P[i,j]-P[i,j+1]) / dyn_N[i,j])     #North
                    
    ## Similar for west and south face fluxes
    for i in range(1,nI-1):
        for j in range(1,nJ-1):    
            
            F[i,j,1] = F[i-1,j,0]            #West
            F[i,j,3] = F[i,j-1,2]            #South
            
            if i == 1:
                F[i,j,1] = rho*(fxw[i,j]*U[i-1,j] + (1-fxw[i,j])*U[i,j])*dy_CV[i,j]        #West
            if j == 1:
                F[i,j,3] = rho*(fys[i,j]*V[i,j-1] + (1-fys[i,j])*V[i,j])*dx_CV[i,j]        #South
    
    # Force global continuity
    F[nI-2,:,0] = F[1,:,1]
    F[:,nJ-2,2] = F[:,1,3]
                    
    ## Calculate pressure correction equation coefficients    
    for i in range(1,nI-1):                                                                  
        for j in range(1,nJ-1):                                                                                                              
            dPp[i,j,0] = dy_CV[i,j] / ( (1-fxe[i,j])*coeffsUV[i,j,4] + fxe[i,j]*coeffsUV[i+1,j,4])
            dPp[i,j,1] = dy_CV[i,j] / (fxw[i,j]*coeffsUV[i-1,j,4] + (1-fxw[i,j])*coeffsUV[i,j,4])                       
            dPp[i,j,2] = dx_CV[i,j] / ( (1-fyn[i,j])*coeffsUV[i,j,4] + fyn[i,j]*coeffsUV[i,j+1,4])
            dPp[i,j,3] = dx_CV[i,j] / (fys[i,j]*coeffsUV[i,j-1,4] + (1-fys[i,j])*coeffsUV[i,j,4])    
                                                                                                  
            if i == nI-2: 
                dPp[i,j,0] = 0
            if i == 1:  
                dPp[i,j,1] = 0
            if j == nJ-2:    
                dPp[i,j,2] = 0
            if j == 1:    
                dPp[i,j,3] = 0
                                           
            coeffsPp[i,j,0] = rho*dPp[i,j,0]                          #aE                                                   
            coeffsPp[i,j,1] = rho*dPp[i,j,1]                          #aW                        
            coeffsPp[i,j,2] = rho*dPp[i,j,2]                          #aN   
            coeffsPp[i,j,3] = rho*dPp[i,j,3]                          #aS   
            
            sourcePp[i,j] = F[i,j,1] - F[i,j,0] + F[i,j,3] - F[i,j,2] #Source P for pressure correction
            Spp[i,j] = 0 
            
            coeffsPp[i,j,4] = coeffsPp[i,j,0] + coeffsPp[i,j,1] + coeffsPp[i,j,2] + coeffsPp[i,j,3] - Spp[i,j] #aP
            
    # Solve for pressure correction (Note that more that one loop is used)
    for iter_gs in range(n_inner_iterations_gs):                                                                                            
        for j in range(1,nJ-1):
            for i in range(1,nI-1):    
                Pp[i,j]= (coeffsPp[i,j,0] * Pp[i+1,j] + coeffsPp[i,j,1] * Pp[i-1,j] + \
                          coeffsPp[i,j,2] * Pp[i,j+1] + coeffsPp[i,j,3] * Pp[i,j-1] + \
                              sourcePp[i,j]) / coeffsPp[i,j,4]
    
    # Set Pp with reference to node (1,1) and copy to boundaries                                                                                    
    pref = Pp[1,1]
    Pp = Pp - pref
    
    Pp[0,:] = Pp[1,:]
    Pp[-1,:] = Pp[-2,:]
    Pp[:,0] = Pp[:,1]
    Pp[:,-1] = Pp[:,-2]
    
    P = P + Pp*alphaP   #New pressure P
    
    # Correct velocities, pressure and mass flows
    ## Correct (explicitly underrelaxed) pressure:
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            
            # Calculate center pressure gradient
            pGrad[i,j,0] = (P[i+1,j] - P[i,j]) / dxe_N[i,j] #East pressure gradient
            pGrad[i,j,1] = (P[i,j+1] - P[i,j]) / dyn_N[i,j] #North pressure gradient
            
            # Constant gradient extrapolation                                                                                               
            if i == 1:
                P[i-1,j] = P[i,j] - pGrad[i,j,0]*dxw_N[i,j]
            if j == 1:
                P[i,j-1] = P[i,j] - pGrad[i,j,1]*dys_N[i,j]
            if i == nI-2:
                P[i+1,j] = P[i,j] + pGrad[i,j,0]*dxe_N[i,j]
            if j == nJ-2:
                P[i,j+1] = P[i,j] + pGrad[i,j,1]*dyn_N[i,j]
    
    # New pressure for east, west, north and south at the faces
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            Pp_e[i,j] = (1-fxe[i,j])*Pp[i,j] + fxe[i,j]*Pp[i+1,j]
            Pp_w[i,j] = fxw[i,j]*Pp[i-1,j] + (1 - fxw[i,j])*Pp[i,j]
            Pp_n[i,j] = (1-fyn[i,j])*Pp[i,j] + fyn[i,j]*Pp[i,j+1]
            Pp_s[i,j] = fys[i,j]*Pp[i,j-1] + (1 - fys[i,j])*Pp[i,j]
            
    # Boundarys for the new pressures at the faces
    for i in range(1,nI-1):
        for j in range(1,nJ-1):            
            Pp_e[-2,j] = Pp[-1,j]
            Pp_w[1,j] = Pp[0,j]
            Pp_n[i,-2] = Pp[i,-1]
            Pp_s[i,1] = Pp[i,0]
    
    # Velocity correction
    for i in range(1,nI-1):
        for j in range(1,nJ-1):    
            U[i,j] = U[i,j] + (Pp_w[i,j] - Pp_e[i,j]) / coeffsUV[i,j,4] *dy_CV[i,j]
            V[i,j] = V[i,j] + (Pp_s[i,j] - Pp_n[i,j]) / coeffsUV[i,j,4] *dx_CV[i,j]   
                
    # Diffusion correction            
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            F[i,j,0] = rho*((1-fxe[i,j])*U[i,j]+fxe[i,j]*U[i+1,j])*dy_CV[i,j]   #East     
            F[i,j,2] = rho*((1-fyn[i,j])*V[i,j]+fyn[i,j]*V[i,j+1])*dx_CV[i,j]   #North       
    
    ## Calculate at the faces using Rhie-Chow        
    for i in range(1,nI-1):
        for j in range(1,nJ-1):  
            if i < nI-2:
                F[i,j,0] = F[i,j,0] + rho*dy_CV[i,j]*(-(1-fxe[i,j])*(-(P[i+1,j] - P[i-1,j]) /(dxw_N[i,j] + dxe_N[i,j])) * dx_CV[i,j]/coeffsUV[i,j,4] - \
                                        fxe[i,j] * (-(P[i+2,j] - P[i,j]) / (dxw_N[i+1,j] + dxe_N[i+1,j])) * dx_CV[i+1,j]/coeffsUV[i+1,j,4] + \
                                        ((1-fxe[i,j])*dx_CV[i,j] / coeffsUV[i,j,4] + fxe[i,j]*dx_CV[i+1,j]/coeffsUV[i+1,j,4]) * \
                                        (P[i,j]-P[i+1,j]) / dxe_N[i,j])   #East  
            if j < nJ-2:
                F[i,j,2] = F[i,j,2] + rho*dy_CV[i,j]*(-(1-fyn[i,j])*(-(P[i,j+1] - P[i,j-1]) /(dys_N[i,j] + dyn_N[i,j])) * dy_CV[i,j]/coeffsUV[i,j,4] - \
                                        fyn[i,j] * (-(P[i,j+2] - P[i,j]) / (dys_N[i,j+1] + dyn_N[i,j+1])) * dy_CV[i,j+1]/coeffsUV[i,j+1,4] + \
                                        ((1-fyn[i,j])*dy_CV[i,j] / coeffsUV[i,j,4] + fyn[i,j]*dy_CV[i,j+1]/coeffsUV[i,j+1,4]) * \
                                        (P[i,j]-P[i,j+1]) / dyn_N[i,j])   #North  
    
    ### Similar for west and south face fluxes
    for i in range(1,nI-1):
        for j in range(1,nJ-1):                
            F[i,j,1] = F[i-1,j,0]            #West
            F[i,j,3] = F[i,j-1,2]            #South
            
            if i == 1:
                F[i,j,1] = rho*(fxw[i,j]*U[i-1,j] + (1-fxw[i,j])*U[i,j])*dy_CV[i,j]        #West
            if j == 1:
                F[i,j,3] = rho*(fys[i,j]*V[i,j-1] + (1-fys[i,j])*V[i,j])*dx_CV[i,j]        #South
    
    
    # Compute residuals
    Fnorm = 1         #Normalization
    epsU = 0          #U
    epsV = 0          #V
    epsC = 0          #Continuity
    
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            epsU = epsU + 1/Fnorm * abs(coeffsUV[i,j,4]*U[i,j] - (coeffsUV[i,j,0]*U[i+1,j] + coeffsUV[i,j,1]*U[i-1,j] + coeffsUV[i,j,2]*U[i,j+1] + coeffsUV[i,j,3]*U[i,j-1] + sourceUV[i,j,0] ))
            epsV = epsV + 1/Fnorm * abs(coeffsUV[i,j,4]*V[i,j] - (coeffsUV[i,j,0]*V[i+1,j] + coeffsUV[i,j,1]*V[i-1,j] + coeffsUV[i,j,2]*V[i,j+1] + coeffsUV[i,j,3]*V[i,j-1] + sourceUV[i,j,1] ))
            epsC = epsC + 1/Fnorm * abs(-F[i,j,0] + F[i,j,1] -F[i,j,2]+F[i,j,3])
            
    residuals_U.append(epsU)                #U momentum residual
    residuals_V.append(epsV)                #V momentum residual
    residuals_c.append(epsC)                #Continuity residual


    print('iteration: %d\nresU = %.5e, resV = %.5e, resCon = %.5e\n\n'\
        % (iter, residuals_U[-1], residuals_V[-1], residuals_c[-1]))
    
    #  Check convergence
    if resTolerance>max([residuals_U[-1], residuals_V[-1], residuals_c[-1]]):
        break

#%% Plotting section 
font = {'fontname':'Times New Roman'}
xv, yv = np.meshgrid(xCoords_N, yCoords_N)
plt.rcParams['font.size'] = '10' #23 for small pictures, 10 for big pictures

# Plot mesh
plt.figure()
plt.plot(xCoords_M, yCoords_M,color='black',linewidth=0.5)
plt.plot(np.transpose(xCoords_M), np.transpose(yCoords_M),color='black',linewidth=0.5)
plt.xlabel('x [m]',**font)
plt.ylabel('y [m]',**font)
plt.title('Computational mesh',**font)


# Plot results
## U velocity
plt.figure()
plt.subplot(2,2,1)
contour = plt.contourf(xCoords_N, yCoords_N, U, cmap='jet', levels=150)
for c in contour.collections:
    c.set_edgecolor('face')
plt.colorbar(label="$[m/s]$")
plt.title('U velocity [m/s]',**font)
plt.xlabel('x [m]',**font)
plt.ylabel('y [m]',**font)


## V velocity
plt.subplot(2,2,2)
contour = plt.contourf(xCoords_N, yCoords_N, V, cmap='viridis', levels=150)
for c in contour.collections:
    c.set_edgecolor('face')
plt.colorbar(label="$[m/s]$")
plt.title('V velocity [m/s]',**font)
plt.xlabel('x [m]',**font)
plt.ylabel('y [m]',**font)


## P pressure
plt.subplot(2,2,3)
contour = plt.contourf(xCoords_N, yCoords_N, P, cmap='hot', levels=150)
for c in contour.collections:
    c.set_edgecolor('face')
plt.colorbar(label="$[Pa]$")
plt.title('Pressure [Pa]',**font)
plt.xlabel('x [m]',**font)
plt.ylabel('y [m]',**font)


# Vector plot
plt.subplot(2,2,4)
plt.quiver(xCoords_N, yCoords_N, U, V)
plt.title('Vector plot of the velocity field',**font)
plt.xlabel('x [m]',**font)
plt.ylabel('y [m]',**font)
plt.show()


# Comparison with data
plt.figure()
data=np.genfromtxt(data_file, skip_header=1)
uInterp = np.zeros((nJ-2,1))
vInterp = np.zeros((nJ-2,1))
for j in range(1,nJ-1):
    for i in range(1,nI-1):
        if xCoords_N[i,j]<0.5 and xCoords_N[i+1,j]>0.5:
            uInterp[j-1] = (U[i+1,j] + U[i,j])*0.5
            vInterp[j-1] = (V[i+1,j] + V[i,j])*0.5
            break
        elif abs(xCoords_N[i,j]-0.5) < 0.000001:
            uInterp[j-1] = U[i,j]
            vInterp[j-1] = V[i,j]
            break

plt.plot(data[:,0],data[:,2],'r.',markersize=20,label='data U')
plt.plot(data[:,1],data[:,2],'b.',markersize=20,label='data V')
plt.plot(uInterp,yCoords_N[1,1:-1],'k',label='sol U')
plt.plot(vInterp,yCoords_N[1,1:-1],'g',label='sol V')
plt.title('Comparison with data at x = 0.5',**font)
plt.xlabel('u, v [m/s]',**font)
plt.ylabel('y [m]',**font)
plt.legend()
plt.show()


#Residual
plt.figure()
plt.title('Residual convergence',**font)
plt.yscale('log')
plt.plot(residuals_U)
plt.plot(residuals_V)
plt.plot(residuals_c)
plt.xlabel('iterations',**font)
plt.ylabel('residuals [-]',**font)
plt.legend(['U momentum','V momentum', 'Continuity'])
plt.show()


