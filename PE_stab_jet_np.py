import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sp
from scipy.misc import factorial
import scipy.linalg as spalg
from scipy.sparse.linalg import eigs
import time
import os

rank = PETSc.COMM_WORLD.Get_rank()
Print = PETSc.Sys.Print
size = PETSc.COMM_WORLD.Get_size()

def cheb(N):
    if N == 0:
        D = 0; x = 1;
    else:
        x = np.cos(np.pi*np.array(range(0,N+1))/N).reshape([N+1,1])
        c = np.ravel(np.vstack([2, np.ones([N-1,1]), 2])) \
            *(-1)**np.ravel(np.array(range(0,N+1)))
        c = c.reshape(c.shape[0],1)
        X = np.tile(x,(1,N+1))
        dX = X-(X.conj().transpose())
        D  = (c*(1/c).conj().transpose())/(dX+(np.eye(N+1)))   # off-diagonal entries
        D  = D - np.diag(np.sum(D,1))   # diagonal entries
    return D,x

def index(A,ix,iy):
    M = A[ix,:]
    M = M[:,iy]
    return M

## Numerical Grid
Ny=10
Nz=3

## Physical Parameters
f  = 1e-4
fs = 1e-4
N2 = 1e-6
g  = 10
nu = 0e1
Ro = 1
Fr = 1

## Length Scales and Jet Strength
Ly = 1e5
Lz = 1e3
Ljet = 1e4
Ujet = Ro*f*Ljet
# Re = Ujet*Ljet/nu

## Range of Wavenumbers
# kk=3*pi*[10]*1e-6
kk = np.array([8.9535e-05])

## Define Differentiation Matrices for y and z
D1,x = cheb(Ny)
y  = x*Ly/2
Dy = 2/Ly*D1
D2y= np.dot(Dy,Dy)

D2,x = cheb(Nz)
z  = x*Lz/2
Dz = 2/Lz*D2
D2z= np.dot(Dz,Dz)

## Define Identity matrices
Iy = np.eye(Ny-1)
Iz = np.eye(Nz-1)
Iyp = np.eye(Ny+1)
Izp = np.eye(Nz+1)
I = np.eye((Ny+1)*(Nz+1))
Dyv = np.kron(Dy,Izp)
D2yv = np.kron(D2y,Izp)
Dzv = np.kron(Iyp,Dz)
D2zv = np.kron(Iyp, D2z)

## Define Cartesian Grid of (y,z)
yy,zz = np.meshgrid(y,z)
yv = np.ravel(yy,order='F')
zv = np.ravel(zz,order='F')

## Background State
U0   = Ujet*(1/np.cosh(y/Ljet))**2
DU0  = np.ravel(np.dot(Dy,U0))
D2U0 = np.ravel(np.dot(D2y,U0))
R0   = fs/g*U0
DR0  = np.ravel(np.dot(Dy,R0))

# matrix functions
U0m    = np.diag(np.ravel(U0[1:Ny]))
DU0m   = np.diag(DU0[1:Ny])
D2U0m  = np.diag(D2U0[1:Ny])
DU0fm  = np.diag(DU0[1:Ny] - f)
DU02fm = np.diag(2*DU0[1:Ny] - f)
R0m    = np.diag(np.ravel(R0[1:Ny]))
DR0m   = np.diag(DR0[1:Ny])
DU02fmp= np.diag(2*DU0-f)

## Find Boundary points

Zbd = np.where(abs(z)==Lz/2)[0]
Zin = np.where(abs(z)!=Lz/2)[0]
Nbc = nlg.solve(index(-Dz,Zbd,Zbd),index(Dz,Zbd,Zin))

Ybd = np.where((abs(yv)==Ly/2) & (abs(zv)!=Lz/2))[0]
Zbd = np.where((abs(zv)==Lz/2) & (abs(yv)!=Ly/2))[0]
In  = np.where((abs(zv)!=Lz/2) & (abs(yv)!=Ly/2))[0]
Corn= np.where((abs(zv)==Lz/2) & (abs(yv)==Ly/2))[0]


Nbcy = nlg.solve(index(-Dyv,Ybd,Ybd), index(Dyv,Ybd,In))
Nbcz = nlg.solve(index(-Dzv,Zbd,Zbd), index(Dzv,Zbd,In))
NbcCorn0 = np.hstack([Nbc,0*Nbc])
NbcCorn1 = np.hstack([0*Nbc,Nbc])
NbcCorn = np.vstack([NbcCorn0,NbcCorn1])
NbcCorn = np.dot(NbcCorn,Nbcy)

Dyvne = Dyv[:,In] + np.dot(Dyv[:,Ybd],Nbcy)
Dzvne = Dzv[:,In] + np.dot(Dzv[:,Zbd],Nbcz)

D2yvne = index(D2yv,In,In) + np.dot(index(D2yv,In,Ybd),Nbcy)
D2zvne = index(D2zv,In,In) + np.dot(index(D2zv,In,Zbd),Nbcz)

Iyz = np.eye((Ny-1)*(Nz-1))
Zyz = np.zeros([(Ny-1)*(Nz-1),(Ny-1)*(Nz-1)])

Ipy = np.eye(2*(Nz-1), (Ny-1)*(Nz-1))
Ipz = np.eye(2*(Ny-1), (Ny-1)*(Nz-1))
Zpy = np.zeros([2*(Nz-1),(Ny-1)*(Nz-1)])
Zpz = np.zeros([2*(Ny-1),(Ny-1)*(Nz-1)])

## Loop over the wavenumbers
savemodes = 1e0
eigVal = np.zeros([savemodes,len(kk)],dtype=np.complex128)
eigVec = np.zeros([4*(Ny-1)*(Nz-1),savemodes,len(kk)],dtype=np.complex128)
count = 0

for k in kk[:]:

    k = kk
    ## Define wavenumbers
    ik = k*1j
    k2 = k**2

    ## Equation and BCs for \hat p
    L1 = k2*I - D2yv - D2zv
    L1[Ybd,:] = Dyv[Ybd,:]
    L1[Zbd,:] = Dzv[Zbd,:]
    L1[Corn,:] = Dyv[Corn,:]

    M1 = np.hstack([f*Dyvne-fs*Dzvne, \
                    ik*np.kron(DU02fmp[:,1:Ny],Izp[:,1:Nz]), \
                    ik*np.kron(fs*Iyp[:,1:Ny],Izp[:,1:Nz]), \
                    g*Dzvne])

    M1[Ybd,:] = np.hstack([-f*Nbcy,Zpy,Zpy,Zpy])
    M1[Zbd,:] = np.hstack([fs*Nbcz,Zpz,Zpz,-g*Nbcz])
    M1[Corn,:]= np.hstack([-f*NbcCorn,Zpy[0:4,:],Zpy[0:4,:],Zpy[0:4,:]])
    # print sp.csr_matrix(M1)
    M1full = nlg.solve(L1,M1)

    ## Contruct Matrix to Analyze
    LapU = k2*Iyz - D2yvne - D2zvne
    LapV = k2*Iyz - index(D2yv,In,In) - D2zvne
    LapW = k2*Iyz - D2yvne - index(D2zv,In,In)

    kronU0    = np.kron(U0m,Iz)
    kronDU0fm = np.kron(DU0fm,Iz)
    kronDR0m  = np.kron(DR0m,Iz)

    A0 = np.hstack([ik*kronU0 + nu*LapU, kronDU0fm, fs*Iyz, Zyz])
    A1 = np.hstack([f*Iyz, ik*kronU0 + nu*LapV, Zyz, Zyz])
    A2 = np.hstack([-fs*Iyz, Zyz, ik*kronU0 + nu*LapW, g*Iyz])
    A3 = np.hstack([Zyz, kronDR0m, -N2*Iyz, ik*kronU0])
    A = np.vstack([A0,A1,A2,A3])

    A = -A - np.vstack([ik*M1full[In,:], np.dot(Dyv[In,:],M1full),
        np.dot(Dzv[In,:],M1full), np.zeros([len(In), 4*len(In)]) ])

    # Using eig
    eVals, eVecs = nlg.eig(A)
    ind = (-np.real(eVals)).argsort() #get indices in descending
    eVecs = eVecs[:,ind]
    eVals = eVals[ind]

    eigVal[0:savemodes,count] = eVals[0:savemodes]
    eigVec[:,0:savemodes,count] = eVecs[:,0:savemodes]
    count = count+1

    grmax,grind = np.real(eVals[0]).max(0), np.real(eVals[0]).argmax(0)
    print "Largest growth rate is: ", grmax, "at wavenumber ", k

    fcn = eigVec[0:(Ny-1)*(Nz-1),0]
    ueig = np.zeros((Ny+1)*(Nz+1),dtype=np.complex128)
    ueig[Ybd] = np.dot(Nbcy,fcn)
    ueig[In] = fcn
    ueig[Zbd] = np.dot(Nbcz,fcn)
    ueig[Corn] = np.dot(NbcCorn,fcn)
    ueig = np.reshape(ueig,[Nz+1,Ny+1])

    fcn = eigVec[(Ny-1)*(Nz-1):2*(Ny-1)*(Nz-1),0]
    veig = np.zeros((Ny+1)*(Nz+1),dtype=np.complex128)
    veig[Ybd] =  np.dot(0,Ybd)
    veig[In] = fcn
    veig[Zbd] = np.dot(Nbcz,fcn)
    veig[Corn] = np.dot(0,Corn)
    veig = np.reshape(veig,[Nz+1,Ny+1])

    fcn = eigVec[2*(Ny-1)*(Nz-1):3*(Ny-1)*(Nz-1),0]
    weig = np.zeros((Ny+1)*(Nz+1),dtype=np.complex128)
    weig[Ybd] =   np.dot(Nbcy, fcn)
    weig[In] = fcn
    weig[Zbd] = np.dot(0, Zbd)
    weig[Corn] = np.dot(0, Corn)
    weig = np.reshape(weig,[Nz+1,Ny+1])

    fcn =eigVec[3*(Ny-1)*(Nz-1):4*(Ny-1)*(Nz-1),0]
    rhoeig = np.zeros((Ny+1)*(Nz+1),dtype=np.complex128)
    rhoeig[Ybd] =  np.dot(Nbcy, fcn)
    rhoeig[In] = fcn
    rhoeig[Zbd] = np.dot(Nbcz, fcn)
    rhoeig[Corn] = np.dot(NbcCorn, fcn)
    rhoeig = np.reshape(rhoeig,[Nz+1,Ny+1])

    peig = np.reshape(np.dot(M1full,eigVec[:,0]),[Nz+1,Ny+1])
    
    fig = plt.figure()
    plt.subplot(2,2,1)
    plt.contourf((ueig.real).conj().transpose())
    plt.colorbar()
    plt.title('u structure')
    plt.subplot(2,2,2)
    plt.contourf((veig.real).conj().transpose())
    plt.colorbar()
    plt.title('v structure')
    plt.subplot(2,2,3)
    plt.contourf((weig.real).conj().transpose())
    plt.colorbar()
    plt.title('w structure')
    plt.subplot(2,2,4)
    plt.contourf((rhoeig.real).conj().transpose())
    plt.colorbar()
    plt.title('rho structure')
    plt.savefig("PE_stab_jet.eps", format='eps', dpi=1000)
    plt.show()
