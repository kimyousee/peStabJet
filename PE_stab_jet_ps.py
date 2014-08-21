import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
import os

rank = PETSc.COMM_WORLD.Get_rank()
Print = PETSc.Sys.Print
size = PETSc.COMM_WORLD.Get_size()

# This code uses petsc4py/slepc4py
# Comments with 3 hashes ### are suggestions on how to convert to FD (scipy.sparse)
# Might also have to add .todense() to some places

def main():
    opts = PETSc.Options()

    Ny = opts.getInt('Ny', 10)
    Nz = opts.getInt('Nz', 3)
    nEV = opts.getInt('nev', 300)

    ### diff = 'cheb' #or 'FD'

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
    ### if diff == 'cheb':
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

    ## Background State
    U0   = Ujet*(1/np.cosh(y/Ljet))**2
    DU0  = np.ravel(np.dot(Dy,U0))
    D2U0 = np.ravel(np.dot(D2y,U0))
    R0   = fs/g*U0
    DR0  = np.ravel(np.dot(Dy,R0))

    # matrix functions
    U0m    = np.diag(np.ravel(U0[1:Ny])) #Ny-1 ^2
    DU0m   = np.diag(DU0[1:Ny]) #Ny-1 ^2
    D2U0m  = np.diag(D2U0[1:Ny]) #Ny-1 ^2
    DU0fm  = np.diag(DU0[1:Ny] - f) 
    DU02fm = np.diag(2*DU0[1:Ny] - f)
    R0m    = np.diag(np.ravel(R0[1:Ny]))
    DR0m   = np.diag(DR0[1:Ny])
    DU02fmp= np.diag(2*DU0-f)
    ### if diff =='FD':
    ### same as above but change all np in eye,kron,diag to sp
    ### change cheb to fd2 and all np.dot to *

    ## Define Cartesian Grid of (y,z)
    yy,zz = np.meshgrid(y,z)
    yv = np.ravel(yy,order='F')
    zv = np.ravel(zz,order='F')


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
    ### change all stacks to be sp (ex: sp.hstack(...))
    NbcCorn0 = np.hstack([Nbc,0*Nbc])
    NbcCorn1 = np.hstack([0*Nbc,Nbc])
    NbcCorn = np.vstack([NbcCorn0,NbcCorn1])
    NbcCorn = np.dot(NbcCorn,Nbcy) ### change np.dot to *

    Dyvne = Dyv[:,In] + np.dot(Dyv[:,Ybd],Nbcy)
    Dzvne = Dzv[:,In] + np.dot(Dzv[:,Zbd],Nbcz)

    D2yvne = index(D2yv,In,In) + np.dot(index(D2yv,In,Ybd),Nbcy)
    D2zvne = index(D2zv,In,In) + np.dot(index(D2zv,In,Zbd),Nbcz)

    ### change all np.eye to sp.eye and np.zeros([...]) to sp.csr_matrix((...)) or sp.lil_matrix((...))
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

        ### np.hstakc -> sp.hstack, sp.kron
        M1 = np.hstack([f*Dyvne-fs*Dzvne, \
                        ik*np.kron(DU02fmp[:,1:Ny],Izp[:,1:Nz]), \
                        ik*np.kron(fs*Iyp[:,1:Ny],Izp[:,1:Nz]), \
                        g*Dzvne])

        M1[Ybd,:] = np.hstack([-f*Nbcy,Zpy,Zpy,Zpy])
        M1[Zbd,:] = np.hstack([fs*Nbcz,Zpz,Zpz,-g*Nbcz])
        M1[Corn,:]= np.hstack([-f*NbcCorn,Zpy[0:4,:],Zpy[0:4,:],Zpy[0:4,:]])
        M1full = nlg.solve(L1,M1)

        ## Contruct Matrix to Analyze
        LapU = k2*Iyz - D2yvne - D2zvne
        LapV = k2*Iyz - index(D2yv,In,In) - D2zvne
        LapW = k2*Iyz - D2yvne - index(D2zv,In,In)
        ### sp.kron
        kronU0    = np.kron(U0m,Iz)
        kronDU0fm = np.kron(DU0fm,Iz)
        kronDR0m  = np.kron(DR0m,Iz)

        A = build_A(Ny,Nz,ik,kronU0,nu,LapU,kronDU0fm,fs,f,LapV,LapW,g,kronDR0m,N2,M1full,In,Dyv,Dzv)

        E = SLEPc.EPS(); E.create(comm=SLEPc.COMM_WORLD)
        E.setOperators(A); E.setDimensions(nEV,SLEPc.DECIDE)
        E.setProblemType(SLEPc.EPS.ProblemType.NHEP);E.setFromOptions()
        E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        # E.setTolerances(1e-6,max_it=25)

        t0 = time.time()
        E.solve()
        Print("Time for solve:", time.time()-t0)

        count = count+1

        nconv = E.getConverged()
        vr, wr = A.getVecs()
        vi, wi = A.getVecs()
        if nconv <= nEV: evals = nconv
        else: evals = nEV
        eigVecs = np.empty([vr.getSize(),evals],dtype='complex')
        eigVals = np.empty([evals],dtype='complex')

        for i in xrange(evals):
            eigVal = E.getEigenvalue(i)
            eigVals[i] = eigVal

            E.getEigenvector(i,vr,vi)
            
            # Put all values of vr into 1 processor
            scatter, vrSeq = PETSc.Scatter.toZero(vr)
            im = PETSc.InsertMode.INSERT_VALUES
            sm = PETSc.ScatterMode.FORWARD
            scatter.scatter(vr,vrSeq,im,sm)

            if rank == 0:
                # store eigenvector in numpy array
                for j in range(0,vrSeq.getSize()):
                    eigVecs[j,i] = vrSeq[j].real+vrSeq[j].imag*1j

        grmax,grind = np.real(eigVals[0]).max(0), np.real(eigVals[0]).argmax(0)
        print "Largest growth rate is: ", grmax, "at wavenumber ", k

        fcn = eigVecs[0:(Ny-1)*(Nz-1),0]
        ueig = np.zeros((Ny+1)*(Nz+1),dtype=np.complex128)
        ueig[Ybd] = np.dot(Nbcy,fcn)
        ueig[In] = fcn
        ueig[Zbd] = np.dot(Nbcz,fcn)
        ueig[Corn] = np.dot(NbcCorn,fcn)
        ueig = np.reshape(ueig,[Nz+1,Ny+1])

        fcn = eigVecs[(Ny-1)*(Nz-1):2*(Ny-1)*(Nz-1),0]
        veig = np.zeros((Ny+1)*(Nz+1),dtype=np.complex128)
        veig[Ybd] =  np.dot(0,Ybd)
        veig[In] = fcn
        veig[Zbd] = np.dot(Nbcz,fcn)
        veig[Corn] = np.dot(0,Corn)
        veig = np.reshape(veig,[Nz+1,Ny+1])

        fcn = eigVecs[2*(Ny-1)*(Nz-1):3*(Ny-1)*(Nz-1),0]
        weig = np.zeros((Ny+1)*(Nz+1),dtype=np.complex128)
        weig[Ybd] =   np.dot(Nbcy, fcn)
        weig[In] = fcn
        weig[Zbd] = np.dot(0, Zbd)
        weig[Corn] = np.dot(0, Corn)
        weig = np.reshape(weig,[Nz+1,Ny+1])

        fcn =eigVecs[3*(Ny-1)*(Nz-1):4*(Ny-1)*(Nz-1),0]
        rhoeig = np.zeros((Ny+1)*(Nz+1),dtype=np.complex128)
        rhoeig[Ybd] =  np.dot(Nbcy, fcn)
        rhoeig[In] = fcn
        rhoeig[Zbd] = np.dot(Nbcz, fcn)
        rhoeig[Corn] = np.dot(NbcCorn, fcn)
        rhoeig = np.reshape(rhoeig,[Nz+1,Ny+1])

        peig = np.reshape(np.dot(M1full,eigVecs[:,0]),[Nz+1,Ny+1])

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
"""
def fd2(N):
    if N==0: D=0; x=1; return
    x = np.linspace(-1,1,N+1) #double check syntax
    h = 2./N
    e = np.ones(N+1)

    data = np.array([-1*e, 0*e, e])/(2*h)
    D = sp.spdiags(data, [-1, 0, 1], N+1,N+1)
    D = sp.csr_matrix(D)
    D[0, 0:2] = np.array([-1, 1])/h
    D[N, N-1:N+1] = np.array([-1, 1])/h

    D2 = sp.spdiags(np.array([e, -2*e, e])/h**2, [-1, 0, 1], N+1, N+1)
    D2 = sp.csr_matrix(D2)
    D2[0, 0:3] = np.array([1, -2, 1])/h**2
    D2[N, N-2:N+1] = np.array([1,-2,1])/h**2

    return D, D2, x
"""
def build_A(Ny,Nz,ik,kronU0,nu,LapU,kronDU0fm,fs,f,LapV,LapW,g,kronDR0m,N2,M1full,In,Dyv,Dzv):

    A = PETSc.Mat().createAIJ([(Ny-1)*(Nz-1)*4, (Ny-1)*(Nz-1)*4])
    A.setFromOptions(); A.setUp()
    start,end = A.getOwnershipRange()
    dim = (Ny-1)*(Nz-1)

    if start < dim: #A0
        A = assignNonzeros(A,ik*kronU0+nu*LapU,0,0)
        A = assignNonzeros(A,kronDU0fm, 0,dim)
        ii = 0 #rows
        for i in range(dim*2,dim*3): #cols
            A[ii,i] = fs
            ii += 1

    if (dim <= start < dim*2) or (start<dim and end > dim):
        A = assignNonzeros(A, ik*kronU0+nu*LapV,dim,dim)
        ii = dim #rows
        for i in range(0,dim): #cols
            A[ii,i] = f
            ii += 1

    if (dim*2 <= start < dim*3) or (start < dim*2 and end > dim*2):
        A = assignNonzeros(A, ik*kronU0 + nu*LapW,dim*2,dim*2)
        i0 = 0
        i3 = dim*3
        for i in range(dim*2,dim*3): #rows
            A[i,i0] = -fs
            A[i,i3] = g
            i0 +=1; i3 += 1

    if (dim*3 <= start <= dim*4) or (start < dim*3 and end > dim*3):
        A = assignNonzeros(A, ik*kronU0,dim*3,dim*3)
        A = assignNonzeros(A, kronDR0m,dim*3,dim)
        ii = dim*2
        for i in range(dim*3,dim*4):
            A[i,ii] = -N2
            ii += 1
    A.assemble()

    temp = PETSc.Mat().createAIJ(A.getSize())
    temp.setFromOptions(); temp.setUp()
    start,end = temp.getOwnershipRange()

    if start < len(In):
        temp = assignNonzeros(temp, ik*M1full[In,:], 0,0)
    if (len(In) < start < len(In)*2) or (start < len(In) and end > len(In)):
        temp = assignNonzeros(temp, np.dot(Dyv[In,:],M1full), len(In),0)
    if (len(In)*2 < start < len(In)*3) or (start < len(In)*2 and end > len(In)*2):
        temp = assignNonzeros(temp, np.dot(Dzv[In,:],M1full), len(In)*2,0)
    temp.assemble()
    A = -A - temp
    return A

def assignNonzeros(A,submat,br,bc):
    # br, bc Used to know which block in A we're in (i.e A01 for Fxy)
    smr,smc = np.nonzero(submat)
    smr = np.asarray(smr).ravel(); smc = np.asarray(smc).ravel()
    astart,aend = A.getOwnershipRange()

    for i in xrange(len(smr)):
        #ar,ac is where the nonzero value belongs in A
        ar = smr[i]+br
        ac = smc[i]+bc

        #check if location of nonzero is within processor's range
        if astart <= ar < aend: 
            A[ar,ac] = submat[smr[i],smc[i]] # assign nonzero value to it's position in A
    return A

if __name__=='__main__':
    main()
