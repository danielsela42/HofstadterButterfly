import matplotlib.pyplot as plt
import numpy as np

def gcd(a, b):
    ''' Get greatest common divisor of a and bo assuming a > b
    '''
    while b != 0:
        a, b = b, a % b
    return a

def coprimes(n):
    ''' Return list of coprimes with denominator up to n
    '''
    cp_list = list()
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i >= j: continue
            if gcd(i, j): cp_list.append((i, j))
    return cp_list

def make_H(kx, ky, p, q, tb, ta):
    ''' Make Hamiltonian at kx, ky momentum points

    Inputs: kx, ky - momentum points
            p - numerator of flux
            q - denominator of flux
                gcd(p, q) = 1
            tb, ta - hopping parameters in x, y respectively
    '''
    # Initialize zeros matrix
    Hmat = np.zeros((q, q), dtype=complex)

    # Create matrix by row
    for i in range(q):
        # Diagonal terms
        Hmat[i, i] = -2*ta*np.cos(kx + 2*np.pi*(p/q)*(i + 1))
        # Immediate off-diagonals
        if q > 2:
            if i < q - 1: Hmat[i, i + 1] = -1*tb
            if i > 0: Hmat[i, i - 1] = -1*tb

    # Term in far corner of matrix
    corner_term = -1*tb*np.exp(-1j*q*ky)
    Hmat[0, q-1] = corner_term
    Hmat[q-1, 0] = np.conj(corner_term)

    # Fix matrix in case q = 2
    if q == 2:
        Hmat[0, q-1] += -1*tb
        Hmat[q-1, 0] += -1*tb

    return Hmat

def get_eigenvalues(n, kx, ky, tb, ta):
    ''' Get eigenvalues of Hamiltonian corresponding to p, q <= n coprime 

    Inputs: n - Maximum valuye of coprimes
            kx, ky - momentum points
            tb, ta - hopping parameters in x, y respectively
    '''
    fluxes = [] # Flux of eigenstates
    energies = [] # Eigenenergies
    cp_list = coprimes(n) # List of coprime p, q
    for p, q in cp_list:
        # Make Hamiltonian
        Hmat = make_H(kx, ky, p, q, tb, ta)

        # Get eigenvalues and corresponding fluxes
        eigs = np.linalg.eigvalsh(Hmat)
        energies.extend(eigs)
        fluxes.extend([p/q]*len(eigs))
    return fluxes, energies

def plot_spectra(n, p, q, tb, ta):
    ''' Generate and plot spectra in 3D

    Inputs: n - Number of points in [-pi, pi) for plot
            p - numerator of flux
            q - denominator of flux
                gcd(p, q) = 1
            tb, ta - hopping parameters in x, y respectively
    '''
    # Generate kx, ky meshgrid
    interval = np.linspace(-np.pi, np.pi, n)
    kxs, kys = np.meshgrid(interval/q, interval)

    # Initialize energy list; each index is an n x n array corresponding to different energies
    E_list = list()
    for k in range(q):
        E_list.append(np.zeros((n, n), dtype=np.float64))

    # Get energies and append to component of E_list
    for i in range(n):
        for j in range(n):
            kx = kxs[i, j]
            ky = kys[i, j]
            Hmat = make_H(kx, ky, p, q, tb, ta)
            energies = np.sort(np.linalg.eigvalsh(Hmat, UPLO='U'))
            for k in range(q):
                E_list[k][i, j] = energies[k]


    # Generate plot in 3D
    _ = plt.figure()
    ax = plt.axes(projection='3d')
    for k in range(q):
        ax.plot_surface(kxs, kys, E_list[k])
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_zlabel(r"$E$")
    ax.set_title(rf"$\phi=2\pi ({p}/{q})$")
    ax.view_init(elev=0, azim=0)

def w_eigs(n, bands, p, q, tb, ta, vect1, vect2, vect0):
    ''' Get args of eigenvalues of wilson loop

    Parameters: n - number of divisions in kx and ky
                bands - list of bands to consider
                eigvects - numpy array of eigenvectors
                           indexed [j, i] with j for ky and i for kx
    '''
    eigs_list = list()
    for i in range(n + 1):
        # Get product of projections
        proj_prod = 1
        # Loop over increment
        for j in range(1, n):
            # Create individual projections
            kx, ky = (i*vect1 + j*vect2)/n + vect0
            Hmat = make_H(kx, ky, p, q, tb, ta)
            _, eigvects = np.linalg.eigh(Hmat, UPLO='U')

            proj = 0
            for band in bands:
                vect = eigvects[:, band]
                proj += np.outer(vect, np.conjugate(vect))

            # Left multiply projection by new, higher increment
            proj_prod = np.dot(proj, proj_prod)

        # Get edge eigvect
        kx, ky = (i*vect1)/n + vect0
        _, eigvect0 = np.linalg.eigh(make_H(kx, ky, p, q, tb, ta), UPLO='U')

        # Create wilson loop
        w_mat = np.zeros((len(bands), len(bands)), dtype=complex)
        for k in range(len(bands)):
            for l in range(len(bands)):
                w_elem = np.dot(np.conjugate(eigvect0[:, bands[k]]), np.dot(proj_prod, eigvect0[:, bands[l]]))
                w_mat[k, l] = w_elem
        
        # Get eigenvalues and arguments
        vals = np.linalg.eigvals(w_mat)
        eigs_list.append((kx, [np.angle(val) for val in vals]))
    return eigs_list

def plot_phase(n, p, q, bands, tb, ta, vect1, vect2, vect0):
    ''' Plot the aruments from the wilson loop

    Inputs: n - number of divisions in kx and ky
            p - numerator of flux (integer)
            q - denominator of flux (integer)
                gcd(p, q) = 1
            bands - list of bands to consider
            eigvects - numpy array of eigenvectors
                    indexed [j, i] with j for ky and i for kx
            lin_kx - the actual kx points
    '''
    # Get arguments from Wilson loop
    eigs_list = w_eigs(n, bands, p, q, tb, ta, vect1, vect2, vect0)
    xs = list()
    ys = list()
    for kx, vals in eigs_list:
        xs.extend([kx]*len(vals))
        ys.extend(vals)
    
    # Location of flux
    if len(bands) == 1:
        bands_str = bands[0] + 1
    else:
        bands_str = bands[1] + 1

    # Create plot
    plt.figure()
    plt.title(rf"$\phi=2\pi ({p}/{q})$ at gap {bands_str}")
    plt.scatter(xs, ys)

def get_hall_cond(n, p, q, tb, ta):
    ''' Get hall conductivity with wilson loop

    Inputs: n - number of divisions in kx, ky
            p - numerator of flux (integer)
            q - denominator of flux (integer)
                gcd(p, q) = 1
            tb, ta - hopping parameters in y, x respectively
    '''
    vect1 = 2*np.pi*np.array([1, 0])
    vect2 = 2*np.pi*np.array([0, 1])
    vect0 = np.pi*np.array([-1, -1])

    # Cases for central band
    if q % 2 != 0:
        # Loop over each band and plot
        for m in range(q):
            plot_phase(n, p, q, [m], tb, ta, vect1, vect2, vect0)
    else:
        # Loop over each band and plot
        for m in range(q):
            if m == (int(q/2) - 1):
                plot_phase(n, p, q, [m, m + 1], tb, ta, vect1, vect2, vect0)
            elif m == int(q/2):
                continue
            else:
                plot_phase(n, p, q, [m], tb, ta, vect1, vect2, vect0)
    plt.show()
    return


if __name__=="__main__":
    n = 200

    ta = 1
    tb = 1

    p = 1
    q = 3
    scale = 1
    get_hall_cond(n, p, q, tb, ta)
