from fractions import Fraction
from re import L
import matplotlib.pyplot as plt
import numpy as np

def gcd(a, b):
    ''' Get greatest common divisor of a and bo assuming a > b
    '''
    while b != 0:
        a, b = b, a % b
    return a

def coprimes(n):
    # Return list of coprimes with denominator up to n
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


def get_hall_cond_1(n, p, q, tb, ta, scale=1):
    ''' Attempt 2 at getting Hall conductivity completely numerically

    Inputs: n - Number of points in [-pi, pi) for plot
            p - numerator of flux
            q - denominator of flux
                gcd(p, q) = 1
            tb, ta - hopping parameters in x, y respectively
    '''
    lin_kx = np.linspace(-np.pi/q, np.pi/q, n, endpoint=False)
    lin_ky = np.linspace(-np.pi, np.pi, n, endpoint=False)

    kxs, kys = np.meshgrid(lin_kx, lin_ky)

    F = list()
    for kx in np.ravel(kxs):
        for ky in np.ravel(kys):
            dis = 2*np.pi/(q * scale)
            loop_points = list()
            for i in [-1, 1]:
                kx_shift = kx + i * dis
                if kx_shift < -np.pi/q:
                    kx_shift += np.pi/q
                elif kx_shift < np.pi/q:
                    kx_shift -= np.pi/q
                ky_shift = ky + i * dis
                if ky_shift < -np.pi:
                    ky_shift += np.pi
                elif ky_shift < np.pi:
                    ky_shift -= np.pi
                loop_points.append(np.array([kx_shift, ky]))
                loop_points.append(np.array([kx, ky_shift]))
            
            loop_vects = list()
            for i in range(4):
                kx1, ky1 = loop_points[i]
                Hmat = make_H(kx1, ky1, p, q, tb, ta)
                _, vects = np.linalg.eigh(Hmat)
                loop_vects.append(vects)

            # Compute overlap
            Fs = list()
            for j in range(len(loop_vects[0])):
                vect_list = [loop_vects[k][:, j] for k in range(4)]
                total = 0
                for i in range(4):
                    total += np.angle(np.vdot(vect_list[i], vect_list[(i + 1) % len(loop_points)]))
                Fs.append(total)
            F.append(Fs)

    F = np.transpose(F)

    return np.sum(F, axis=1)/(2*np.pi)


def get_hall_cond_2(n, p, q, tb, ta):
    ''' Attempt 2 at getting Hall conductivity completely numerically

    Inputs: n - Number of points in [-pi, pi) for plot
            p - numerator of flux
            q - denominator of flux
                gcd(p, q) = 1
            tb, ta - hopping parameters in x, y respectively
    '''
    endpoint = False
    endpoint = True

    # Defin kx, ky intervals in BZ
    lin_kx = np.linspace(-np.pi/q, np.pi/q, n, endpoint=endpoint)
    lin_ky = np.linspace(-np.pi, np.pi, n, endpoint=endpoint)

    # Define meshgrid of kxs and kys
    kxs, kys = np.meshgrid(lin_kx, lin_ky)

    # Initialize eigenvector storage
    eigvects = np.zeros((n, n, q, q), dtype=complex)
    
    # Diagonalize at each kx, ky in mesh
    band_num = 0
    for i in range(n):
        for j in range(n):
            kx = kxs[i, j]
            ky = kys[i, j]
            Hmat = make_H(kx, ky, p, q, tb, ta)
            vals, vects = np.linalg.eigh(Hmat, UPLO='U')
            eigvects[i, j] = vects

            # Verify correct number of eigenvalues
            if i == 0 and j == 0: band_num = len(vals)
            if band_num != len(vals): print("ERROR: Non-matching bands")

    # Loop shift, indices
    loop_coords = [(0, 0), (0, 1), (1, 1), (1, 0)]

    # Adjust looping endpoint
    if endpoint: end = n-1
    else: end = n

    F = list()
    # Loop over each band
    for m in range(band_num):
        # print(m)
        vortices = 0
        loop_sum_tot = 0
        # Loop over each position
        for i in range(end):
            for j in range(end):
                loop_phase = 1 # Accumulated phase
                loop_sum = 0 # Accumulated angle
                for k in range(4):
                    # Define current and next step in loop
                    di, dj = loop_coords[k]
                    di_n, dj_n = loop_coords[(k + 1) % 4]
                    i_in = (i + di) % n # Current
                    j_in = (j + dj) % n # Current
                    i_in_n = (i + di_n) % n # Next
                    j_in_n = (j + dj_n) % n # Next
                    dot_prod = np.vdot(eigvects[i_in, j_in][:, m], eigvects[i_in_n, j_in_n][:, m])
                   
                    loop_phase *= dot_prod
                    loop_sum += np.angle(dot_prod)
                    
                loop_sum_tot += loop_sum
                loop_angle = np.angle(loop_phase)

                # Find vortex number
                vortex = (loop_angle - loop_sum)/(2*np.pi)
                # vortex = loop_angle/(2*np.pi)
                vortices += vortex
                # print("\t", loop_angle/(2*np.pi), loop_sum/(2*np.pi), vortex)

        print(m + 1, "loop_sum", loop_sum_tot/(2*np.pi))
        F.append(vortices)

    # Adjust middle two bands if even, combine chern number
    if q % 2 == 0:
        adjust_F = np.zeros(q - 1)
        adjust_F[int(q/2) - 1] = F[int(q/2) - 1] + F[int(q/2)]
        for i in range(q - 1):
            if i < q/2:
                adjust_F[i] = F[i]
            elif i > q/2:
                adjust_F[i] = F[i + 1]
    else:
        adjust_F = F

    return np.round(np.cumsum(np.array(adjust_F)))


def w_eigs(n, bands, eigvects):
    ''' Get args of eigenvalues of Wilson loop

    Inputs: n - number of divisions in kx and ky
                bands - list of bands to consider
                eigvects - numpy array of eigenvectors
                        indexed [j, i] with j for ky and i for kx
    '''
    eigs_list = list()
    for i in range(n):
        # Get product of projections
        proj_prod = 1
        # Loop over increment
        for j in range(1, n):
            # Create individual projections
            proj = 0
            for band in bands:
                vect = eigvects[j, i][:, band]
                proj += np.outer(vect, np.conjugate(vect))

            # Left multiply projection by new, higher increment
            proj_prod = np.dot(proj, proj_prod)

        # Create Wilson loop
        w_mat = np.zeros((len(bands), len(bands)), dtype=complex)
        for k in range(len(bands)):
            for l in range(len(bands)):
                w_elem = np.dot(eigvects[0, i][:, k], np.dot(proj_prod, eigvects[0, i][:, l]))
                w_mat[k, l] = w_elem
        
        # Get eigenvlaues and arguments
        vals = np.linalg.eigvals(w_mat)
        eigs_list.append((i, [np.angle(val) for val in vals]))
    return eigs_list

def plot_phase(n, p, q, bands, eigvects, lin_kx):
    ''' Plot the aruments from the wilson loop

    Parameters: n - number of divisions in kx and ky
                p - numerator of flux (integer)
                q - denominator of flux (integer)
                    gcd(p, q) = 1
                bands - list of bands to consider
                eigvects - numpy array of eigenvectors
                        indexed [j, i] with j for ky and i for kx
                lin_kx - the actual kx points
    '''
    # Get eigenvalues corresponding to Wilson loop
    eigs_list = w_eigs(n, bands, eigvects)
    xs = list()
    ys = list()
    for i, vals in eigs_list:
        xs.extend([lin_kx[i]]*len(vals))
        ys.extend(vals)
    
    if len(bands) == 1:
        bands_str = bands[0] + 1
    else:
        bands_str = bands[1] + 1

    # Plot the arguments
    plt.figure()
    plt.title(rf"$\phi=2\pi ({p}/{q})$ at gap {bands_str}")
    plt.scatter(xs, ys)

def get_hall_cond_3(n, p, q, tb, ta):
    ''' Get hall conductivity with wilson loop

    Inputs: n - number of divisions in kx, ky
            p - numerator of flux (integer)
            q - denominator of flux (integer)
                gcd(p, q) = 1
            tb, ta - hopping parameters in y, x respectively
    '''
    # Define kx, ky intervals in BZ
    lin_kx = np.linspace(-np.pi/q, np.pi/q, n, endpoint=True)
    lin_ky = np.linspace(-np.pi, np.pi, n, endpoint=False)

    # Define meshgrid of kxs and kys
    kxs, kys = np.meshgrid(lin_kx, lin_ky)

    # Initialize eigenvector storage
    eigvects = np.zeros((n, n, q, q), dtype=complex)
    
    # Diagonalize at each kx, ky in mesh
    band_num = 0
    for i in range(n):
        for j in range(n):
            kx = kxs[i, j]
            ky = kys[i, j]
            # Make matrix
            Hmat = make_H(kx, ky, p, q, tb, ta)

            # Get vals and vects
            vals, vects = np.linalg.eigh(Hmat, UPLO='U')
            eigvects[i, j] = vects # Store vects

            # Verify correct number of eigenvalues
            if i == 0 and j == 0: band_num = len(vals)
            if band_num != len(vals): print("ERROR: Non-matching bands")

    # Cases for central band
    if band_num % 2 != 0:
        # Loop over each band and plot
        for m in range(band_num):
            plot_phase(n, p, q, [m], eigvects, lin_kx)
    else:
        # Loop over each band and plot
        for m in range(band_num):
            if m == (int(band_num/2) - 1):
                plot_phase(n, p, q, [m, m + 1], eigvects, lin_kx)
            elif m == int(band_num/2):
                continue
            else:
                plot_phase(n, p, q, [m], eigvects, lin_kx)
    plt.show()
    return


if __name__=="__main__":
    n = 200

    ta = 1
    tb = 1

    p = 1
    q = 4
    scale = 1

    get_hall_cond_3(n, p, q, tb, ta)

    n = 50
    plot_spectra(n, p, q, tb, ta)
