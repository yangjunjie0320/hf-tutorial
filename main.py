import os, typing
import numpy
import scipy

# Use eigh to diagonalize matrices
from scipy.linalg import eigh

def solve_rhf(nelecs, hcore: numpy.ndarray, ovlp: numpy.ndarray, eri: numpy.ndarray,
              ene_nuc :float = 0.0, max_iter :int = 100, tol: float = 1e-8) -> numpy.ndarray:
    """
    Solve the Hartree-Fock with SCF iterations.
    Reference: Szabo and Ostlund, 3.4.6. (p. 146, start from step 2)

    The SCF procedure is:
        - Initialize the density matrix by diagonalizing the core Hamiltonian
        - Compute the Fock matrix
        - Diagonalize the Fock matrix
        - Compute the new density matrix
        - Compute the energy
        - Compute the errors and check convergence
            - If converged, return the energy
            - If not converged, return to second step with new density matrix

    Inputs:
        nelecs : tuple
            the number of alpha and beta electrons
        hcore : numpy.ndarray
            The core Hamiltonian matrix.
        ovlp : numpy.ndarray
            The overlap matrix.
        eri : numpy.ndarray
            The two-electron repulsion integrals.
        ene_nuc : float
            The nuclear repulsion energy.
        max_iter : int
            The maximum number of SCF iterations.
        tol : float, optional
            The convergence tolerance, by default 1e-8.
    """

    nelec_alph, nelec_beta = nelecs
    assert nelec_alph == nelec_beta, "This code only supports closed-shell systems."
    
    nao = hcore.shape[0]
    
    assert hcore.shape == (nao, nao)
    assert ovlp.shape  == (nao, nao)
    assert eri.shape   == (nao, nao, nao, nao)

    print("Great! We are ready to solve the Hartree-Fock equations...")

    iter_scf     = 0
    is_converged = False
    is_max_iter  = False

    ene_err = 1.0
    dm_err  = 1.0

    ene_rhf = None
    ene_old = None
    ene_cur = None

    dm_old    = None # Initial density matrix
    dm_cur    = None

    while not is_converged and not is_max_iter:
        # Fill in the code here to perform SCF iterations.

        # Compute the errors
        if ene_old is not None:
            dm_err  = numpy.linalg.norm(dm_cur - dm_old)
            ene_err = abs(ene_cur - ene_old)
            print(f"SCF iteration {iter_scf:3d}, energy = {ene_rhf: 12.8f}, error = {ene_err: 6.4e}, {dm_err: 6.4e}")

        iter_scf += 1
        is_max_iter  = iter_scf >= max_iter
        is_converged = ene_err < tol and dm_err < tol

    if is_converged:
        print(f"SCF converged in {iter_scf} iterations.")
    else:
        if ene_rhf is not None:
            print(f"SCF did not converge in {max_iter} iterations.")
        else:
            raise RuntimeError("SCF is not running, fill in the code in the main loop.")

    return ene_rhf

def main(inp: str) -> None:
    """
    The main function of the program.

    Inputs:
        inp : str
            the input can be either h2o, which gives the integrals for water at equilibrium geometry,
            or some 
    """

    inp = inp.split('-')
    mol = inp[0]

    tol = 1e-8
    
    if len(inp) == 2:
        int_dir = f"./data/{mol}"
        if not os.path.exists(int_dir):
            raise RuntimeError("Molecule not supported.")
        
        int_dir = f"./data/{mol}/{float(inp[1]):.4f}"
        if not os.path.exists(int_dir):
            raise RuntimeError("Bond length not supported.")

        nelecs  = numpy.load(f"{int_dir}/nelecs.npy")
        hcore   = numpy.load(f"{int_dir}/hcore.npy")
        ovlp    = numpy.load(f"{int_dir}/ovlp.npy")
        eri     = numpy.load(f"{int_dir}/eri.npy")
        ene_nuc = numpy.load(f"{int_dir}/ene_nuc.npy")

        ene_rhf_ref = numpy.load(f"{int_dir}/ene_rhf.npy")
        ene_uhf_ref = numpy.load(f"{int_dir}/ene_uhf.npy")

        # Implement the Hartree-Fock with SCF iterations 
        ene_rhf = solve_rhf(nelecs, hcore, ovlp, eri, tol=tol, max_iter=100, ene_nuc=ene_nuc)

        # This is the solution for RHF from Junjie, uncomment this to run it.
        # import sol
        # ene_rhf = sol.solve_rhf(nelecs, hcore, ovlp, eri, tol=tol, max_iter=100, ene_nuc=ene_nuc)

        print(f"RHF energy: {ene_rhf: 12.8f}, reference: {ene_rhf_ref: 12.8f}, error: {abs(ene_rhf - ene_rhf_ref): 6.4e}")

        assert abs(ene_rhf - ene_rhf_ref) < tol
    
    else:
        raise RuntimeError("Invalid input.")

if __name__ == "__main__":
    inp = "h2o-1.0"
    main(inp)