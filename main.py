import os
import numpy
import scipy

# Use eigh to diagonalize matrices
from scipy.linalg import eigh

def solve_hf(hcore: numpy.ndarray, ovlp: numpy.ndarray, eri: numpy.ndarray, tol: float = 1e-8) -> numpy.ndarray:
    """
    Solve the Hartree-Fock with SCF iterations.
    Reference: Szabo and Ostlund, 3.4.6. (p. 146, start from step 2)

    The SCF procedure is:
        - Initialize the density matrix
        - Compute the Fock matrix
        - Diagonalize the Fock matrix
        - Compute the new density matrix
        - Check convergence
            - If converged, return the energy
            - If not converged, return to second step with new density matrix

    Inputs:
        hcore : numpy.ndarray
            The core Hamiltonian matrix.
        ovlp : numpy.ndarray
            The overlap matrix.
        eri : numpy.ndarray
            The two-electron repulsion integrals.
        tol : float, optional
            The convergence tolerance, by default 1e-8.
    """

    nao = hcore.shape[0]
    assert hcore.shape == (nao, nao)
    assert ovlp.shape  == (nao, nao)
    assert eri.shape   == (nao, nao, nao, nao)

    print("Great! We are ready to solve the Hartree-Fock equations...")

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

    hcore = None
    ovlp = None
    eri = None
    
    if len(inp) == 1:
        int_dir = f"./data/{mol}"
        if not os.path.exists(int_dir):
            raise RuntimeError("Molecule not supported.")

        hcore = numpy.load(f"{int_dir}/hcore.npy")
        ovlp  = numpy.load(f"{int_dir}/ovlp.npy")
        eri   = numpy.load(f"{int_dir}/eri.npy")

    elif len(inp) == 2:
        int_dir = f"./data/{mol}"
        if not os.path.exists(int_dir):
            raise RuntimeError("Molecule not supported.")
        
        int_dir = f"./data/{mol}/{float(inp[1]):.4f}"
        if not os.path.exists(int_dir):
            raise RuntimeError("Bond length not supported.")

        hcore = numpy.load(f"{int_dir}/hcore.npy")
        ovlp  = numpy.load(f"{int_dir}/ovlp.npy")
        eri   = numpy.load(f"{int_dir}/eri.npy")

    solve_hf(hcore, ovlp, eri, tol=1e-8)

if __name__ == "__main__":
    inp = "h2-0.7"
    main(inp)