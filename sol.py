import numpy
import scipy

# Use eigh to diagonalize matrices
from scipy.linalg import eigh

def solve_rhf(nelecs, hcore: numpy.ndarray, ovlp: numpy.ndarray, eri: numpy.ndarray,
              ene_nuc :float = 0.0, max_iter :int = 100, tol: float = 1e-8) -> float:

    nelec_alph, nelec_beta = nelecs
    assert nelec_alph == nelec_beta, "This code only supports closed-shell systems."
    
    nao = hcore.shape[0]
    
    assert hcore.shape == (nao, nao)
    assert ovlp.shape  == (nao, nao)
    assert eri.shape   == (nao, nao, nao, nao)

    iter_scf     = 0
    is_converged = False
    is_max_iter  = False

    ene_err = 1.0
    dm_err  = 1.0

    ene_rhf = None
    ene_old = None
    ene_cur = None

    nmo  = nao
    nocc = (nelec_alph + nelec_beta) // 2
    mo_occ = numpy.zeros(nmo, dtype=int)
    mo_occ[:nocc] = 2
    occ_list = numpy.where(mo_occ > 0)[0]

    # Diagonalize the core Hamiltonian to get the initial guess for density matrix
    energy_mo, coeff_mo = eigh(hcore, ovlp)
    coeff_occ = coeff_mo[:, occ_list]
    dm_old    = numpy.dot(coeff_occ, coeff_occ.T) * 2.0
    dm_cur    = None
    fock      = None

    while not is_converged and not is_max_iter:
        # Compute the Fock matrix
        coul =   numpy.einsum("pqrs,rs->pq", eri, dm_old)
        exch = - numpy.einsum("prqs,rs->pq", eri, dm_old) / 2.0
        fock = hcore + coul + exch

        # Diagonalize the Fock matrix
        energy_mo, coeff_mo = eigh(fock, ovlp)

        # Compute the new density matrix
        coeff_occ = coeff_mo[:, occ_list]
        dm_cur    = numpy.dot(coeff_occ, coeff_occ.T) * 2.0

        # Compute the energy
        ene_cur = 0.5 * numpy.einsum("pq,pq->", hcore + fock, dm_cur)
        ene_rhf = ene_cur + ene_nuc
        
        # Compute the errors
        if ene_old is not None:
            dm_err  = numpy.linalg.norm(dm_cur - dm_old)
            ene_err = abs(ene_cur - ene_old)
            print(f"SCF iteration {iter_scf:3d}, energy = {ene_rhf: 12.8f}, error = {ene_err: 6.4e}, {dm_err: 6.4e}")

        dm_old  = dm_cur
        ene_old = ene_cur
        
        # Check convergence
        iter_scf += 1
        is_max_iter  = iter_scf >= max_iter
        is_converged = ene_err < tol and dm_err < tol

    if is_converged:
        print(f"SCF converged in {iter_scf} iterations.")
    else:
        if ene_rhf is not None:
            print(f"SCF did not converge in {max_iter} iterations.")
        else:
            ene_rhf = 0.0
            print("SCF is not running.")

    return ene_rhf