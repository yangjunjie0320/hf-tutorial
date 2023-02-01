import numpy
import scipy

# Use eigh to diagonalize matrices
from scipy.linalg import eigh

def solve_rhf(nelecs, hcore: numpy.ndarray, ovlp: numpy.ndarray, eri: numpy.ndarray,
              ene_nuc :float = 0.0, max_iter :int = 100, tol: float = 1e-8, 
              full_return: bool = False):

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

    res_dict = {
        "ene_rhf" :     ene_rhf,
        "rdm1_ao" :     dm_cur,
        "fock_ao" :     fock,
        "coeff_ao_mo" : coeff_mo,
    }

    if is_converged:
        print(f"SCF converged in {iter_scf} iterations.")
    else:
        if ene_rhf is not None:
            print(f"SCF did not converge in {max_iter} iterations.")
        else:
            raise RuntimeError("SCF is not running, fill in the code in the main loop.")

    if not full_return:
        return ene_rhf

    else:
        return ene_rhf, res_dict

def solve_cis(nelecs, fock_ao, eri_ao, coeff_ao_mo, singlet: bool = True, triplet: bool = False):
    assert singlet or triplet
    assert not (singlet and triplet)

    if triplet:
        raise NotImplementedError("Triplet CIS is not implemented.")

    nocc_alph, nocc_beta = nelecs
    assert nocc_alph == nocc_beta

    nao  = fock_ao.shape[0]
    nocc = nocc_alph
    nmo  = coeff_ao_mo.shape[1]
    nvir = nmo - nocc

    fock_mo  = numpy.dot(coeff_ao_mo.T, numpy.dot(fock_ao, coeff_ao_mo))
    foo      = fock_mo[:nocc, :nocc]
    fvv      = fock_mo[nocc:, nocc:]
    ene_occ  = numpy.diag(foo)
    ene_vir  = numpy.diag(fvv)
    assert numpy.linalg.norm(numpy.diag(ene_occ) - foo) < 1e-10
    assert numpy.linalg.norm(numpy.diag(ene_vir) - fvv) < 1e-10

    e_vir_minus_e_occ = numpy.zeros((nvir, nocc))
    for i in range(nocc):
        for a in range(nvir):
            e_vir_minus_e_occ[a, i] = ene_vir[a] - ene_occ[i]

    eri_mo   = numpy.einsum("mnkl,mp,nq,kr,ls->pqrs", eri_ao, coeff_ao_mo, coeff_ao_mo, coeff_ao_mo, coeff_ao_mo, optimize=True)
    eri_ovov = eri_mo[:nocc, nocc:, :nocc, nocc:]
    eri_oovv = eri_mo[:nocc, :nocc, nocc:, nocc:]
    ham_cis  = eri_ovov * 2.0
    ham_cis -= numpy.einsum("ijab->iajb", eri_oovv)

    ham_cis = ham_cis.reshape(nvir * nocc, nvir * nocc) + numpy.diag(e_vir_minus_e_occ.T.ravel())
    ene_cis_list, amp_cis_list = eigh(ham_cis)

    return ene_cis_list, amp_cis_list
