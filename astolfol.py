import os, typing
import numpy
import scipy
from scipy.linalg import eigh

def solve_rhf(number, h_core: numpy.ndarray, s_overlap: numpy.ndarray, energy_between_elec: numpy.ndarray, energy_between_nuc :float = 0.0, max_iter :int = 100, tol: float = 1e-8) -> float:

    number_AO = h_core.shape[0]

    alpha, beta = number

    iter_scf     = 0
    is_converged = False
    is_max_iter  = False

    ene_err = 1.0
    dm_err  = 1.0

    ene_rhf = None
    ene_old = None
    ene_new = None

    number_MO  = number_AO
    MO_occpy_alpha = numpy.zeros(number_MO, dtype=int)
    MO_occpy_beta = numpy.zeros(number_MO, dtype=int)
    MO_occpy_alpha[:alpha] = 1
    MO_occpy_beta[:beta] = 1.01
    occ_list_alpha = numpy.where(MO_occpy_alpha > 0)[0]
    occ_list_beta = numpy.where(MO_occpy_beta > 0)[0]

    energy_mo, coeff_mo = eigh(h_core, s_overlap)
    coeff_occpy_alpha = coeff_mo[:, occ_list_alpha]
    coeff_occpy_beta = coeff_mo[:, occ_list_beta]
    density_matrix_alpha_old = numpy.dot(coeff_occpy_alpha, coeff_occpy_alpha.T)
    density_matrix_beta_old = numpy.dot(coeff_occpy_beta, coeff_occpy_beta.T)
    density_matrix_alpha_new = None
    density_matrix_beta_new = None
    fock = None

    while not is_converged and not is_max_iter:
        # Compute the Fock matrix
        coul = numpy.einsum("pqrs,rs->pq", energy_between_elec, density_matrix_alpha_old + density_matrix_beta_old)
        exch_alpha = - numpy.einsum("prqs,rs->pq", energy_between_elec, density_matrix_alpha_old)
        exch_beta = - numpy.einsum("prqs,rs->pq", energy_between_elec, density_matrix_beta_old) 
        fock_alpha = h_core + coul + exch_alpha
        fock_beta  = h_core + coul + exch_beta

        # For restricted hartree-fock, we can assume fock_alpha is
        # identical to fock_beta.
        # Think about how to change it to unrestricted hartree-fock.
        fock_diff = numpy.linalg.norm(fock_alpha - fock_beta)
        print(f"Fock matrix difference: {fock_diff: 6.4e}")
        # If you use an initial guess to break alpha beta symmetry, you
        # may found in some case the fock matrices are not identical,
        # and the difference will be smaller or converge to some finite
        # value.

        fock = fock_alpha

        # Diagonalize the Fock matrix
        energy_mo, coeff_mo = eigh(fock, s_overlap)

        # Compute the new density matrix
        coeff_occpy_alpha = coeff_mo[:, occ_list_alpha]
        coeff_occpy_beta = coeff_mo[:, occ_list_beta]
        density_matrix_alpha_new = numpy.dot(coeff_occpy_alpha, coeff_occpy_alpha.T)
        density_matrix_beta_new = numpy.dot(coeff_occpy_beta, coeff_occpy_beta.T)

        # Compute the energy
        h = numpy.einsum("pq,pq->", h_core, density_matrix_alpha_new + density_matrix_beta_new)
        k_alpha = numpy.einsum("pq,pq->", exch_alpha, density_matrix_alpha_new)
        k_beta = numpy.einsum("pq,pq->", exch_beta, density_matrix_beta_new)
        j = numpy.einsum("pq,pq->", coul, density_matrix_alpha_new + density_matrix_beta_new)
        ene_new = h + 0.5 * (j + k_alpha + k_beta)
        ene_rhf = ene_new + energy_between_nuc

        # Compute the errors
        if ene_old is not None:
            dm_err  = numpy.linalg.norm(density_matrix_alpha_new - density_matrix_alpha_old) + numpy.linalg.norm(density_matrix_beta_new - density_matrix_beta_old)
            ene_err = abs(ene_new - ene_old)
            print(f"SCF iteration {iter_scf:3d}, energy = {ene_rhf: 12.8f}, error = {ene_err: 6.4e}, {dm_err: 6.4e}")

        density_matrix_alpha_old = density_matrix_alpha_new
        density_matrix_beta_old = density_matrix_beta_new
        ene_old = ene_new

        # Check convergence
        iter_scf += 1
        is_max_iter  = (iter_scf >= max_iter)
        is_converged = (ene_err < tol) and (dm_err < tol)

    if is_converged:
        print(f"SCF converged in {iter_scf} iterations.")
    else:
        if ene_rhf is not None:
            print(f"SCF did not converge in {max_iter} iterations.")
        else:
            ene_rhf = 0.0
            print("SCF is not running.")

    return ene_rhf

def body(input:str) -> None:
    Input = input.split('-')
    Molecule = Input[0]
    integer_data = f"./data/{Molecule}/{float(Input[1]):.4f}"
    
    num_elec = numpy.load(f"{integer_data}/nelecs.npy")
    h_core = numpy.load(f"{integer_data}/hcore.npy")
    s_overlap = numpy.load(f"{integer_data}/ovlp.npy")
    energy_between_elec = numpy.load(f"{integer_data}/eri.npy")
    energy_between_nuc = numpy.load(f"{integer_data}/ene_nuc.npy")

    ref = numpy.load(f"{integer_data}/ene_rhf.npy")

    tol = 1e-8

    ene_rhf = solve_rhf(num_elec, h_core, s_overlap, energy_between_elec, tol=tol, max_iter=200, energy_between_nuc=energy_between_nuc)
    
    print(f"RHF energy: {ene_rhf: 12.8f}, Ref: {ref: 12.8f}, Err: {abs(ene_rhf - ref): 6.4e}")
    
    assert abs(ene_rhf - ref) < tol

# bondlength = input()
bl = 1.0
input = f"h2o-{bl:.4f}"
ene = body(input)