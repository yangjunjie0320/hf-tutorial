import os, sys
import numpy
import pyscf

sys.path.append("..")

def test_rhf():
    m = pyscf.gto.M(
        atom = f'''
            O    0.0000000    0.0184041   -0.0000000
            H    0.0000000   -0.5383516   -0.7830366
            H   -0.0000000   -0.5383516    0.7830366
        ''',
        basis = 'sto-3g',
        verbose = 0,
    )

    tol      = 1e-8
    max_iter = 100

    mf = pyscf.scf.RHF(m)
    mf.max_cycle = max_iter
    mf.conv_tol  = tol
    mf.kernel()
    assert mf.converged
    e_ref = mf.e_tot

    nelecs = m.nelec
    e_nuc  = mf.energy_nuc()
    hcore  = m.intor('int1e_nuc')
    hcore += m.intor('int1e_kin')
    ovlp   = m.intor('int1e_ovlp')
    eri    = m.intor('int2e')
    
    from sol import solve_rhf
    e_rhf = solve_rhf(
        nelecs, hcore, ovlp, eri, 
        ene_nuc=e_nuc, max_iter=100, tol=tol
        )

    assert numpy.abs(e_rhf - e_ref) < tol