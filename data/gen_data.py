import os
import numpy
import pyscf
from pyscf import scf
from pyscf import gto

def gen_data(inp):
    inp = inp.split('-')
    inp[0] = inp[0]

    hcore = None
    ovlp  = None
    eri   = None

    int_dir = None

    if len(inp) == 2:
        bl = float(inp[1])
        int_dir = f"./{inp[0]}/{bl:.4f}"
        os.makedirs(int_dir, exist_ok=True)

        if inp[0] == 'h2':
            mol = pyscf.gto.M(
                atom = f'''
                H    0.0000000    0.0000000    0.0000000
                H    0.0000000    0.0000000    {bl: 6.4f}
                ''',
                basis = 'sto-3g',
                verbose = 0,
            )

        elif inp[0] == 'heh+':
            mol = pyscf.gto.M(
                atom = f'''
                H    0.0000000    0.0000000    0.0000000
                He   0.0000000    0.0000000    {bl: 6.4f}
                ''',
                basis = 'sto-3g',
                charge = 1,
                verbose = 0,
            )

        elif inp[0] == 'h2o':
            mol = pyscf.gto.M(
                atom = f'''
                    O
                    H  1  {bl: 6.4f}
                    H  1  {bl: 6.4f}  2 105
                ''',
                basis = 'sto-3g',
                verbose = 0,
            )

    else:
        raise RuntimeError("Invalid input.")

    assert int_dir is not None
    if os.path.exists(int_dir):
        import shutil
        shutil.rmtree(int_dir)
        os.makedirs(int_dir)

    rhf_obj = pyscf.scf.RHF(mol)
    rhf_obj.max_cycle = 200
    rhf_obj.conv_tol  = 1e-10
    rhf_obj.kernel()
    if rhf_obj.converged:
        e_ref = rhf_obj.e_tot
        numpy.save(f"{int_dir}/ene_rhf.npy", e_ref)
    else:
        raise RuntimeError("SCF did not converge.")
    
    uhf_obj = pyscf.scf.UHF(mol)
    uhf_obj.max_cycle = 200
    uhf_obj.conv_tol  = 1e-10
    uhf_obj.kernel(rhf_obj.make_rdm1())

    if uhf_obj.converged:
        res = uhf_obj.stability(return_status=True)
        mo_coeff = res[0]

        stable_iter = 0
        while not res[2] and stable_iter < 10:
            uhf_obj.kernel(mo_coeff)
            res = uhf_obj.stability(return_status=True)
            mo_coeff = res[0]
            stable_iter += 1
        
        assert uhf_obj.converged, "SCF did not converge."
        assert res[2],            "SCF is not stable."

        e_ref = uhf_obj.e_tot
        numpy.save(f"{int_dir}/ene_uhf.npy", e_ref)
    else:
        raise RuntimeError("SCF did not converge.")

    nelecs = mol.nelec
    e_nuc  = rhf_obj.energy_nuc()
    hcore  = mol.intor('int1e_nuc')
    hcore += mol.intor('int1e_kin')
    ovlp   = mol.intor('int1e_ovlp')
    eri    = mol.intor('int2e')
    numpy.save(f"{int_dir}/nelecs.npy", nelecs)
    numpy.save(f"{int_dir}/ene_nuc.npy", e_nuc)
    numpy.save(f"{int_dir}/hcore.npy", hcore)
    numpy.save(f"{int_dir}/ovlp.npy", ovlp)
    numpy.save(f"{int_dir}/eri.npy", eri)

if __name__ == "__main__":
    for bl in numpy.linspace(0.5, 2.5, 21):
        inp = f"h2-{bl:.4f}"
        gen_data(inp)
        inp = f"heh+-{bl:.4f}"
        gen_data(inp)
        inp = f"h2o-{bl:.4f}"
        gen_data(inp)

