import os
import numpy
import pyscf

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

    mf = pyscf.scf.RHF(mol)
    mf.max_cycle = 200
    mf.conv_tol  = 1e-10
    mf.kernel()
    if mf.converged:
        e_ref = mf.e_tot
        numpy.save(f"{int_dir}/ene_rhf.npy", e_ref)
    else:
        raise RuntimeError("SCF did not converge.")
    
    mf = pyscf.scf.UHF(mol)
    mf.max_cycle = 200
    mf.conv_tol  = 1e-10
    mf.kernel()
    if mf.converged:
        e_ref = mf.e_tot
        numpy.save(f"{int_dir}/ene_uhf.npy", e_ref)
    else:
        raise RuntimeError("SCF did not converge.")

    nelecs = mol.nelec
    e_nuc  = mf.energy_nuc()
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

