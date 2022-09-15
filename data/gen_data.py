import os
import numpy
import pyscf

def gen_data(inp):
    inp = inp.split('-')
    mol = inp[0]

    hcore = None
    ovlp  = None
    eri   = None

    int_dir = None

    if len(inp) == 2:
        r = float(inp[1])
        int_dir = f"./{mol}/{r:.4f}"
        os.makedirs(int_dir, exist_ok=True)

        if mol == 'h2':
            m = pyscf.gto.M(
                atom = f'''
                H    0.0000000    0.0000000    0.0000000
                H    0.0000000    0.0000000    {r: 6.4f}
                ''',
                basis = 'sto-3g',
                verbose = 0,
            )

        elif mol == 'heh+':
            m = pyscf.gto.M(
                atom = f'''
                H    0.0000000    0.0000000    0.0000000
                He   0.0000000    0.0000000    {r: 6.4f}
                ''',
                basis = 'sto-3g',
                charge = 1,
                verbose = 0,
            )

        elif mol == 'h2o':
            m = pyscf.gto.M(
                atom = f'''
                    O
                    H  1  {r: 6.4f}
                    H  1  {r: 6.4f}  2 105
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

    mf = pyscf.scf.RHF(m)
    mf.max_cycle = 200
    mf.conv_tol  = 1e-10
    mf.kernel()
    if mf.converged:
        e_ref = mf.e_tot
        numpy.save(f"{int_dir}/ene_rhf.npy", e_ref)
    else:
        raise RuntimeError("SCF did not converge.")
    
    mf = pyscf.scf.UHF(m)
    mf.max_cycle = 200
    mf.conv_tol  = 1e-10
    mf.kernel()
    if mf.converged:
        e_ref = mf.e_tot
        numpy.save(f"{int_dir}/ene_uhf.npy", e_ref)
    else:
        raise RuntimeError("SCF did not converge.")

    nelecs = m.nelec
    e_nuc  = mf.energy_nuc()
    hcore  = m.intor('int1e_nuc')
    hcore += m.intor('int1e_kin')
    ovlp   = m.intor('int1e_ovlp')
    eri    = m.intor('int2e')
    numpy.save(f"{int_dir}/nelecs.npy", nelecs)
    numpy.save(f"{int_dir}/ene_nuc.npy", e_nuc)
    numpy.save(f"{int_dir}/hcore.npy", hcore)
    numpy.save(f"{int_dir}/ovlp.npy", ovlp)
    numpy.save(f"{int_dir}/eri.npy", eri)

if __name__ == "__main__":
    for r in numpy.linspace(0.5, 2.5, 21):
        inp = f"h2-{r:.4f}"
        gen_data(inp)
        inp = f"heh+-{r:.4f}"
        gen_data(inp)
        inp = f"h2o-{r:.4f}"
        gen_data(inp)

