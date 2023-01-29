import os
import numpy
import pyscf
from pyscf import gto
from pyscf import scf, fci

def gen_data(inp, dm_rhf):
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
            atoms = f"H  0.00000000    0.00000000    0.00000000\n"
            atoms += f"H  0.00000000    0.00000000  {bl: 12.8f}\n"

            with open(f"{int_dir}/atoms.xyz", 'w') as f:
                f.write(f"2\n")
                f.write(f"{bl: 12.8f}\n")
                f.write(atoms)

            mol = pyscf.gto.M(
                atom = atoms,
                basis = 'sto-3g',
                verbose = 0,
            )

        elif inp[0] == 'heh+':
            atoms = f"He 0.00000000    0.00000000    0.00000000\n"
            atoms += f"H  0.00000000    0.00000000  {bl: 12.8f}\n"

            with open(f"{int_dir}/atoms.xyz", 'w') as f:
                f.write(f"2\n")
                f.write(f"{bl: 12.8f}\n")
                f.write(atoms)

            mol = pyscf.gto.M(
                atom = atoms,
                basis = 'sto-3g',
                charge = 1,
                verbose = 0,
            )

        elif inp[0] == 'h2o':
            theta = 104.52 * numpy.pi / 180.0 / 2.0
            z_coord = bl * numpy.cos(theta)
            y_coord = bl * numpy.sin(theta)

            atoms = f"O 0.00000000    0.00000000    0.00000000\n"
            atoms += f"H 0.00000000  {y_coord: 12.8f}  {z_coord: 12.8f}\n"
            atoms += f"H 0.00000000  {-y_coord: 12.8f}  {z_coord: 12.8f}\n"

            with open(f"{int_dir}/atoms.xyz", 'w') as f:
                f.write(f"3\n")
                f.write(f"{bl: 12.8f}\n")
                f.write(atoms)

            mol = pyscf.gto.M(
                atom  = atoms,
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
    rhf_obj.kernel(dm_rhf)

    if rhf_obj.converged:
        ene_rhf = rhf_obj.e_tot
        numpy.save(f"{int_dir}/ene_rhf.npy", ene_rhf)

        ene_fci = pyscf.fci.FCI(rhf_obj).kernel()[0]
        numpy.save(f"{int_dir}/ene_fci.npy", ene_fci)
    else:
        raise RuntimeError("SCF did not converge.")

    dm_rhf = rhf_obj.make_rdm1()

    alph_ao_label = None
    beta_ao_label = None

    if inp[0] == 'h2':
        alph_ao_label = ['0 H 1s']
        beta_ao_label = ['1 H 1s']
    
    elif inp[0] == 'heh+':
        alph_ao_label = ['0 He 1s']
        beta_ao_label = ['0 He 1s']

    elif inp[0] == 'h2o':
        alph_ao_label = ['1 H 1s', '2 H 1s']
        beta_ao_label = ['O 2pz',  'O 2py']

    alph_ao_idx = mol.search_ao_label(alph_ao_label)
    beta_ao_idx = mol.search_ao_label(beta_ao_label)

    dms_bs = [dm_rhf / 2.0, dm_rhf / 2.0]
    dms_bs = numpy.asarray(dms_bs)

    dms_bs[0, alph_ao_idx, alph_ao_idx] = 1.0
    dms_bs[1, beta_ao_idx, beta_ao_idx] = 1.0
    dms_bs[0, beta_ao_idx, beta_ao_idx] = 0.0
    dms_bs[1, alph_ao_idx, alph_ao_idx] = 0.0

    uhf_obj = pyscf.scf.UHF(mol)
    uhf_obj.max_cycle = 200
    uhf_obj.conv_tol  = 1e-10
    uhf_obj.kernel(dms_bs)  

    if uhf_obj.converged:
        ene_uhf = uhf_obj.e_tot
        numpy.save(f"{int_dir}/ene_uhf.npy", ene_uhf)

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
    print(f"Generated data for {inp[0]:8s} at {bl: 6.4f}: ref RHF energy = {rhf_obj.e_tot: 16.10f}, ref UHF energy = {uhf_obj.e_tot: 16.10f}, ref FCI energy = {ene_fci: 16.10f}")

    return dm_rhf

if __name__ == "__main__":
    for mol in ['h2', 'heh+', 'h2o']:
        dm_rhf = None
        for bl in numpy.linspace(0.5, 3.0, 61):
            inp = f"{mol}-{bl:.4f}"
            dm_rhf = gen_data(inp, dm_rhf)

