import os
import numpy
import pyscf

def gen_data(inp):
    inp = inp.split('-')
    mol = inp[0]

    hcore = None
    ovlp  = None
    eri   = None
    
    if len(inp) == 1:
        int_dir = f"./{mol}"
        os.makedirs(int_dir, exist_ok=True)

        m = pyscf.gto.M(
            atom = '''
            O   -0.0000000   -0.1113512    0.0000000
            H    0.0000000    0.4454047   -0.7830363
            H   -0.0000000    0.4454047    0.7830363
            ''',
            basis = 'sto-3g',
            verbose = 0,
        )

        hcore = m.intor('int1e_nuc')
        ovlp  = m.intor('int1e_ovlp')
        eri   = m.intor('int2e')

        numpy.save(f"{int_dir}/hcore.npy", hcore)
        numpy.save(f"{int_dir}/ovlp.npy", ovlp)
        numpy.save(f"{int_dir}/eri.npy", eri)

    elif len(inp) == 2:
        int_dir = f"./{mol}/{inp[1]}"
        os.makedirs(int_dir, exist_ok=True)

        if mol == 'h2':
            m = pyscf.gto.M(
                atom = f'''
                H    0.0000000    0.0000000    0.0000000
                H    0.0000000    0.0000000    {inp[1]}
                ''',
                basis = 'sto-3g',
                verbose = 0,
            )
        elif mol == 'heh+':
            m = pyscf.gto.M(
                atom = f'''
                H    0.0000000    0.0000000    0.0000000
                He   0.0000000    0.0000000    {inp[1]}
                ''',
                basis = 'sto-3g',
                charge = 1,
                verbose = 0,
            )

        hcore = m.intor('int1e_nuc')
        ovlp  = m.intor('int1e_ovlp')
        eri   = m.intor('int2e')

        numpy.save(f"{int_dir}/hcore.npy", hcore)
        numpy.save(f"{int_dir}/ovlp.npy", ovlp)
        numpy.save(f"{int_dir}/eri.npy", eri)

if __name__ == "__main__":
    inp = "h2o"
    gen_data(inp)

    for r in numpy.linspace(0.5, 2.5, 21):
        inp = f"h2-{r:.4f}"
        gen_data(inp)
        inp = f"heh+-{r:.4f}"
        gen_data(inp)