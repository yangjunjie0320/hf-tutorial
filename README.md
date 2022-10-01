### Introduction
This is a tutorial on how to implement a simple program
to solve the Hartree-Fock with SCF iterations.

### Impement the SCF procedure
We start from step in the algorithm described in Szabo and Ostlund, 3.4.6, p. 146

- Obtain a guess at the density matrix.
- Calculate the exchange and coulomb matrices from the density matrix
    and the two-electron repulsion integrals.
- Add exchange and coulomb matrices to the core-Hamiltonian to obtain the
    Fock matrix.
- Diagonalize the Fock matrix.
- Select the occupied orbitals and calculate the new density matrix.
- Compute the energy
- Compute the errors and check convergence
    - If converged, return the energy
    - If not converged, return to second step with new density matrix

You can first implement it for hydrogen molecule by running the following code:

```python
    mol = "h2"
    r   = 1.0
    inp = f"{mol}-{r:.4f}"
    e = main(inp) # H2 molecule with bond length 1.0
```
then try some more complicated molecules, such as HeH+
```python
    mol = "heh+"
    r   = 1.0
    inp = f"{mol}-{r:.4f}"
    e = main(inp) # HeH+ molecule with bond length 1.0
```
and H2O

```python
    mol = "h2o"
    r   = 1.0
    inp = f"{mol}-{r:.4f}"
    e = main(inp) # water molecule with bond length 1.0
```

### Potential Energy Surface

After implementing the SCF procedure, we can use it to compute the potential energy surface 
of the $\mathrm{H}_2$ molecule. We can do this by varying the internuclear distance and computing the
energy at each point.
```python
    for r in numpy.linspace(0.5, 2.5, 21):
        inp = f"h2-{r:.4f}"
        e = main(inp)
        print(f"H2: r={r: 6.4f}, e={e: 12.8f}")
```
You may also try other molecules, such as HeH+ and H2O.

### Unrestricted Hartree-Fock
Write a new function `solve_uhf` to implement the unrestricted Hartree-Fock method.
See what's the difference between the restricted and unrestricted Hartree-Fock potential energy surfaces. (Note that to obtain a reasonable dissociation potential energy surface, you may need 
to play some tricks to get the broken spin-symmetry UHF solution.)
### Dependencies
- `numpy`
- `scipy`
- If you wish to use `./data/gen_data.py` to generate the integrals, `pyscf` is also required.

### Reference
- Szabo and Ostlund, _Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory_,
  Dover Publications, New York, 1996
- Helgaker, Jørgensen, and Olsen, _Molecular Electronic-Structure Theory_, Wiley, New York, 2000
