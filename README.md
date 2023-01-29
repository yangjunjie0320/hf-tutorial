### Introduction
This is a tutorial on how to implement a simple program
to solve the Hartree-Fock with SCF iterations.

### Impement the SCF procedure
We will be following the algorithm described in Szabo and Ostlund, 3.4.6, p. 146 to implement the Hartree-Fock method for the hydrogen molecule. The steps are as follows:

1. Obtain a guess at the density matrix.
2. Calculate the exchange and coulomb matrices from the density matrix and the two-electron repulsion integrals.
3. Add exchange and coulomb matrices to the core-Hamiltonian to obtain the Fock matrix.
4. Diagonalize the Fock matrix.
5. Select the occupied orbitals and calculate the new density matrix.
6. Compute the energy.
7. Compute the errors and check for convergence.
  - If converged, return the energy.
  - If not converged, return to second step with the new density matrix.

To start, you can implement this algorithm for the hydrogen molecule by running the following code:

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

After implementing the self-consistent field (SCF) procedure, we can use it to compute the potential energy surface of the $\mathrm{H}_2$ molecule. We can do this by varying the internuclear distance and computing the energy at each point.
```python
    for r in numpy.linspace(0.5, 3.0, 61):
        inp = f"h2-{r:.4f}"
        e = main(inp)
        print(f"H2: r={r: 6.4f}, e={e: 12.8f}")
```

You can also try other molecules, such as $\mathrm{HeH}^+$ and $\mathrm{H_2O}$.
To visualize the potential energy surface, you can use the `matplotlib` package. 
An example of how to plot the data can be found in the file ./data/plot.ipynb.

### Unrestricted Hartree-Fock
To implement the unrestricted Hartree-Fock (UHF) method, we will create a new function `solve_uhf` that follows the same steps as the restricted Hartree-Fock method, but with the added flexibility of allowing different spin-orbitals to have different occupation numbers. 

The UHF method can be used to study molecules that have broken spin-symmetry, such as those that have a non-zero magnetic moment or unpaired electrons. In this case, the potential energy surface can be different from that obtained from the restricted Hartree-Fock method. 

It's important to note that even if you have implemented the UHF method, the results may not be consistent with the reference. In order to get the correct dissociation potential energy surface, you may need to carefully break the spin symmetry.


### Dependencies
- `numpy`
- `scipy`
- If you wish to use `./data/gen_data.py` to generate the integrals, `pyscf` is also required.

### Reference
- Szabo and Ostlund, _Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory_,
  Dover Publications, New York, 1996
- Helgaker, Jørgensen, and Olsen, _Molecular Electronic-Structure Theory_, Wiley, New York, 2000
