### Introduction
This is a tutorial on how to implement a simple program
to solve the Hartree-Fock with SCF iterations.

### Impement the SCF procedure
We will be following the algorithm described in Szabo and Ostlund, 3.4.6, p. 146 to implement the Hartree-Fock method for the hydrogen molecule. The steps are as follows:

1. Obtain a guess at the density matrix.

$$ H_{core} C_{guess} = S C_{guess} \epsilon $$

$$ P_{\lambda \sigma} = 2 \sum_{ij}^{occ} C_{\lambda i} C_{\sigma j} $$

2. Calculate the exchange and coulomb matrices from the density matrix and the two-electron repulsion integrals.

$$
\begin{aligned}
G_{\mu \nu} &= 2J_{\mu \nu} - K_{\mu \nu} \\
&= \sum_{\lambda \sigma} P_{\lambda \sigma} \left( 2 (\mu \nu | \lambda \sigma) - (\mu \sigma | \lambda \nu) \right)
\end{aligned}
$$

3. Add exchange and coulomb matrices to the core-Hamiltonian to obtain the Fock matrix.

$$ F = H_{core} + G $$

4. Diagonalize the Fock matrix.(use [eigh](https://docs.scipy.org/doc/scipy//reference/generated/scipy.linalg.eig.html))

$$ F C = S C \epsilon $$

5. Select the occupied orbitals and calculate the new density matrix.

$$ P_{\lambda \sigma} = 2 \sum_{ij}^{occ} C_{\lambda i} C_{\sigma j} $$

6. Compute the energy.

$$
\begin{aligned}
&E = V_\mathrm{nuc} + E_\mathrm{e}  \\
&V_\mathrm{nuc} = \sum_{i &lt; j}^{N} \frac{Z_{i} Z_{j}}{R_{ij}}  \\
&E_e = \frac{1}{2} tr \left[ P (F + H_{core}) \right]
\end{aligned}
$$


7.  Compute the errors and check for convergence.
  - If converged, return the energy.
  - If not converged, return to the second step with the new density matrix.

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
The unrestricted Hartree-Fock (UHF) method can be used to study molecules that have broken spin-symmetry, such as those that have a non-zero magnetic moment or unpaired electrons. In this case, the potential energy surface can be different from that obtained from the restricted Hartree-Fock method. 

To implement the UHF method, we will create a new function `solve_uhf` that follows the same steps as the restricted Hartree-Fock method, but with the added flexibility of allowing different spin-orbitals to have different occupation numbers. 

There have something different between RHF and UHF. In UHF method, you need to solve two generalized eigenvalue problems:

$$
\begin{cases}
  F^{\alpha} C^{\alpha} = S C^{\alpha} \epsilon^{\alpha}  \\
  F^{\beta} C^{\beta} = S C^{\beta} \epsilon^{\beta}  \\
\end{cases}
$$

And the exchange and coulomb matrices become to

$$
\begin{cases}
G^{\alpha}_{\mu \nu} = \sum_{\lambda \sigma} \left[ P_{\lambda \sigma}^{t} (\mu \nu | \lambda \sigma) - P_{\lambda \sigma}^{\alpha} (\mu \sigma | \lambda \nu) \right]  \\
G^{\beta}_{\mu \nu} = \sum_{\lambda \sigma} \left[ P_{\lambda \sigma}^{t} (\mu \nu | \lambda \sigma) - P_{\lambda \sigma}^{\beta} (\mu \sigma | \lambda \nu) \right]  \\
P^{t} = P^{\alpha} + P^{\beta}
\end{cases}
$$

The Fock matrix building is the same as RHF, just add $G^{\alpha}$ and $G^{\beta}$ to $H_{core}$.

If you use the one-electron matrix guess to calculate molecules that all electrons are paired, such as hydrogen, water, methane, etc. You will find that $P^{\alpha}$ is equal to $P^{\beta}$. Therefore, we need to break the symmetry between the alpha orbital and the beta orbital.

The simplest method is to mix the HOMO and the LUMO use this formula if they are orthonormal

$$
\begin{cases}
\psi_{homo}^{\alpha} = -\psi_{lumo}^{\beta} = \frac{1}{\sqrt{2}} \left( \psi_{homo} + \psi_{lumo} \right) \\
\psi_{lumo}^{\alpha} = -\psi_{homo}^{\beta} = \frac{1}{\sqrt{2}} \left( \psi_{lumo} - \psi_{homo} \right) \\
\end{cases}
$$

It's important to note that even if you have implemented the UHF method, the results may not be consistent with the reference. In order to get the correct dissociation potential energy surface, you may need to carefully break the spin symmetry.

### Tips

When you first multiply two matrices, you maybe calculate it in a loop. But in numpy, you can improve it with [einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html). Such as an n-col k-row matrix A times a k-col m-row matrix B got an n-col m-row matrix C can be written like

```python
    numpy.eninsum("nk, km -> nm", A, B, C)
```

It means

$$
\begin{aligned}
A B &= C  \\
\{ a_{nk} \} \{ b_{km} \} &= \{ c_{nm} \}  \\
\end{aligned}
$$

### Dependencies
- `numpy`
- `scipy`
- If you wish to use `./data/gen_data.py` to generate the integrals, `pyscf` is also required.

### Reference
- Szabo and Ostlund, _Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory_,
  Dover Publications, New York, 1996
- Helgaker, Jørgensen, and Olsen, _Molecular Electronic-Structure Theory_, Wiley, New York, 2000
