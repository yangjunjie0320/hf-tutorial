# Hartree-Fock Tutorial
[![CI](https://github.com/yangjunjie0320/hf-tutorial/actions/workflows/ci.yml/badge.svg)](https://github.com/yangjunjie0320/hf-tutorial/actions/workflows/ci.yml)

This tutorial demonstrates how to implement a simple program to solve the Hartree-Fock equations using self-consistent field (SCF) iterations.

## Setup and Dependencies

Begin by cloning the repository:

```bash
git clone https://github.com/yangjunjie0320/hf-tutorial.git
cd hf-tutorial
```

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

The core dependencies are:
- `numpy`
- `scipy`
- `h5py`
- `line_profiler` (for profiling)

> **Note**: If you wish to use `./data/gen_data.py` to generate the integrals, `pyscf` is also required.

You can verify the setup by running `main.py`:

```bash
python main.py
```

You should see:
```bash
Great! We are ready to solve the Hartree-Fock equations...
...
RuntimeError: SCF is not running, fill in the code in the main loop.
```

This confirms that the framework is set up correctly and ready for you to implement the SCF procedure.
Find the function that raises the `RuntimeError` and implement the SCF procedure in it.

There is a loose style check with `ruff`, you can run it with:
```bash
ruff check .
```
Follow the suggestions to fix the style issues if you want.

## The SCF Algorithm

We will follow the algorithm described in Szabo and Ostlund, Section 3.4.6 (p. 146) to implement the Hartree-Fock method. The SCF procedure consists of the following steps.

> **Note**: github could not render `\mathbf` correctly, check the raw markdown file.

### Step 1: Initial Guess

Obtain an initial guess of the Fock matrix using the core Hamiltonian (this is called the "1e guess" in some software; you can use a smarter guess if desired). Set $\mathbf{F} = \mathbf{H}_{\text{core}}$.

### Step 2: Diagonalize the Fock Matrix

Solve the generalized eigenvalue problem with [`scipy.linalg.eigh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html), then select the occupied orbitals and calculate the new density matrix:

$$ \mathbf{F} \mathbf{C} = \mathbf{S} \mathbf{C} \epsilon \quad \text{and} \quad 
\mathbf{C}_{\text{occ}} \leftarrow \mathbf{C} \quad \text{then} \quad
P_{\mu \nu} = 2 \sum_{i \in \text{occ}} C_{\mu i} C_{\nu i} $$

### Step 3: Build Coulomb and Exchange Matrices

Calculate the Coulomb $\mathbf{J}$ and exchange $\mathbf{K}$ matrices from the density matrix $\mathbf{P}$ and the two-electron repulsion integrals:

$$
J_{\mu \nu} = \sum_{\lambda \sigma} P_{\lambda \sigma} \left( \mu \nu | \lambda \sigma \right)
\quad \text{and} \quad
K_{\mu \nu} = \sum_{\lambda \sigma} P_{\lambda \sigma} \left( \mu \sigma | \lambda \nu \right)
$$

### Step 4: Build the Fock Matrix

Add the Coulomb and exchange matrices to the core Hamiltonian to obtain the Fock matrix:

$$ \mathbf{F} = \mathbf{H}_{\text{core}} + \mathbf{J} - \frac{1}{2} \mathbf{K} $$

### Step 5: Compute the Energy

Compute the energy from the density matrix and Fock matrix:

$$ E = E_{\text{nuc}} + \frac{1}{2} \text{tr} \left( \mathbf{P} \mathbf{F} + \mathbf{P} \mathbf{H}_{\text{core}} \right) $$

### Step 6: Check Convergence

Compute the errors and check for convergence:
- **If converged**: Finish the calculation and return the energy and density matrix.
- **If not converged**: Return to Step 2 with the new density matrix.

There might be some cases that SCF is hard to converge, you can try to use [DIIS](https://vergil.chemistry.gatech.edu/static/content/diis.pdf) 
to accelerate the convergence.

## Code Framework

This project provides a simple framework for implementing the SCF procedure. The code is organized as follows:

- **`main.py`**: The main entry point of the program. Contains the framework for implementing the SCF procedure:
  - `solve_rhf()`: Template function for implementing the restricted Hartree-Fock (RHF) method
  - `solve_uhf()`: Template function for implementing the unrestricted Hartree-Fock (UHF) method
  - `main()`: Loads molecular integrals and runs the SCF calculation

- **`sol.py`**: Reference implementation of the RHF method. Check it if you are stuck.

- **`data/gen_data.py`**: Script to generate the one-electron and two-electron integrals for molecules (e.g., $\text{H}_2$, $\text{HeH}^+$, $\text{H}_2\text{O}$) using PySCF.

- **`data/plot.ipynb`**: Jupyter notebook to visualize the potential energy surface by plotting energy vs. internuclear distance. Use as a reference to plot your own results.

- **`test/test_rhf.py`**: Test script to verify the correctness of your SCF implementation by comparing against reference energies.

## Potential Energy Surface

After implementing the self-consistent field (SCF) procedure, you can use it to compute the potential energy surface of molecules. For example, to compute the potential energy surface of the $\text{H}_2$ molecule, vary the internuclear distance and compute the energy at each point:

```python
for r in numpy.linspace(0.5, 3.0, 61):
    inp = f"h2-{r:.4f}"
    e = main(inp)
    print(f"H2: r={r: 6.4f}, e={e: 12.8f}")
```

You can also try other molecules, such as $\text{HeH}^+$ and $\text{H}_2\text{O}$.

To visualize the potential energy surface, you can use the `matplotlib` package. An example of how to plot the data can be found in `./data/plot.ipynb`.

## Unrestricted Hartree-Fock

The unrestricted Hartree-Fock (UHF) method can be used to study molecules with broken spin-symmetry, such as those with a non-zero magnetic moment or unpaired electrons. In this case, the potential energy surface can be different from that obtained using the restricted Hartree-Fock method.

To implement the UHF method, create a new function `solve_uhf` that follows the same steps as the restricted Hartree-Fock method, but with the added flexibility of allowing different spin-orbitals to have different occupation numbers.

> **Important**: Even if you have implemented the UHF method, the results may not be consistent with the reference. To obtain the correct dissociation potential energy surface, you may need to carefully break the spin symmetry with a smart initial guess.

## Performance Tips

Calculating the Coulomb and exchange matrices is the most time-consuming part of the SCF procedure (not only in this tutorial, but also in real-world applications).

### Getting Started

You can start with [`numpy.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) (setting `optimize=True` might help). It offers an intuitive way to express the contraction of two tensors, though it may not always be the most efficient approach in terms of performance.

```python
# einsum notation will contract the repeated indices automatically
# J_{\mu \nu} = \sum_{\lambda \sigma} P_{\lambda \sigma} (\mu \nu | \lambda \sigma)
# p ~ mu, q ~ nu, r ~ lambda, s ~ sigma
coul = numpy.einsum("pqrs,rs->pq", eri, dm, optimize=True)
exch = -numpy.einsum("prqs,rs->pq", eri, dm, optimize=True) / 2.0
```

The equivalent Python loop is:
```python
nao = dm.shape[0]
coul = numpy.zeros((nao, nao))
for mu in range(nao):
    for nu in range(nao):
        for lm in range(nao):
            for sg in range(nao):
                coul[mu, nu] += eri[mu, nu, lm, sg] * dm[lm, sg]
```

### Modularize the Code

Begin by moving the matrix computation to standalone functions:
```python
def compute_coulomb_matrix_v0(eri, dm):
    nao = dm.shape[0]
    coul = numpy.zeros((nao, nao))
    for mu in range(nao):
        for nu in range(nao):
            for lm in range(nao):
                for sg in range(nao):
                    coul[mu, nu] += eri[mu, nu, lm, sg] * dm[lm, sg]
    return coul

def compute_coulomb_matrix_v1(eri, dm):
    nao = dm.shape[0]
    coul = numpy.einsum("pqrs,rs->pq", eri, dm, optimize=True)
    return coul
```
These functions can be tested and benchmarked separately, making optimization easier.
You will see that the nested loop is much slower than the `einsum` version. If you are ambitious, try implementing it with a Python loop first to understand the algorithm (for both Coulomb and exchange matrices).

Then use a profiler like `line_profiler` to identify the bottlenecks:
```bash
kernprof -l -v main.py
```

### Optimize with Matrix Operations

Try converting the `einsum` to more efficient matrix operations using `numpy.dot` or `numpy.tensordot`:
```python
def compute_coulomb_matrix_v2(eri, dm):
    """Alternative implementation using reshape and dot"""
    nao = dm.shape[0]
    eri_2d = eri.reshape(nao * nao, nao * nao)
    dm_1d = dm.reshape(nao * nao)
    coul_1d = numpy.dot(eri_2d, dm_1d)
    return coul_1d.reshape(nao, nao)
```

> **Note**: The reshape approach may not always be faster than `einsum` with `optimize=True`, as `einsum` uses optimized BLAS routines internally. Benchmark to compare!

### Use Advanced Optimization Tools

For production code or when dealing with large systems, consider using powerful optimization tools:

- **`numba`**: Just-in-time (JIT) compilation for Python code. Simply add decorators to your functions for significant speedups with minimal code changes.

- **`cython`**: Compile Python extensions to C for better performance. Requires writing Cython code but offers fine-grained control over optimization.

- **`C/C++`**: Native kernel functions for maximum control and performance. Use OpenMP (e.g., `#pragma omp parallel for`) for easy parallelization across CPU cores.

- **`CUDA`**: GPU kernel functions for massive parallelization.

## References

- Szabo and Ostlund, _Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory_, Dover Publications, New York, 1996
- Helgaker, Jørgensen, and Olsen, _Molecular Electronic-Structure Theory_, Wiley, New York, 2000
