# Hofstadter Model & Wilson Loop Analysis

This repository contains a Python implementation of the **Hofstadter model**, a tight-binding lattice model describing electrons on a 2D lattice in a magnetic field. The code computes energy spectra, visualizes band structures, and analyzes the Hall conductivity using **Wilson loops**.

---

## Physical Background

The Hofstadter model describes electrons moving on a two-dimensional square lattice with a perpendicular uniform magnetic field. The magnetic field affects adds phase factors to the hopping between neighboring sites. The total phase picked up around a lattice plaquette corresponds to a magnetic flux $`\phi`$. When the flux is rational, i.e. $`\phi = 2 \pi \frac{p}{q}`$ for coprime integers $`p, q`$, the magnetic field enlarges the effective unit cell of the lattice in real space. The single-band spectrum splits into $`q`$ subbands. As the flux is varied, these subbands form the intricate and self-similar **Hofstadter butterfly**. Beyond their energies, these subbands have important **topological properties**. Each band can carry a **Chern number**, an integer that measures how the quantum states twist as momentum varies across the Brillouin zone. More precisely, the Chern number is the integer corresponding to the Berry phase along a closed path. These integers are not sensitive to small perturbations and lead to robust physical effects. A key consequence is the **quantized Hall effect**: when the Fermi energy lies in a gap between subbands, the Hall conductivity is quantized and determined by the sum of Chern numbers of the occupied bands.

This code analyzes these topological properties using **Wilson loops**, a practical and gauge-invariant way to compute Berry phases. By tracking how occupied Bloch states evolve along closed paths in momentum space, the phases of the Wilson loop eigenvalues reveal the winding behavior associated with non-zero Chern numbers.

---

## Features

* Construction of the Hofstadter Hamiltonian for rational magnetic flux
* Spectrum computation for coprime flux fractions
* 3D visualization of energy bands in momentum space
* Wilson loop calculations for Berry phase analysis
* Topological characterization of bands (Hall conductivity)

---

## Requirements

Install the required Python packages:

```bash
pip install numpy matplotlib
```

---

## Code Structure

### Hofstadter Hamiltonian

**`make_H(kx, ky, p, q, tb, ta)`**
  Constructs the $`q \times q`$ Hofstadter Hamiltonian at momentum
  $`(k_x, k_y)`$ for magnetic flux

```math
\phi=2\pi\frac{p}{q}.
```
tb and ta are hopping parameters.

---

### Energy Spectrum

**`get_eigenvalues(n, kx, ky, tb, ta)`**
Computes Hamiltonian eigenvalues for all coprime fluxes with denominator up to `n`.

**`plot_spectra(n, p, q, tb, ta)`**
Generates a 3D surface plot of the energy bands as functions of $`k_x`$ and $`k_y`$.

---

### Wilson Loop & Topology

**`w_eigs(...)`**
Computes the eigenvalue phases of the Wilson loop operator for selected bands.

**`plot_phase(...)`**
Plots Wilson loop phases (Berry phases) versus momentum.

**`get_hall_cond(n, p, q, tb, ta)`**
Analyzes the Hall conductivity by computing Wilson loops across the Brillouin zone.

---

## Usage

Run the script directly:

```bash
python main.py
```

### Default Parameters

```python
n = 200        # k-space resolution
ta = 1         # hopping parameter in x-direction
tb = 1         # hopping parameter in y-direction
p = 1          # flux numerator
q = 3          # flux denominator
```

This computes Wilson loop phases for the magnetic flux:

```math
\phi =  2 \pi \frac{1}{3}
```

---

## Notes

* Runtime increases rapidly with larger values of `q` and finer `n`.
* Periodic boundary conditions are assumed.
* Plots are intended for qualitative and educational analysis.

---

## Possible Extensions

* Automatic extraction of Chern numbers
* Hofstadter butterfly visualization
* Performance optimization via vectorization
* Interactive plotting tools
