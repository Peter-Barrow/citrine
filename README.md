# Welcome to citrine 💎

*citrine* is a small Python library to calculate the joint-spectra of nonlinear crystals used in parametric downconversion sources.

## About


## Features
- Calculate grating periods
- Produce Δk matrices for different crystals and phase matching conditions
- Pump envelope functions
- Phase matching functions
- Types for commonly used units
    - e.g. ```citrine.Wavelength```
- Library of crystal parameters
    - ``` citrine.crystals ``` sub-module


## Requirements
- numpy

## Installation

```sh title="pypi"
python3 -m pip install citrine
```

### Alternatively from Git
Install from git for the latest version
```sh title="git"
python3 -m pip install git+https://github.com/Peter-Barrow/citrine.git
```

## Quick Examples

``` python
import citrine
import numpy as np
import matplotlib.pyplot as plt

# Define central wavelengths for pump, signal, and idler
lambda_p_central = citrine.Wavelength(775, Magnitude.nano)
lambda_s_central = citrine.Wavelength(1550, Magnitude.nano)
lambda_i_central = citrine.Wavelength(1550, Magnitude.nano)

# Define a range of wavelengths for signal and idler (in nm)
spectral_width = citrine.Wavelength(25, Magnitude.nano)
steps = 1024
lambda_s = citrine.spectral_window(lambda_s_central, spectral_width, steps)
lambda_i = citrine.spectral_window(lambda_i_central, spectral_width, steps)

# Define pump bandwidth (in nm and convert)
sigma_lambda_p = citrine.Wavelength(0.5, Magnitude.nano)
sigma_p = 2 * np.pi * citrine.bandwidth_conversion(sigma_lambda_p, lambda_p_central)

# Calculate grating period
grating_period = citrine.calculate_grating_period(
    lambda_p_central,
    lambda_s_central,
    lambda_i_central,
    citrine.crystals.ppKTP,
)

# Calculate Δk matrix
delta_k = citrine.delta_k_matrix(
    lambda_p_central,
    lambda_s,
    lambda_i,
    ppKTP,
)

# Calculate the phase matching function
phase_matching = citrine.phase_matching_function(delta_k, grating_period, 1e-2)

# Calculate the Gaussian pump envelope
pump_envelope = citrine.pump_envelope_gaussian(lambda_p_central, sigma_p, lambda_s, lambda_i)

# Calculate the Joint Spectral Amplitude (JSA)
JSA = pump_envelope * phase_matching

# Plotting
fig, axs = plt.subplots(
    2, 2, figsize=(8, 8), sharex=True, sharey=True, squeeze=True
)

# Plot Δk
axs[0, 0].pcolormesh(
    lambda_i.value, lambda_s.value, delta_k, cmap='viridis'
)
axs[0, 0].set_title(r'$\Delta k$')
axs[0, 0].set_xlabel(r'$\lambda_s$ (nm)')
axs[0, 0].set_ylabel(r'$\lambda_i$ (nm)')

# Plot the pump envelope
axs[0, 1].pcolormesh(
    lambda_i.value,
    lambda_s.value,
    pump_envelope,
    cmap='viridis',
)
axs[0, 1].set_title(r'$\alpha(\lambda_s, \lambda_i)$')
axs[0, 1].set_xlabel(r'$\lambda_s$ (nm)')
axs[0, 1].set_ylabel(r'$\lambda_i$ (nm)')

# Plot the phase matching function
axs[1, 0].pcolormesh(
    lambda_i.value,
    lambda_s.value,
    phase_matching,
    cmap='viridis',
)
axs[1, 0].set_title(r'$\phi(\lambda_s, \lambda_i)$')
axs[1, 0].set_xlabel(r'$\lambda_s$ (nm)')
axs[1, 0].set_ylabel(r'$\lambda_i$ (nm)')

# Plot the Joint Spectral Amplitude (JSA)
axs[1, 1].pcolormesh(lambda_i.value, lambda_s.value, JSA, cmap='viridis')
axs[1, 1].set_title(
    r'$\alpha(\lambda_s, \lambda_i) \phi(\lambda_s, \lambda_i)$'
)
axs[1, 1].set_xlabel(r'$\lambda_s$ (nm)')
axs[1, 1].set_ylabel(r'$\lambda_i$ (nm)')

# Add overall title
fig.suptitle(
    r'$\omega_p = {:.2f}\mathrm{{nm}}, \omega_s = {:.2f}\mathrm{{nm}}, \omega_i = {:.2f}\mathrm{{nm}}, \Lambda = {:.2f}\mathrm{{\mu m}}$'.format(
        lambda_p_central.value,
        lambda_s_central.value,
        lambda_i_central.value,
        grating_period * (1e6), # convert to microns for printing
    )
)

# Show plot
plt.tight_layout()
plt.show()

```
