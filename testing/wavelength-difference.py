import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

from citrine.crystals import KTiOPO4_Fradkin
from citrine import (
    Wavelength,
    Magnitude,
    bandwidth_conversion,
    spectral_window,
    calculate_grating_period,
    wavelength_temperature_tuning,
    delta_k_matrix,
    phase_matching_function,
    calculate_jsa_marginals,
)
import citrine.pump_envelope as pef
from seaborn.palettes import color_palette


def main():
    colour_map = color_palette('flare_r', as_cmap=True)
    crystal = KTiOPO4_Fradkin
    lambda_p = Wavelength(775.0, Magnitude.nano)
    lambda_s = Wavelength(1550.0, Magnitude.nano)
    lambda_i = Wavelength(1550.0, Magnitude.nano)

    poling_period = calculate_grating_period(
        lambda_p,
        lambda_s,
        lambda_i,
        crystal,
    )

    print(f'Poling period: {poling_period * 1e6:.2f}µm')

    (temperature, wavelengths_signal, wavelengths_idler) = (
        wavelength_temperature_tuning(
            lambda_p,
            lambda_s,
            lambda_i,
            poling_period,
            crystal,
            (10, 50),
            num_points=3,
        )
    )

    # sigma_lambda_p = Wavelength(1.5, Magnitude.nano)
    # sigma_p = 2 * np.pi * bandwidth_conversion(sigma_lambda_p, lambda_p)
    spectral_width = Wavelength(10, Magnitude.nano)
    steps = 128
    for t, wls, wli in zip(temperature, wavelengths_signal, wavelengths_idler):
        print(t, wls, wli)
        lambda_s_tuned = Wavelength(wls, Magnitude.base)
        lambda_i_tuned = Wavelength(wli, Magnitude.base)
        lambda_s = Wavelength(1550.0, Magnitude.nano)
        lambda_i = Wavelength(1550.0, Magnitude.nano)
        lambda_s_central = lambda_s
        lambda_i_central = lambda_i
        lambda_s = spectral_window(lambda_s_central, spectral_width, steps)
        lambda_i = spectral_window(
            lambda_i_central, spectral_width, steps, reverse=True
        )
        # Calculate Δk matrix
        delta_k = delta_k_matrix(
            lambda_p,
            lambda_s,
            lambda_i,
            crystal,
            temperature=t,
        )

        # Calculate the phase matching function
        phase_matching = phase_matching_function(delta_k, poling_period, 2.9e-2)

        # Calculate the Gaussian pump envelope
        pulse_length = (1.4e-12) * 2

        # tau_p = pulse_length / (2 * np.arccosh(np.sqrt(2)))
        # pump_envelope = pef.sech2(
        #     lambda_p,
        #     tau_p,
        #     lambda_s,
        #     lambda_i,
        # )

        sigma_lambda_p = Wavelength(0.3, Magnitude.nano)
        sigma_p = 2 * np.pi * bandwidth_conversion(sigma_lambda_p, lambda_p)
        pump_envelope = pef.skewed_gaussian(
            lambda_p,
            sigma_p,
            lambda_s,
            lambda_i,
            alpha=2.5,
            # tail_strength=0.5,
            # cutoff_steepness=2,
            # edge_asymmetry=0.5,
        )

        JSA = pump_envelope * phase_matching

        fig, axs = plt.subplots(
            2, 2, figsize=(8, 8), sharex=True, sharey=True, squeeze=True
        )

        # Plot Δk
        axs[0, 0].pcolormesh(
            lambda_i.value,
            lambda_s.value,
            delta_k,
            cmap=colour_map,
            shading='gouraud',
        )
        axs[0, 0].set_title(r'$\Delta k$')
        axs[0, 0].set_xlabel(r'$\lambda_s$ (nm)')
        axs[0, 0].set_ylabel(r'$\lambda_i$ (nm)')

        # Plot the pump envelope
        axs[0, 1].pcolormesh(
            lambda_i.value,
            lambda_s.value,
            np.sqrt(np.abs(pump_envelope**2)),
            cmap=colour_map,
            shading='gouraud',
        )
        axs[0, 1].set_title(r'$\alpha(\lambda_s, \lambda_i)$')
        axs[0, 1].set_xlabel(r'$\lambda_s$ (nm)')
        axs[0, 1].set_ylabel(r'$\lambda_i$ (nm)')

        # Plot the phase matching function
        axs[1, 0].pcolormesh(
            lambda_i.value,
            lambda_s.value,
            phase_matching,
            cmap=colour_map,
            shading='gouraud',
        )
        axs[1, 0].set_title(r'$\phi(\lambda_s, \lambda_i)$')
        axs[1, 0].set_xlabel(r'$\lambda_s$ (nm)')
        axs[1, 0].set_ylabel(r'$\lambda_i$ (nm)')

        # Plot the Joint Spectral Amplitude (JSA)
        axs[1, 1].pcolormesh(
            lambda_i.value,
            lambda_s.value,
            np.real(JSA),
            cmap=colour_map,
            shading='gouraud',
        )
        axs[1, 1].set_title(
            r'$\alpha(\lambda_s, \lambda_i) \phi(\lambda_s, \lambda_i)$'
        )
        axs[1, 1].set_xlabel(r'$\lambda_s$ (nm)')
        axs[1, 1].set_ylabel(r'$\lambda_i$ (nm)')

        # Add overall title
        fig.suptitle(
            r'$\omega_p = {:.2f}\mathrm{{nm}}, \omega_s = {:.2f}\mathrm{{nm}}, \omega_i = {:.2f}\mathrm{{nm}}, \Lambda = {:.2f}\mathrm{{\mu m}}$'.format(
                lambda_p.value,
                lambda_s_tuned.to_unit(Magnitude.nano).value,
                lambda_i_tuned.to_unit(Magnitude.nano).value,
                poling_period * (1e6),
            )
        )

        plt.tight_layout()

        marginal_spectra = calculate_jsa_marginals(JSA, lambda_s, lambda_i)

        fig, axs = plt.subplots()
        axs.plot(
            marginal_spectra['signal_wl'],
            np.abs(marginal_spectra['signal_intensity']),
            label='Signal',
        )
        axs.plot(
            marginal_spectra['idler_wl'],
            np.abs(marginal_spectra['idler_intensity']),
            label='Idler',
        )
        axs.legend()

    # This produces the heatmaps seen here: https://arxiv.org/abs/2012.05134
    # temperature = np.linspace(10, 150, steps)
    # n_i = []
    # k_i = []
    # mismatch = []
    # for i, t in enumerate(temperature):
    #     (n_p, n_s, _n_i) = crystal.refractive_indices(
    #         lambda_p, lambda_s, window, t
    #     )
    #     n_i.append(_n_i)
    #     k_p = (2 * np.pi * n_p) / lambda_p.to_absolute().value
    #     k_s = (2 * np.pi * n_s) / window.to_absolute().value
    #     _k_i = (2 * np.pi * _n_i) / lambda_p.to_absolute().value
    #     k_g = (2 * np.pi) / (poling_period)  # Grating vector
    #     k_i.append(_k_i)
    #     mismatch.append(k_p - k_s - _k_i - k_g)
    # n_i = np.array(n_i)
    # k_i = np.array(k_i)
    # mismatch = np.array(mismatch)
    # n_s = np.zeros([steps, steps])
    # k_s = np.zeros([steps, steps])
    # mismatch = np.zeros([steps, steps])
    # data = []
    # for i, t in enumerate(temperature):
    #     for j, s in enumerate(window.to_absolute().value):
    #         _wl_s = Wavelength(s, Magnitude.base)
    #         (n_p, _n_s, n_i) = crystal.refractive_indices(
    #             lambda_p, _wl_s, lambda_i, t
    #         )
    #         k_p = (2 * np.pi * n_p) / lambda_p.to_absolute().value
    #         _k_s = (2 * np.pi * _n_s) / lambda_s.to_absolute().value
    #         k_i = (2 * np.pi * n_i) / lambda_i.to_absolute().value
    #         k_g = (2 * np.pi) / (poling_period)  # Grating vector
    #         n_s[i, j] = _n_s
    #         k_s[i, j] = _k_s
    #         mismatch[i, j] = k_p - _k_s - k_i - k_g
    #     m = np.abs(mismatch[i, :])
    #     min_idx = np.where(m == np.min(m))[0]
    #     data.append([t, np.mean(window.value[min_idx])])
    # print(n_s)
    # print(k_s)

    _, axes = plt.subplots()
    axes.plot(temperature, wavelengths_signal * 1e6, label='Signal')
    axes.plot(temperature, wavelengths_idler * 1e6, label='Idler')
    axes.set_xlabel('Temperature')
    axes.set_ylabel('Wavelength (nm)')
    axes.legend()

    plt.show()


if __name__ == '__main__':
    main()
