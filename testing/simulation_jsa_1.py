from citrine import (
    Wavelength,
    Magnitude,
    spectral_window,
    bandwidth_conversion,
    delta_k_matrix,
    calculate_grating_period,
    phase_matching_function,
    calculate_jsa_marginals,
    PhaseMatchingCondition,
)

# from citrine import *
from citrine.crystals import KTiOPO4_Fradkin
import citrine.pump_envelope as pef
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from seaborn.palettes import color_palette


matplotlib.use('TkAgg')


def simulation():
    """
    Run the simulation for calculating the JSA, pump envelope, phase matching function,
    and delta k matrix. It also generates plots for visualizing each of these functions.
    """

    # Define central wavelengths for pump, signal, and idler
    lambda_p_central = Wavelength(775, Magnitude.nano)
    lambda_s_central = Wavelength(1550.1, Magnitude.nano)
    lambda_i_central = Wavelength(1549.9, Magnitude.nano)

    # Define a range of wavelengths for signal and idler (in nm)
    spectral_width = Wavelength(5, Magnitude.nano)
    steps = 128
    lambda_s = spectral_window(lambda_s_central, spectral_width, steps)
    lambda_i = spectral_window(
        lambda_i_central, spectral_width, steps, reverse=True
    )

    # Define pump bandwidth (in nm and convert)
    sigma_lambda_p = Wavelength(1.5, Magnitude.nano)
    sigma_p = 2 * np.pi * bandwidth_conversion(sigma_lambda_p, lambda_p_central)

    temperature = 45

    ppKTP = KTiOPO4_Fradkin
    ppKTP.phase_matching = PhaseMatchingCondition.type2_o

    # Calculate grating period
    grating_period = calculate_grating_period(
        lambda_p_central,
        lambda_s_central,
        lambda_i_central,
        ppKTP,
        temperature=temperature,
    )

    # Calculate Δk matrix
    delta_k = delta_k_matrix(
        lambda_p_central,
        lambda_s,
        lambda_i,
        ppKTP,
        temperature=temperature,
    )

    # Calculate the phase matching function
    phase_matching = phase_matching_function(delta_k, grating_period, 2.9e-2)

    # Calculate the Gaussian pump envelope
    pulse_length = (1.4e-12) * 2

    tau_p = pulse_length / (2 * np.arccosh(np.sqrt(2)))
    sigma_p = (2 * np.sqrt(np.log(2))) / pulse_length
    sigma_lambda_p = Wavelength(0.3, Magnitude.nano)
    sigma_p = 2 * np.pi * bandwidth_conversion(sigma_lambda_p, lambda_p_central)
    # pump_envelope = pef.sech2(
    #     lambda_p_central,
    #     tau_p,
    #     lambda_s,
    #     lambda_i,
    # )
    # pump_envelope = pef.gaussian(
    #     lambda_p_central,
    #     sigma_p,
    #     lambda_s,
    #     lambda_i,
    # )
    lambda_p_central_shifted = Wavelength(774.95, Magnitude.nano)
    pump_envelope = pef.skewed_gaussian(
        lambda_p_central,
        sigma_p,
        lambda_s,
        lambda_i,
        alpha=2.5,
        # tail_strength=0.5,
        # cutoff_steepness=2,
        # edge_asymmetry=0.5,
    )
    # pump_envelope_b = pef.skewed_gaussian(
    #     lambda_p_central,
    #     sigma_p,
    #     lambda_s,
    #     lambda_i,
    #     alpha=-1.0,
    #     # tail_strength=0.5,
    #     # cutoff_steepness=2,
    #     # edge_asymmetry=0.5,
    # )
    # pump_envelope += pump_envelope_b

    # pump_envelope = np.pump_envelope
    # pump_envelope = pump_envelope_gaussian(
    #     lambda_p_central, sigma_p, lambda_s, lambda_i
    # )

    # Calculate the Joint Spectral Amplitude (JSA)
    JSA = pump_envelope * phase_matching
    JSI = np.abs(JSA)

    # Plot the results
    fig, axs = plt.subplots(
        2, 2, figsize=(8, 8), sharex=True, sharey=True, squeeze=True
    )

    colour_map = 'magma'
    colour_map = color_palette('rocket', as_cmap=True)
    colour_map = 'viridis'
    colour_map = color_palette('flare_r', as_cmap=True)

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
            lambda_p_central.value,
            lambda_s_central.value,
            lambda_i_central.value,
            grating_period * (1e6),
        )
    )

    # Show plot
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

    # fig, axs = plt.subplots()
    # axs.pcolormesh(
    #     lambda_i.value,
    #     lambda_s.value,
    #     delta_k,
    #     cmap=colour_map,
    #     shading='gouraud',
    # )
    # axs.set_title(r'$\Delta k$')
    # axs.set_xlabel(r'$\lambda_s$ (nm)')
    # axs.set_ylabel(r'$\lambda_i$ (nm)')

    # fig, axs = plt.subplots()
    # # Plot the pump envelope
    # axs.pcolormesh(
    #     lambda_i.value,
    #     lambda_s.value,
    #     pump_envelope,
    #     cmap=colour_map,
    #     shading='gouraud',
    # )
    # axs.set_title(r'$\alpha(\lambda_s, \lambda_i)$')
    # axs.set_xlabel(r'$\lambda_s$ (nm)')
    # axs.set_ylabel(r'$\lambda_i$ (nm)')

    # fig, axs = plt.subplots()
    # # Plot the phase matching function
    # axs.pcolormesh(
    #     lambda_i.value,
    #     lambda_s.value,
    #     phase_matching,
    #     cmap=colour_map,
    #     shading='gouraud',
    # )
    # axs.set_title(r'$\phi(\lambda_s, \lambda_i)$')
    # axs.set_xlabel(r'$\lambda_s$ (nm)')
    # axs.set_ylabel(r'$\lambda_i$ (nm)')

    # fig, axs = plt.subplots()
    # # Plot the Joint Spectral Amplitude (JSA)
    # axs.pcolormesh(
    #     lambda_i.value, lambda_s.value, JSA, cmap=colour_map, shading='gouraud'
    # )
    # axs.set_title(r'$\alpha(\lambda_s, \lambda_i) \phi(\lambda_s, \lambda_i)$')
    # axs.set_xlabel(r'$\lambda_s$ (nm)')
    # axs.set_ylabel(r'$\lambda_i$ (nm)')

    # # Next calculate HOM dip

    # delays = Time(np.linspace(-30, 30, 256), Magnitude.pico)
    # hom_probabilities, purity, schmidt_number, entropy = (
    #     hong_ou_mandel_interference(JSA, lambda_s, lambda_i, delays)
    # )

    # fig = plt.figure()
    # plt.plot(delays.value, hom_probabilities)
    # plt.xlabel(f'Delay ({delays.unit.name} seconds)')
    # plt.ylabel('Probability')
    # plt.title(
    #     f'Purity:{purity:.2f} | Schmidt Number:{schmidt_number:.2f} | Entropy:{entropy:.2f} (JSI)'
    # )

    # hom_probabilities, purity, schmidt_number, entropy = (
    #     hong_ou_mandel_interference(JSI, lambda_s, lambda_i, delays)
    # )

    # fig = plt.figure()
    # plt.plot(delays.value, hom_probabilities)
    # plt.xlabel(f'Delay ({delays.unit.name} seconds)')
    # plt.ylabel('Probability')
    # plt.title(
    #     f'Purity:{purity:.2f} | Schmidt Number:{schmidt_number:.2f} | Entropy:{entropy:.2f} (JSA)'
    # )

    # fig, axs = plt.subplots()
    # # Plot the Joint Spectral Amplitude (JSA)
    # axs.pcolormesh(
    #     lambda_i.value,
    #     lambda_s.value,
    #     JSI,
    #     cmap='magma',  # colour_map,
    #     shading='gouraud',
    # )
    # axs.set_title('Joint Spectral Intensity')
    # axs.set_xlabel(r'$\lambda_s$ (nm)')
    # axs.set_ylabel(r'$\lambda_i$ (nm)')

    plt.show()


if __name__ == '__main__':
    simulation()


# import sys
#
# sys.path.append('/home/bp38/Projects/citrine/src/')
# from citrine import *
# from simulation_jsa_1 import simulation
#
# simulation()


# import sys
# sys.path.append('/home/bp38/Projects/citrine/src/')
# from citrine import *
# import numpy as np
# import matplotlib.pyplot as plt
# lambda_p_central = Wavelength(775, Magnitude.nano)
# lambda_s_central = Wavelength(1550, Magnitude.nano)
# lambda_i_central = Wavelength(1550, Magnitude.nano)
# spectral_width = Wavelength(25, Magnitude.nano)
# steps = 256
# lambda_s = spectral_window(lambda_s_central, spectral_width, steps)
# lambda_i = spectral_window(lambda_i_central, spectral_width, steps)
# sigma_lambda_p = Wavelength(0.5, Magnitude.nano)
# tau_p = 1.3e-12
# sigma_p = tau_p / (2 * np.arccosh(np.sqrt(2)))
# pump_envelope = pump_envelope_sech2(
#     lambda_p_central, sigma_p, lambda_s, lambda_i
# )
# fig = plt.figure()
# plt.pcolormesh(pump_envelope)
# plt.show()
