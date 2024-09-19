from .citrine import SellmeierCoefficients, Crystal, Orientation
from numpy import array

sellmeier_e_ppKTP = SellmeierCoefficients(
    zeroth_order=[2.09930, 0.922683, 0.04676950, 0.0138408],
    first_order=array([6.2897, 6.3061, -6.0629, 2.6486]) * (1e-6),
    second_order=array([-0.14445, 2.2244, -3.5770, 1.3470]) * (1e-8),
    temperature=25,
)

sellmeier_o_ppKTP = SellmeierCoefficients(
    zeroth_order=[
        2.12725,
        1.184310,
        5.14852e-2,
        0.6603000,
        100.00507,
        9.68956e-3,
    ],
    first_order=array([9.9587, 9.9228, -8.9603, 4.1010]) * (1e-6),
    second_order=array([-1.1882, 10.459, -9.8136, 3.1481]) * (1e-8),
    temperature=25,
)

# Define the ppKTP crystal
ppKTP = Crystal(
    name='ppKTP',
    sellmeier_o=sellmeier_o_ppKTP,
    sellmeier_e=sellmeier_e_ppKTP,
    pump_orientation=Orientation.extraordinary,
    signal_orientation=Orientation.extraordinary,
    idler_orientation=Orientation.ordinary,
)

sellmeier_e_KTPrifr = SellmeierCoefficients(
    zeroth_order=None,
    first_order=array([9.9587, 9.9228, -8.9603, 4.1010]) * (1e-6),
    second_order=array([-1.1882, 10.459, -9.8136, 3.1481]) * (1e-8),
    temperature=25,
)


sellmeier_o_KTPrifr = SellmeierCoefficients(
    zeroth_order=None,
    first_order=array([6.2897, 6.3061, -6.0629, 2.6486]) * (1e-6),
    second_order=array([-0.14445, 2.2244, -3.5770, 1.3470]) * (1e-8),
    temperature=25,
)

# Define the ppKTP crystal
KTPrifr = Crystal(
    name='KTPrifr',
    sellmeier_o=sellmeier_o_KTPrifr,
    sellmeier_e=sellmeier_e_KTPrifr,
    pump_orientation=Orientation.extraordinary,
    signal_orientation=Orientation.extraordinary,
    idler_orientation=Orientation.ordinary,
)
