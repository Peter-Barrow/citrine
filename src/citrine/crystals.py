from .citrine import SellmeierCoefficients, Crystal, PhaseMatchingCondition
from numpy import array


KTiOPO4_Fradkin = Crystal(
    name='Potassium Titanyl Phosphate',
    doi='10.1063/1.123408',
    sellmeier_o=SellmeierCoefficients(
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
    ),
    sellmeier_e=SellmeierCoefficients(
        zeroth_order=[2.09930, 0.922683, 0.04676950, 0.0138408],
        first_order=array([6.2897, 6.3061, -6.0629, 2.6486]) * (1e-6),
        second_order=array([-0.14445, 2.2244, -3.5770, 1.3470]) * (1e-8),
        temperature=25,
    ),
    phase_matching=PhaseMatchingCondition.type2_o,
)

KTiOPO4_Emanueli = Crystal(
    name='Potassium Titanyl Phosphate',
    doi='10.1364/AO.42.006661',
    sellmeier_o=SellmeierCoefficients(
        zeroth_order=None,
        first_order=array([6.2897, 6.3061, -6.0629, 2.6486]) * (1e-6),
        second_order=array([-0.14445, 2.2244, -3.5770, 1.3470]) * (1e-8),
        temperature=25,
    ),
    sellmeier_e=SellmeierCoefficients(
        zeroth_order=None,
        first_order=array([9.9587, 9.9228, -8.9603, 4.1010]) * (1e-6),
        second_order=array([-1.1882, 10.459, -9.8136, 3.1481]) * (1e-8),
        temperature=25,
    ),
    phase_matching=PhaseMatchingCondition.type2_o,
)

KTiOAsO4_Emanueli = Crystal(
    name='Potassium Titanyl Aresnate',
    doi='10.1364/AO.42.006661',
    sellmeier_o=SellmeierCoefficients(
        zeroth_order=None,
        first_order=array([-4.1053, 44.261, -38.012, 11.302]) * (1e-6),
        second_order=array([0.5857, 3.9386, -4.0081, 1.4316]) * (1e-8),
        temperature=25,
    ),
    sellmeier_e=SellmeierCoefficients(
        zeroth_order=None,
        first_order=array([-6.1537, 64.505, -56.447, 17.169]) * (1e-6),
        second_order=array([-0.96751, 13.192, -11.78, 3.6292]) * (1e-8),
        temperature=25,
    ),
    phase_matching=PhaseMatchingCondition.type2_o,
)

LiNbO3_Zelmon = Crystal(
    name='Lithium Niobate',
    doi='10.1364/JOSAB.14.003319',
    sellmeier_o=SellmeierCoefficients(
        zeroth_order=None,
        first_order=array([2.9804, 0.02047, 0.5981, 0.0666, 8.9543, 416.08]),
        second_order=None,
        temperature=21,
    ),
    sellmeier_e=SellmeierCoefficients(
        zeroth_order=None,
        first_order=array([2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.6]),
        second_order=None,
        temperature=21,
    ),
    phase_matching=PhaseMatchingCondition.type0_e,
)

LiNbO3_5molMgO_Zelmon = Crystal(
    name='Lithium Niobate 5Mol Magnesium Doped',
    doi='10.1364/JOSAB.14.003319',
    sellmeier_o=SellmeierCoefficients(
        zeroth_order=None,
        first_order=array(
            [2.2454, 0.01242, 1.3005, 0.0513, 6.8972, 331.33],
        ),
        second_order=None,
        temperature=21,
    ),
    sellmeier_e=SellmeierCoefficients(
        zeroth_order=None,
        first_order=array(
            [2.4272, 0.01478, 1.4617, 0.005612, 9.6536, 371.216],
        ),
        second_order=None,
        temperature=21,
    ),
    phase_matching=PhaseMatchingCondition.type0_e,
)
