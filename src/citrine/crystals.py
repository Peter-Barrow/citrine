from .citrine import (
    Crystal,
    PhaseMatchingCondition,
    SellmeierCoefficientsSimple,
    SellmeierCoefficientsTemperatureDependent,
    SellmeierCoefficientsJundt,
    SellmeierCoefficientsBornAndWolf,
)
from numpy import array

__all__ = [
    'KTiOPO4_Fradkin',
    'KTiOPO4_Emanueli',
    'KTiOAsO4_Emanueli',
    'LiNbO3_Zelmon',
    'LiNbO3_5molMgO_Zelmon',
    'LiNbO3_Newlight',
    'LiNbO3_5molMgO_Gayer',
]


KTiOPO4_Fradkin = Crystal(
    name='Potassium Titanyl Phosphate',
    doi='10.1063/1.123408',
    sellmeier_o=SellmeierCoefficientsTemperatureDependent(
        zeroth_order=array([
            2.12725,
            1.184310,
            5.14852e-2,
            0.6603000,
            100.00507,
            9.68956e-3,
        ]),
        first_order=array([9.9587, 9.9228, -8.9603, 4.1010]) * (1e-6),
        second_order=array([-1.1882, 10.459, -9.8136, 3.1481]) * (1e-8),
        temperature=25,
    ),
    sellmeier_e=SellmeierCoefficientsTemperatureDependent(
        zeroth_order=array([2.09930, 0.922683, 0.04676950, 0.0138408]),
        first_order=array([6.2897, 6.3061, -6.0629, 2.6486]) * (1e-6),
        second_order=array([-0.14445, 2.2244, -3.5770, 1.3470]) * (1e-8),
        temperature=25,
    ),
    phase_matching=PhaseMatchingCondition.type2_o,
)

KTiOPO4_Emanueli = Crystal(
    name='Potassium Titanyl Phosphate',
    doi='10.1364/AO.42.006661',
    sellmeier_o=SellmeierCoefficientsTemperatureDependent(
        zeroth_order=None,
        first_order=array([6.2897, 6.3061, -6.0629, 2.6486]) * (1e-6),
        second_order=array([-0.14445, 2.2244, -3.5770, 1.3470]) * (1e-8),
        temperature=25,
    ),
    sellmeier_e=SellmeierCoefficientsTemperatureDependent(
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
    sellmeier_o=SellmeierCoefficientsTemperatureDependent(
        zeroth_order=None,
        first_order=array([-4.1053, 44.261, -38.012, 11.302]) * (1e-6),
        second_order=array([0.5857, 3.9386, -4.0081, 1.4316]) * (1e-8),
        temperature=25,
    ),
    sellmeier_e=SellmeierCoefficientsTemperatureDependent(
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
    sellmeier_o=SellmeierCoefficientsBornAndWolf(
        coefficients=array([2.9804, 0.02047, 0.5981, 0.0666, 8.9543, 416.08]),
        temperature=21,
    ),
    sellmeier_e=SellmeierCoefficientsBornAndWolf(
        coefficients=array([2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.6]),
        temperature=21,
    ),
    phase_matching=PhaseMatchingCondition.type0_e,
)

LiNbO3_5molMgO_Zelmon = Crystal(
    name='Lithium Niobate 5Mol Magnesium Doped',
    doi='10.1364/JOSAB.14.003319',
    sellmeier_o=SellmeierCoefficientsBornAndWolf(
        coefficients=array(
            [2.2454, 0.01242, 1.3005, 0.0513, 6.8972, 331.33],
        )
        * (1e-6),
        temperature=21,
    ),
    sellmeier_e=SellmeierCoefficientsBornAndWolf(
        coefficients=array(
            [2.4272, 0.01478, 1.4617, 0.005612, 9.6536, 371.216],
        )
        * (1e-6),
        temperature=21,
    ),
    phase_matching=PhaseMatchingCondition.type0_e,
)

LiNbO3_Newlight = Crystal(
    name='Lithium Niobate',
    doi='',
    sellmeier_o=SellmeierCoefficientsSimple(
        coefficients=array([4.9048, 0.11768, -0.04750, -0.027169]),
        temperature=20,
        dn_dt=-0.874e-6,
    ),
    sellmeier_e=SellmeierCoefficientsSimple(
        coefficients=array([4.5820, 0.099169, -0.04443, -0.021950]),
        temperature=20,
        dn_dt=39.073e-6,
    ),
    phase_matching=PhaseMatchingCondition.type0_e,
)

LiNbO3_5molMgO_Gayer = Crystal(
    name='Lithium Niobate 5Mol Magnesium Doped',
    doi=' 10.1007/s00340-008-2998-2',
    sellmeier_e=SellmeierCoefficientsJundt(
        a_terms=array([5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32e-2]),
        b_terms=array([2.860e-6, 4.700e-8, 6.113e-8, 1.516e-4]),
        temperature=24.5,
    ),
    sellmeier_o=SellmeierCoefficientsJundt(
        a_terms=array([5.653, 0.1185, 0.2091, 89.61, 10.85, 1.97e-2]),
        b_terms=array([7.941e-7, 3.134e-8, -4.641e-9, -2.188e-6]),
        temperature=24.5,
    ),
    phase_matching=PhaseMatchingCondition.type0_e,
)
