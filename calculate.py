from policyengine_us import Microsimulation
from policyengine_core.reforms import Reform
import pandas as pd
import numpy as np
from itertools import product
import os
from huggingface_hub import hf_hub_download, login, HfApi
import os

def upload(local_file_path: str, repo: str, repo_file_path: str):
    token = os.environ.get(
        "HUGGING_FACE_TOKEN",
    )
    login(token=token)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=repo_file_path,
        repo_id=repo,
        repo_type="dataset",
    )


def generate_policy_combinations():
    # SALT configurations with explicit joint/non-joint amounts and phase-out parameters
    salt_configs = [
        {
            "name": "0_cap",
            "cap_joint": 0,
            "cap_other": 0,
            "phase_out": False,
            "phase_out_rate": 0,
            "phase_out_threshold_joint": 0,
            "phase_out_threshold_other": 0,
        },
        {
            "name": "tcja_base",
            "cap_joint": 10_000,
            "cap_other": 10_000,
            "phase_out": False,
            "phase_out_rate": 0,
            "phase_out_threshold_joint": 0,
            "phase_out_threshold_other": 0,
        },
        {
            "name": "tcja_base_with_married_bonus",
            "cap_joint": 20_000,
            "cap_other": 10_000,
            "phase_out": False,
            "phase_out_rate": 0,
            "phase_out_threshold_joint": 0,
            "phase_out_threshold_other": 0,
        },
        {
            "name": "tcja_base_with_phaseout",
            "cap_joint": 10_000,
            "cap_other": 10_000,
            "phase_out": True,
            "phase_out_rate": 0.1,  # 10%
            "phase_out_threshold_joint": 400_000,
            "phase_out_threshold_other": 200_000,
        },
        {
            "name": "15_30_k_with_phaseout",
            "cap_joint": 30_000,
            "cap_other": 15_000,
            "phase_out": True,
            "phase_out_rate": 0.1,  # 10%
            "phase_out_threshold_joint": 400_000,
            "phase_out_threshold_other": 200_000,
        },
        {
            "name": "15_30_k_without_phaseout",
            "cap_joint": 30_000,
            "cap_other": 15_000,
            "phase_out": False,
            "phase_out_rate": 0,
            "phase_out_threshold_joint": 0,
            "phase_out_threshold_other": 0,
        },
        {
            "name": "15_k_without_phaseout",
            "cap_joint": 15_000,
            "cap_other": 15_000,
            "phase_out": False,
            "phase_out_rate": 0,
            "phase_out_threshold_joint": 0,
            "phase_out_threshold_other": 0,
        },
        {
            "name": "15_k_with_phaseout",
            "cap_joint": 15_000,
            "cap_other": 15_000,
            "phase_out": True,
            "phase_out_rate": 0.1,
            "phase_out_threshold_joint": 400_000,
            "phase_out_threshold_other": 200_000,
        },
        {
            "name": "tcja_married_bonus_and_phase_out",
            "cap_joint": 20_000,
            "cap_other": 10_000,
            "phase_out": True,
            "phase_out_rate": 0.1,  # 10%
            "phase_out_threshold_joint": 400_000,
            "phase_out_threshold_other": 200_000,
        },
        {
            "name": "uncapped",
            "cap_joint": float("inf"),
            "cap_other": float("inf"),
            "phase_out": False,
            "phase_out_rate": 0,
            "phase_out_threshold_joint": 0,
            "phase_out_threshold_other": 0,
        },
        {
            "name": "uncapped_with_phaseout",
            "cap_joint": float("inf"),
            "cap_other": float("inf"),
            "phase_out": True,
            "phase_out_rate": 0.1,
            "phase_out_threshold_joint": 400_000,
            "phase_out_threshold_other": 200_000,
        },
    ]

    # AMT configurations with explicit joint/non-joint amounts for 2026
    amt_configs = [
        {
            "name": "repealed",
            "exemption_joint": float("inf"),
            "exemption_other": float("inf"),
            "phase_out_joint": 0,
            "phase_out_other": 0,
        },
        {
            "name": "pre_tcja_ex_pre_tcja_po",
            "exemption_joint": 139_850,
            "exemption_other": 89_925,
            "phase_out_joint": 1_278_575,
            "phase_out_other": 639_300,
        },
        {
            "name": "pre_tcja_ex_tcja_po",
            "exemption_joint": 139_850,
            "exemption_other": 89_925,
            "phase_out_joint": 209_000,
            "phase_out_other": 156_700,
        },
        {
            "name": "tcja_ex_pre_tcja_po",
            "exemption_joint": 109_700,
            "exemption_other": 70_500,
            "phase_out_joint": 1_278_575,
            "phase_out_other": 639_300,
        },
        {
            "name": "tcja_both",
            "exemption_joint": 109_700,
            "exemption_other": 70_500,
            "phase_out_joint": 209_000,
            "phase_out_other": 156_700,
        },
    ]

    behavioral = [False, True]

    tcja_other_extended = [False, True]

    return list(product(salt_configs, amt_configs, behavioral, tcja_other_extended))

def get_behavioral_reform():
    """Returns the behavioral response reform parameters"""
    return {
        "gov.simulation.labor_supply_responses.elasticities.income": {
            "2024-01-01.2100-12-31": -0.05
        },
        "gov.simulation.labor_supply_responses.elasticities.substitution.by_position_and_decile.primary.1": {
            "2024-01-01.2100-12-31": 0.31
        },
        "gov.simulation.labor_supply_responses.elasticities.substitution.by_position_and_decile.primary.10": {
            "2024-01-01.2100-12-31": 0.22
        },
        "gov.simulation.labor_supply_responses.elasticities.substitution.by_position_and_decile.primary.2": {
            "2024-01-01.2100-12-31": 0.28
        },
        "gov.simulation.labor_supply_responses.elasticities.substitution.by_position_and_decile.primary.3": {
            "2024-01-01.2100-12-31": 0.27
        },
        "gov.simulation.labor_supply_responses.elasticities.substitution.by_position_and_decile.primary.4": {
            "2024-01-01.2100-12-31": 0.27
        },
        "gov.simulation.labor_supply_responses.elasticities.substitution.by_position_and_decile.primary.5": {
            "2024-01-01.2100-12-31": 0.25
        },
        "gov.simulation.labor_supply_responses.elasticities.substitution.by_position_and_decile.primary.6": {
            "2024-01-01.2100-12-31": 0.25
        },
        "gov.simulation.labor_supply_responses.elasticities.substitution.by_position_and_decile.primary.7": {
            "2024-01-01.2100-12-31": 0.22
        },
        "gov.simulation.labor_supply_responses.elasticities.substitution.by_position_and_decile.primary.8": {
            "2024-01-01.2100-12-31": 0.22
        },
        "gov.simulation.labor_supply_responses.elasticities.substitution.by_position_and_decile.primary.9": {
            "2024-01-01.2100-12-31": 0.22
        },
        "gov.simulation.labor_supply_responses.elasticities.substitution.by_position_and_decile.secondary": {
            "2024-01-01.2100-12-31": 0.27
        },
    }

def get_other_tcja_provisions():
    """Returns the reform dictionary for TCJA extension"""
    return {
        "gov.irs.credits.ctc.amount.base[0].amount": {"2026-01-01.2100-12-31": 2000},
        "gov.irs.credits.ctc.phase_out.threshold.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2100-12-31": 200000
        },
        "gov.irs.credits.ctc.phase_out.threshold.JOINT": {
            "2026-01-01.2100-12-31": 400000
        },
        "gov.irs.credits.ctc.phase_out.threshold.SEPARATE": {
            "2026-01-01.2100-12-31": 200000
        },
        "gov.irs.credits.ctc.phase_out.threshold.SINGLE": {
            "2026-01-01.2100-12-31": 200000
        },
        "gov.irs.credits.ctc.phase_out.threshold.SURVIVING_SPOUSE": {
            "2026-01-01.2100-12-31": 400000
        },
        "gov.irs.credits.ctc.refundable.individual_max": {
            "2026-01-01.2026-12-31": 1800,
            "2027-01-01.2027-12-31": 1800,
            "2028-01-01.2028-12-31": 1800,
            "2029-01-01.2029-12-31": 1900,
            "2030-01-01.2030-12-31": 1900,
            "2031-01-01.2031-12-31": 1900,
            "2032-01-01.2032-12-31": 2000,
            "2033-01-01.2033-12-31": 2000,
            "2034-01-01.2034-12-31": 2000,
            "2035-01-01.2035-12-31": 2000,
        },
        "gov.irs.credits.ctc.refundable.phase_in.threshold": {
            "2026-01-01.2100-12-31": 2500
        },
        "gov.irs.deductions.itemized.casualty.active": {"2026-01-01.2100-12-31": False},
        "gov.irs.deductions.itemized.charity.ceiling.all": {
            "2026-01-01.2100-12-31": 0.6
        },
        "gov.irs.deductions.itemized.limitation.agi_rate": {
            "2026-01-01.2026-12-31": 1,
            "2027-01-01.2027-12-31": 1,
            "2028-01-01.2028-12-31": 1,
            "2029-01-01.2029-12-31": 1,
            "2030-01-01.2030-12-31": 1,
            "2031-01-01.2031-12-31": 1,
            "2032-01-01.2032-12-31": 1,
            "2033-01-01.2033-12-31": 1,
            "2034-01-01.2034-12-31": 1,
        },
        "gov.irs.deductions.itemized.limitation.applicable_amount.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2026-12-31": 1000000,
            "2027-01-01.2027-12-31": 1000000,
            "2028-01-01.2028-12-31": 1000000,
            "2029-01-01.2029-12-31": 1000000,
            "2030-01-01.2030-12-31": 1000000,
            "2031-01-01.2031-12-31": 1000000,
            "2032-01-01.2032-12-31": 1000000,
            "2033-01-01.2033-12-31": 1000000,
            "2034-01-01.2034-12-31": 1000000,
        },
        "gov.irs.deductions.itemized.limitation.applicable_amount.JOINT": {
            "2026-01-01.2026-12-31": 1000000,
            "2027-01-01.2027-12-31": 1000000,
            "2028-01-01.2028-12-31": 1000000,
            "2029-01-01.2029-12-31": 1000000,
            "2030-01-01.2030-12-31": 1000000,
            "2031-01-01.2031-12-31": 1000000,
            "2032-01-01.2032-12-31": 1000000,
            "2033-01-01.2033-12-31": 1000000,
            "2034-01-01.2034-12-31": 1000000,
        },
        "gov.irs.deductions.itemized.limitation.applicable_amount.SEPARATE": {
            "2026-01-01.2026-12-31": 1000000,
            "2027-01-01.2027-12-31": 1000000,
            "2028-01-01.2028-12-31": 1000000,
            "2029-01-01.2029-12-31": 1000000,
            "2030-01-01.2030-12-31": 1000000,
            "2031-01-01.2031-12-31": 1000000,
            "2032-01-01.2032-12-31": 1000000,
            "2033-01-01.2033-12-31": 1000000,
            "2034-01-01.2034-12-31": 1000000,
        },
        "gov.irs.deductions.itemized.limitation.applicable_amount.SINGLE": {
            "2026-01-01.2026-12-31": 1000000,
            "2027-01-01.2027-12-31": 1000000,
            "2028-01-01.2028-12-31": 1000000,
            "2029-01-01.2029-12-31": 1000000,
            "2030-01-01.2030-12-31": 1000000,
            "2031-01-01.2031-12-31": 1000000,
            "2032-01-01.2032-12-31": 1000000,
            "2033-01-01.2033-12-31": 1000000,
            "2034-01-01.2034-12-31": 1000000,
        },
        "gov.irs.deductions.itemized.limitation.applicable_amount.SURVIVING_SPOUSE": {
            "2026-01-01.2026-12-31": 1000000,
            "2027-01-01.2027-12-31": 1000000,
            "2028-01-01.2028-12-31": 1000000,
            "2029-01-01.2029-12-31": 1000000,
            "2030-01-01.2030-12-31": 1000000,
            "2031-01-01.2031-12-31": 1000000,
            "2032-01-01.2032-12-31": 1000000,
            "2033-01-01.2033-12-31": 1000000,
            "2034-01-01.2034-12-31": 1000000,
        },
        "gov.irs.deductions.itemized.limitation.itemized_deduction_rate": {
            "2026-01-01.2026-12-31": 1,
            "2027-01-01.2027-12-31": 1,
            "2028-01-01.2028-12-31": 1,
            "2029-01-01.2029-12-31": 1,
            "2030-01-01.2030-12-31": 1,
            "2031-01-01.2031-12-31": 1,
            "2032-01-01.2032-12-31": 1,
            "2033-01-01.2033-12-31": 1,
            "2034-01-01.2034-12-31": 1,
        },
        "gov.irs.deductions.itemized.salt_and_real_estate.cap.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2100-12-31": 10000
        },
        "gov.irs.deductions.itemized.salt_and_real_estate.cap.JOINT": {
            "2026-01-01.2100-12-31": 10000
        },
        "gov.irs.deductions.itemized.salt_and_real_estate.cap.SEPARATE": {
            "2026-01-01.2100-12-31": 5000
        },
        "gov.irs.deductions.itemized.salt_and_real_estate.cap.SINGLE": {
            "2026-01-01.2100-12-31": 10000
        },
        "gov.irs.deductions.itemized.salt_and_real_estate.cap.SURVIVING_SPOUSE": {
            "2026-01-01.2100-12-31": 10000
        },
        "gov.irs.deductions.qbi.max.business_property.rate": {
            "2026-01-01.2100-12-31": 0.025
        },
        "gov.irs.deductions.qbi.max.rate": {"2026-01-01.2100-12-31": 0.2},
        "gov.irs.deductions.qbi.max.w2_wages.alt_rate": {"2026-01-01.2100-12-31": 0.25},
        "gov.irs.deductions.qbi.max.w2_wages.rate": {"2026-01-01.2100-12-31": 0.5},
        "gov.irs.deductions.qbi.phase_out.length.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2100-12-31": 50000
        },
        "gov.irs.deductions.qbi.phase_out.length.JOINT": {
            "2026-01-01.2100-12-31": 100000
        },
        "gov.irs.deductions.qbi.phase_out.length.SEPARATE": {
            "2026-01-01.2100-12-31": 50000
        },
        "gov.irs.deductions.qbi.phase_out.length.SINGLE": {
            "2026-01-01.2100-12-31": 50000
        },
        "gov.irs.deductions.qbi.phase_out.length.SURVIVING_SPOUSE": {
            "2026-01-01.2100-12-31": 100000
        },
        "gov.irs.deductions.qbi.phase_out.start.HEAD_OF_HOUSEHOLD": {
            "2024-01-01.2024-12-31": 198225,
            "2025-01-01.2025-12-31": 200275,
            "2026-01-01.2026-12-31": 204900,
            "2027-01-01.2027-12-31": 209050,
            "2028-01-01.2028-12-31": 213075,
            "2029-01-01.2029-12-31": 217125,
            "2030-01-01.2030-12-31": 221375,
            "2031-01-01.2031-12-31": 225775,
            "2032-01-01.2032-12-31": 230275,
            "2033-01-01.2033-12-31": 234875,
            "2034-01-01.2034-12-31": 239600,
            "2035-01-01.2035-12-31": 244450,
        },
        "gov.irs.deductions.qbi.phase_out.start.JOINT": {
            "2024-01-01.2024-12-31": 396450,
            "2025-01-01.2025-12-31": 400575,
            "2026-01-01.2026-12-31": 409800,
            "2027-01-01.2027-12-31": 418100,
            "2028-01-01.2028-12-31": 426175,
            "2029-01-01.2029-12-31": 434225,
            "2030-01-01.2030-12-31": 442775,
            "2031-01-01.2031-12-31": 451525,
            "2032-01-01.2032-12-31": 460525,
            "2033-01-01.2033-12-31": 469750,
            "2034-01-01.2034-12-31": 479200,
            "2035-01-01.2035-12-31": 488900,
        },
        "gov.irs.deductions.qbi.phase_out.start.SEPARATE": {
            "2024-01-01.2024-12-31": 198225,
            "2025-01-01.2025-12-31": 200275,
            "2026-01-01.2026-12-31": 204900,
            "2027-01-01.2027-12-31": 209050,
            "2028-01-01.2028-12-31": 213075,
            "2029-01-01.2029-12-31": 217125,
            "2030-01-01.2030-12-31": 221375,
            "2031-01-01.2031-12-31": 225775,
            "2032-01-01.2032-12-31": 230275,
            "2033-01-01.2033-12-31": 234875,
            "2034-01-01.2034-12-31": 239600,
            "2035-01-01.2035-12-31": 244450,
        },
        "gov.irs.deductions.qbi.phase_out.start.SINGLE": {
            "2024-01-01.2024-12-31": 198225,
            "2025-01-01.2025-12-31": 200275,
            "2026-01-01.2026-12-31": 204900,
            "2027-01-01.2027-12-31": 209050,
            "2028-01-01.2028-12-31": 213075,
            "2029-01-01.2029-12-31": 217125,
            "2030-01-01.2030-12-31": 221375,
            "2031-01-01.2031-12-31": 225775,
            "2032-01-01.2032-12-31": 230275,
            "2033-01-01.2033-12-31": 234875,
            "2034-01-01.2034-12-31": 239600,
            "2035-01-01.2035-12-31": 244450,
        },
        "gov.irs.deductions.qbi.phase_out.start.SURVIVING_SPOUSE": {
            "2024-01-01.2024-12-31": 396450,
            "2025-01-01.2025-12-31": 400575,
            "2026-01-01.2026-12-31": 409800,
            "2027-01-01.2027-12-31": 418100,
            "2028-01-01.2028-12-31": 426175,
            "2029-01-01.2029-12-31": 434225,
            "2030-01-01.2030-12-31": 442775,
            "2031-01-01.2031-12-31": 451525,
            "2032-01-01.2032-12-31": 460525,
            "2033-01-01.2033-12-31": 469750,
            "2034-01-01.2034-12-31": 479200,
            "2035-01-01.2035-12-31": 488900,
        },
        "gov.irs.deductions.standard.amount.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2026-12-31": 22950,
            "2027-01-01.2027-12-31": 23425,
            "2028-01-01.2028-12-31": 23875,
            "2029-01-01.2029-12-31": 24325,
            "2030-01-01.2030-12-31": 24800,
            "2031-01-01.2031-12-31": 25300,
            "2032-01-01.2032-12-31": 25800,
            "2033-01-01.2033-12-31": 26300,
            "2034-01-01.2034-12-31": 26825,
            "2035-01-01.2035-12-31": 27375,
        },
        "gov.irs.deductions.standard.amount.JOINT": {
            "2026-01-01.2026-12-31": 30600,
            "2027-01-01.2027-12-31": 31225,
            "2028-01-01.2028-12-31": 31825,
            "2029-01-01.2029-12-31": 32425,
            "2030-01-01.2030-12-31": 33050,
            "2031-01-01.2031-12-31": 33725,
            "2032-01-01.2032-12-31": 34400,
            "2033-01-01.2033-12-31": 35075,
            "2034-01-01.2034-12-31": 35775,
            "2035-01-01.2035-12-31": 36500,
        },
        "gov.irs.deductions.standard.amount.SEPARATE": {
            "2026-01-01.2026-12-31": 15300,
            "2027-01-01.2027-12-31": 15600,
            "2028-01-01.2028-12-31": 15900,
            "2029-01-01.2029-12-31": 16225,
            "2030-01-01.2030-12-31": 16525,
            "2031-01-01.2031-12-31": 16850,
            "2032-01-01.2032-12-31": 17200,
            "2033-01-01.2033-12-31": 17550,
            "2034-01-01.2034-12-31": 17900,
            "2035-01-01.2035-12-31": 18250,
        },
        "gov.irs.deductions.standard.amount.SINGLE": {
            "2026-01-01.2026-12-31": 15300,
            "2027-01-01.2027-12-31": 15600,
            "2028-01-01.2028-12-31": 15900,
            "2029-01-01.2029-12-31": 16225,
            "2030-01-01.2030-12-31": 16525,
            "2031-01-01.2031-12-31": 16850,
            "2032-01-01.2032-12-31": 17200,
            "2033-01-01.2033-12-31": 17550,
            "2034-01-01.2034-12-31": 17900,
            "2035-01-01.2035-12-31": 18250,
        },
        "gov.irs.deductions.standard.amount.SURVIVING_SPOUSE": {
            "2026-01-01.2026-12-31": 30600,
            "2027-01-01.2027-12-31": 31225,
            "2028-01-01.2028-12-31": 31825,
            "2029-01-01.2029-12-31": 32425,
            "2030-01-01.2030-12-31": 33050,
            "2031-01-01.2031-12-31": 33725,
            "2032-01-01.2032-12-31": 34400,
            "2033-01-01.2033-12-31": 35075,
            "2034-01-01.2034-12-31": 35775,
            "2035-01-01.2035-12-31": 36500,
        },
        "gov.irs.income.bracket.rates.2": {"2026-01-01.2100-12-31": 0.12},
        "gov.irs.income.bracket.rates.3": {"2026-01-01.2100-12-31": 0.22},
        "gov.irs.income.bracket.rates.4": {"2026-01-01.2100-12-31": 0.24},
        "gov.irs.income.bracket.rates.5": {"2026-01-01.2100-12-31": 0.32},
        "gov.irs.income.bracket.rates.7": {"2026-01-01.2100-12-31": 0.37},
        "gov.irs.income.bracket.thresholds.3.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2026-12-31": 105475,
            "2027-01-01.2027-12-31": 107600,
            "2028-01-01.2028-12-31": 109700,
            "2029-01-01.2029-12-31": 111775,
            "2030-01-01.2030-12-31": 113950,
            "2031-01-01.2031-12-31": 116225,
            "2032-01-01.2032-12-31": 118525,
            "2033-01-01.2033-12-31": 120900,
            "2034-01-01.2034-12-31": 123350,
            "2035-01-01.2035-12-31": 125825,
        },
        "gov.irs.income.bracket.thresholds.3.JOINT": {
            "2026-01-01.2026-12-31": 210950,
            "2027-01-01.2027-12-31": 215225,
            "2028-01-01.2028-12-31": 219375,
            "2029-01-01.2029-12-31": 223525,
            "2030-01-01.2030-12-31": 227925,
            "2031-01-01.2031-12-31": 232425,
            "2032-01-01.2032-12-31": 237075,
            "2033-01-01.2033-12-31": 241825,
            "2034-01-01.2034-12-31": 246675,
            "2035-01-01.2035-12-31": 251675,
        },
        "gov.irs.income.bracket.thresholds.3.SEPARATE": {
            "2026-01-01.2026-12-31": 105475,
            "2027-01-01.2027-12-31": 107600,
            "2028-01-01.2028-12-31": 109700,
            "2029-01-01.2029-12-31": 111775,
            "2030-01-01.2030-12-31": 113950,
            "2031-01-01.2031-12-31": 116225,
            "2032-01-01.2032-12-31": 118525,
            "2033-01-01.2033-12-31": 120900,
            "2034-01-01.2034-12-31": 123350,
            "2035-01-01.2035-12-31": 125825,
        },
        "gov.irs.income.bracket.thresholds.3.SINGLE": {
            "2026-01-01.2026-12-31": 105475,
            "2027-01-01.2027-12-31": 107600,
            "2028-01-01.2028-12-31": 109700,
            "2029-01-01.2029-12-31": 111775,
            "2030-01-01.2030-12-31": 113950,
            "2031-01-01.2031-12-31": 116225,
            "2032-01-01.2032-12-31": 118525,
            "2033-01-01.2033-12-31": 120900,
            "2034-01-01.2034-12-31": 123350,
            "2035-01-01.2035-12-31": 125825,
        },
        "gov.irs.income.bracket.thresholds.3.SURVIVING_SPOUSE": {
            "2026-01-01.2026-12-31": 210950,
            "2027-01-01.2027-12-31": 215225,
            "2028-01-01.2028-12-31": 219375,
            "2029-01-01.2029-12-31": 223525,
            "2030-01-01.2030-12-31": 227925,
            "2031-01-01.2031-12-31": 232425,
            "2032-01-01.2032-12-31": 237075,
            "2033-01-01.2033-12-31": 241825,
            "2034-01-01.2034-12-31": 246675,
            "2035-01-01.2035-12-31": 251675,
        },
        "gov.irs.income.bracket.thresholds.4.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2026-12-31": 201350,
            "2027-01-01.2027-12-31": 205425,
            "2028-01-01.2028-12-31": 209400,
            "2029-01-01.2029-12-31": 213375,
            "2030-01-01.2030-12-31": 217550,
            "2031-01-01.2031-12-31": 221875,
            "2032-01-01.2032-12-31": 226275,
            "2033-01-01.2033-12-31": 230825,
            "2034-01-01.2034-12-31": 235475,
            "2035-01-01.2035-12-31": 240225,
        },
        "gov.irs.income.bracket.thresholds.4.JOINT": {
            "2026-01-01.2026-12-31": 402725,
            "2027-01-01.2027-12-31": 410875,
            "2028-01-01.2028-12-31": 418800,
            "2029-01-01.2029-12-31": 426725,
            "2030-01-01.2030-12-31": 435125,
            "2031-01-01.2031-12-31": 443725,
            "2032-01-01.2032-12-31": 452575,
            "2033-01-01.2033-12-31": 461650,
            "2034-01-01.2034-12-31": 470925,
            "2035-01-01.2035-12-31": 480450,
        },
        "gov.irs.income.bracket.thresholds.4.SEPARATE": {
            "2026-01-01.2026-12-31": 201350,
            "2027-01-01.2027-12-31": 205425,
            "2028-01-01.2028-12-31": 209400,
            "2029-01-01.2029-12-31": 213375,
            "2030-01-01.2030-12-31": 217550,
            "2031-01-01.2031-12-31": 221875,
            "2032-01-01.2032-12-31": 226275,
            "2033-01-01.2033-12-31": 230825,
            "2034-01-01.2034-12-31": 235475,
            "2035-01-01.2035-12-31": 240225,
        },
        "gov.irs.income.bracket.thresholds.4.SINGLE": {
            "2026-01-01.2026-12-31": 201350,
            "2027-01-01.2027-12-31": 205425,
            "2028-01-01.2028-12-31": 209400,
            "2029-01-01.2029-12-31": 213375,
            "2030-01-01.2030-12-31": 217550,
            "2031-01-01.2031-12-31": 221875,
            "2032-01-01.2032-12-31": 226275,
            "2033-01-01.2033-12-31": 230825,
            "2034-01-01.2034-12-31": 235475,
            "2035-01-01.2035-12-31": 240225,
        },
        "gov.irs.income.bracket.thresholds.4.SURVIVING_SPOUSE": {
            "2026-01-01.2026-12-31": 402725,
            "2027-01-01.2027-12-31": 410875,
            "2028-01-01.2028-12-31": 418800,
            "2029-01-01.2029-12-31": 426725,
            "2030-01-01.2030-12-31": 435125,
            "2031-01-01.2031-12-31": 443725,
            "2032-01-01.2032-12-31": 452575,
            "2033-01-01.2033-12-31": 461650,
            "2034-01-01.2034-12-31": 470925,
            "2035-01-01.2035-12-31": 480450,
        },
        "gov.irs.income.bracket.thresholds.5.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2026-12-31": 255700,
            "2027-01-01.2027-12-31": 260875,
            "2028-01-01.2028-12-31": 265925,
            "2029-01-01.2029-12-31": 270950,
            "2030-01-01.2030-12-31": 276275,
            "2031-01-01.2031-12-31": 281750,
            "2032-01-01.2032-12-31": 287375,
            "2033-01-01.2033-12-31": 293125,
            "2034-01-01.2034-12-31": 299025,
            "2035-01-01.2035-12-31": 305075,
        },
        "gov.irs.income.bracket.thresholds.5.JOINT": {
            "2026-01-01.2026-12-31": 511400,
            "2027-01-01.2027-12-31": 521775,
            "2028-01-01.2028-12-31": 531850,
            "2029-01-01.2029-12-31": 541925,
            "2030-01-01.2030-12-31": 552575,
            "2031-01-01.2031-12-31": 563500,
            "2032-01-01.2032-12-31": 574725,
            "2033-01-01.2033-12-31": 586250,
            "2034-01-01.2034-12-31": 598050,
            "2035-01-01.2035-12-31": 610125,
        },
        "gov.irs.income.bracket.thresholds.5.SEPARATE": {
            "2026-01-01.2026-12-31": 255700,
            "2027-01-01.2027-12-31": 260875,
            "2028-01-01.2028-12-31": 265925,
            "2029-01-01.2029-12-31": 270950,
            "2030-01-01.2030-12-31": 276275,
            "2031-01-01.2031-12-31": 281750,
            "2032-01-01.2032-12-31": 287375,
            "2033-01-01.2033-12-31": 293125,
            "2034-01-01.2034-12-31": 299025,
            "2035-01-01.2035-12-31": 305075,
        },
        "gov.irs.income.bracket.thresholds.5.SINGLE": {
            "2026-01-01.2026-12-31": 255700,
            "2027-01-01.2027-12-31": 260875,
            "2028-01-01.2028-12-31": 265925,
            "2029-01-01.2029-12-31": 270950,
            "2030-01-01.2030-12-31": 276275,
            "2031-01-01.2031-12-31": 281750,
            "2032-01-01.2032-12-31": 287375,
            "2033-01-01.2033-12-31": 293125,
            "2034-01-01.2034-12-31": 299025,
            "2035-01-01.2035-12-31": 305075,
        },
        "gov.irs.income.bracket.thresholds.5.SURVIVING_SPOUSE": {
            "2026-01-01.2026-12-31": 511400,
            "2027-01-01.2027-12-31": 521775,
            "2028-01-01.2028-12-31": 531850,
            "2029-01-01.2029-12-31": 541925,
            "2030-01-01.2030-12-31": 552575,
            "2031-01-01.2031-12-31": 563500,
            "2032-01-01.2032-12-31": 574725,
            "2033-01-01.2033-12-31": 586250,
            "2034-01-01.2034-12-31": 598050,
            "2035-01-01.2035-12-31": 610125,
        },
        "gov.irs.income.bracket.thresholds.6.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2026-12-31": 639300,
            "2027-01-01.2027-12-31": 652250,
            "2028-01-01.2028-12-31": 664825,
            "2029-01-01.2029-12-31": 677425,
            "2030-01-01.2030-12-31": 690725,
            "2031-01-01.2031-12-31": 704400,
            "2032-01-01.2032-12-31": 718425,
            "2033-01-01.2033-12-31": 732825,
            "2034-01-01.2034-12-31": 747575,
            "2035-01-01.2035-12-31": 762675,
        },
        "gov.irs.income.bracket.thresholds.6.JOINT": {
            "2026-01-01.2026-12-31": 767125,
            "2027-01-01.2027-12-31": 782650,
            "2028-01-01.2028-12-31": 797775,
            "2029-01-01.2029-12-31": 812875,
            "2030-01-01.2030-12-31": 828850,
            "2031-01-01.2031-12-31": 845250,
            "2032-01-01.2032-12-31": 862100,
            "2033-01-01.2033-12-31": 879350,
            "2034-01-01.2034-12-31": 897050,
            "2035-01-01.2035-12-31": 915200,
        },
        "gov.irs.income.bracket.thresholds.6.SEPARATE": {
            "2026-01-01.2026-12-31": 383550,
            "2027-01-01.2027-12-31": 391325,
            "2028-01-01.2028-12-31": 398875,
            "2029-01-01.2029-12-31": 406450,
            "2030-01-01.2030-12-31": 414425,
            "2031-01-01.2031-12-31": 422625,
            "2032-01-01.2032-12-31": 431050,
            "2033-01-01.2033-12-31": 439675,
            "2034-01-01.2034-12-31": 448525,
            "2035-01-01.2035-12-31": 457600,
        },
        "gov.irs.income.bracket.thresholds.6.SINGLE": {
            "2026-01-01.2026-12-31": 639300,
            "2027-01-01.2027-12-31": 652250,
            "2028-01-01.2028-12-31": 664825,
            "2029-01-01.2029-12-31": 677425,
            "2030-01-01.2030-12-31": 690725,
            "2031-01-01.2031-12-31": 704400,
            "2032-01-01.2032-12-31": 718425,
            "2033-01-01.2033-12-31": 732825,
            "2034-01-01.2034-12-31": 747575,
            "2035-01-01.2035-12-31": 762675,
        },
        "gov.irs.income.bracket.thresholds.6.SURVIVING_SPOUSE": {
            "2026-01-01.2026-12-31": 767125,
            "2027-01-01.2027-12-31": 782650,
            "2028-01-01.2028-12-31": 797775,
            "2029-01-01.2029-12-31": 812875,
            "2030-01-01.2030-12-31": 828850,
            "2031-01-01.2031-12-31": 845250,
            "2032-01-01.2032-12-31": 862100,
            "2033-01-01.2033-12-31": 879350,
            "2034-01-01.2034-12-31": 897050,
            "2035-01-01.2035-12-31": 915200,
        },
        "gov.irs.income.exemption.amount": {"2026-01-01.2100-12-31": 0},
    }

def create_reform_dict(salt_config, amt_config, behavioral, tcja_other_extended):
    """Create reform dictionary with optional behavioral responses"""
    reform_dict = {}

    # SALT caps
    reform_dict.update(
        {
            "gov.irs.deductions.itemized.salt_and_real_estate.cap.JOINT": {
                "2026-01-01.2100-12-31": salt_config["cap_joint"]
            },
            "gov.irs.deductions.itemized.salt_and_real_estate.cap.SURVIVING_SPOUSE": {
                "2026-01-01.2100-12-31": salt_config["cap_joint"]
            },
            "gov.irs.deductions.itemized.salt_and_real_estate.cap.SEPARATE": {
                "2026-01-01.2100-12-31": salt_config["cap_joint"] / 2
            },
            "gov.irs.deductions.itemized.salt_and_real_estate.cap.SINGLE": {
                "2026-01-01.2100-12-31": salt_config["cap_other"]
            },
            "gov.irs.deductions.itemized.salt_and_real_estate.cap.HEAD_OF_HOUSEHOLD": {
                "2026-01-01.2100-12-31": salt_config["cap_other"]
            },
            # Phase-out parameters
            "gov.contrib.salt_phase_out.in_effect": {
                "2024-01-01.2100-12-31": salt_config["phase_out"]
            },
            # Add joint rate parameter
            "gov.contrib.salt_phase_out.rate.joint[1].rate": {
                "2024-01-01.2100-12-31": salt_config["phase_out_rate"]
            },
            "gov.contrib.salt_phase_out.rate.other[1].rate": {
                "2024-01-01.2100-12-31": salt_config["phase_out_rate"]
            },
            "gov.contrib.salt_phase_out.rate.joint[1].threshold": {
                "2024-01-01.2100-12-31": salt_config["phase_out_threshold_joint"]
            },
            "gov.contrib.salt_phase_out.rate.other[1].threshold": {
                "2024-01-01.2100-12-31": salt_config["phase_out_threshold_other"]
            },
        }
    )

    # AMT parameters - only if not repealed
    reform_dict.update(
        {
            "gov.irs.income.amt.exemption.amount.JOINT": {
                "2026-01-01.2026-12-31": amt_config["exemption_joint"]
            },
            "gov.irs.income.amt.exemption.amount.SURVIVING_SPOUSE": {
                "2026-01-01.2026-12-31": amt_config["exemption_joint"]
            },
            "gov.irs.income.amt.exemption.amount.SEPARATE": {
                "2026-01-01.2026-12-31": amt_config["exemption_joint"] / 2
            },
            "gov.irs.income.amt.exemption.amount.SINGLE": {
                "2026-01-01.2026-12-31": amt_config["exemption_other"]
            },
            "gov.irs.income.amt.exemption.amount.HEAD_OF_HOUSEHOLD": {
                "2026-01-01.2026-12-31": amt_config["exemption_other"]
            },
            # Phase-out thresholds
            "gov.irs.income.amt.exemption.phase_out.start.JOINT": {
                "2026-01-01.2026-12-31": amt_config["phase_out_joint"]
            },
            "gov.irs.income.amt.exemption.phase_out.start.SURVIVING_SPOUSE": {
                "2026-01-01.2026-12-31": amt_config["phase_out_joint"]
            },
            "gov.irs.income.amt.exemption.phase_out.start.SEPARATE": {
                "2026-01-01.2026-12-31": amt_config["phase_out_joint"] / 2
            },
            "gov.irs.income.amt.exemption.phase_out.start.SINGLE": {
                "2026-01-01.2026-12-31": amt_config["phase_out_other"]
            },
            "gov.irs.income.amt.exemption.phase_out.start.HEAD_OF_HOUSEHOLD": {
                "2026-01-01.2026-12-31": amt_config["phase_out_other"]
            },
        }
    )

    # Add behavioral responses if enabled
    if behavioral:
        reform_dict.update(get_behavioral_reform())

    # Add TCJA extension reform
    if tcja_other_extended:
        reform_dict.update(get_other_tcja_provisions())

    return reform_dict

def get_tcja_extension_reform():
    """Returns the reform dictionary for TCJA extension"""
    return {
        "gov.irs.credits.ctc.amount.base[0].amount": {"2026-01-01.2100-12-31": 2000},
        "gov.irs.credits.ctc.phase_out.threshold.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2100-12-31": 200000
        },
        "gov.irs.credits.ctc.phase_out.threshold.JOINT": {
            "2026-01-01.2100-12-31": 400000
        },
        "gov.irs.credits.ctc.phase_out.threshold.SEPARATE": {
            "2026-01-01.2100-12-31": 200000
        },
        "gov.irs.credits.ctc.phase_out.threshold.SINGLE": {
            "2026-01-01.2100-12-31": 200000
        },
        "gov.irs.credits.ctc.phase_out.threshold.SURVIVING_SPOUSE": {
            "2026-01-01.2100-12-31": 400000
        },
        "gov.irs.credits.ctc.refundable.individual_max": {
            "2026-01-01.2026-12-31": 1800,
            "2027-01-01.2027-12-31": 1800,
            "2028-01-01.2028-12-31": 1800,
            "2029-01-01.2029-12-31": 1900,
            "2030-01-01.2030-12-31": 1900,
            "2031-01-01.2031-12-31": 1900,
            "2032-01-01.2032-12-31": 2000,
            "2033-01-01.2033-12-31": 2000,
            "2034-01-01.2034-12-31": 2000,
            "2035-01-01.2035-12-31": 2000,
        },
        "gov.irs.credits.ctc.refundable.phase_in.threshold": {
            "2026-01-01.2100-12-31": 2500
        },
        "gov.irs.deductions.itemized.casualty.active": {"2026-01-01.2100-12-31": False},
        "gov.irs.deductions.itemized.charity.ceiling.all": {
            "2026-01-01.2100-12-31": 0.6
        },
        "gov.irs.deductions.itemized.limitation.agi_rate": {
            "2026-01-01.2026-12-31": 1,
            "2027-01-01.2027-12-31": 1,
            "2028-01-01.2028-12-31": 1,
            "2029-01-01.2029-12-31": 1,
            "2030-01-01.2030-12-31": 1,
            "2031-01-01.2031-12-31": 1,
            "2032-01-01.2032-12-31": 1,
            "2033-01-01.2033-12-31": 1,
            "2034-01-01.2034-12-31": 1,
        },
        "gov.irs.deductions.itemized.limitation.applicable_amount.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2026-12-31": 1000000,
            "2027-01-01.2027-12-31": 1000000,
            "2028-01-01.2028-12-31": 1000000,
            "2029-01-01.2029-12-31": 1000000,
            "2030-01-01.2030-12-31": 1000000,
            "2031-01-01.2031-12-31": 1000000,
            "2032-01-01.2032-12-31": 1000000,
            "2033-01-01.2033-12-31": 1000000,
            "2034-01-01.2034-12-31": 1000000,
        },
        "gov.irs.deductions.itemized.limitation.applicable_amount.JOINT": {
            "2026-01-01.2026-12-31": 1000000,
            "2027-01-01.2027-12-31": 1000000,
            "2028-01-01.2028-12-31": 1000000,
            "2029-01-01.2029-12-31": 1000000,
            "2030-01-01.2030-12-31": 1000000,
            "2031-01-01.2031-12-31": 1000000,
            "2032-01-01.2032-12-31": 1000000,
            "2033-01-01.2033-12-31": 1000000,
            "2034-01-01.2034-12-31": 1000000,
        },
        "gov.irs.deductions.itemized.limitation.applicable_amount.SEPARATE": {
            "2026-01-01.2026-12-31": 1000000,
            "2027-01-01.2027-12-31": 1000000,
            "2028-01-01.2028-12-31": 1000000,
            "2029-01-01.2029-12-31": 1000000,
            "2030-01-01.2030-12-31": 1000000,
            "2031-01-01.2031-12-31": 1000000,
            "2032-01-01.2032-12-31": 1000000,
            "2033-01-01.2033-12-31": 1000000,
            "2034-01-01.2034-12-31": 1000000,
        },
        "gov.irs.deductions.itemized.limitation.applicable_amount.SINGLE": {
            "2026-01-01.2026-12-31": 1000000,
            "2027-01-01.2027-12-31": 1000000,
            "2028-01-01.2028-12-31": 1000000,
            "2029-01-01.2029-12-31": 1000000,
            "2030-01-01.2030-12-31": 1000000,
            "2031-01-01.2031-12-31": 1000000,
            "2032-01-01.2032-12-31": 1000000,
            "2033-01-01.2033-12-31": 1000000,
            "2034-01-01.2034-12-31": 1000000,
        },
        "gov.irs.deductions.itemized.limitation.applicable_amount.SURVIVING_SPOUSE": {
            "2026-01-01.2026-12-31": 1000000,
            "2027-01-01.2027-12-31": 1000000,
            "2028-01-01.2028-12-31": 1000000,
            "2029-01-01.2029-12-31": 1000000,
            "2030-01-01.2030-12-31": 1000000,
            "2031-01-01.2031-12-31": 1000000,
            "2032-01-01.2032-12-31": 1000000,
            "2033-01-01.2033-12-31": 1000000,
            "2034-01-01.2034-12-31": 1000000,
        },
        "gov.irs.deductions.itemized.limitation.itemized_deduction_rate": {
            "2026-01-01.2026-12-31": 1,
            "2027-01-01.2027-12-31": 1,
            "2028-01-01.2028-12-31": 1,
            "2029-01-01.2029-12-31": 1,
            "2030-01-01.2030-12-31": 1,
            "2031-01-01.2031-12-31": 1,
            "2032-01-01.2032-12-31": 1,
            "2033-01-01.2033-12-31": 1,
            "2034-01-01.2034-12-31": 1,
        },
        "gov.irs.deductions.itemized.salt_and_real_estate.cap.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2100-12-31": 10000
        },
        "gov.irs.deductions.itemized.salt_and_real_estate.cap.JOINT": {
            "2026-01-01.2100-12-31": 10000
        },
        "gov.irs.deductions.itemized.salt_and_real_estate.cap.SEPARATE": {
            "2026-01-01.2100-12-31": 5000
        },
        "gov.irs.deductions.itemized.salt_and_real_estate.cap.SINGLE": {
            "2026-01-01.2100-12-31": 10000
        },
        "gov.irs.deductions.itemized.salt_and_real_estate.cap.SURVIVING_SPOUSE": {
            "2026-01-01.2100-12-31": 10000
        },
        "gov.irs.deductions.qbi.max.business_property.rate": {
            "2026-01-01.2100-12-31": 0.025
        },
        "gov.irs.deductions.qbi.max.rate": {"2026-01-01.2100-12-31": 0.2},
        "gov.irs.deductions.qbi.max.w2_wages.alt_rate": {"2026-01-01.2100-12-31": 0.25},
        "gov.irs.deductions.qbi.max.w2_wages.rate": {"2026-01-01.2100-12-31": 0.5},
        "gov.irs.deductions.qbi.phase_out.length.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2100-12-31": 50000
        },
        "gov.irs.deductions.qbi.phase_out.length.JOINT": {
            "2026-01-01.2100-12-31": 100000
        },
        "gov.irs.deductions.qbi.phase_out.length.SEPARATE": {
            "2026-01-01.2100-12-31": 50000
        },
        "gov.irs.deductions.qbi.phase_out.length.SINGLE": {
            "2026-01-01.2100-12-31": 50000
        },
        "gov.irs.deductions.qbi.phase_out.length.SURVIVING_SPOUSE": {
            "2026-01-01.2100-12-31": 100000
        },
        "gov.irs.deductions.qbi.phase_out.start.HEAD_OF_HOUSEHOLD": {
            "2024-01-01.2024-12-31": 198225,
            "2025-01-01.2025-12-31": 200275,
            "2026-01-01.2026-12-31": 204900,
            "2027-01-01.2027-12-31": 209050,
            "2028-01-01.2028-12-31": 213075,
            "2029-01-01.2029-12-31": 217125,
            "2030-01-01.2030-12-31": 221375,
            "2031-01-01.2031-12-31": 225775,
            "2032-01-01.2032-12-31": 230275,
            "2033-01-01.2033-12-31": 234875,
            "2034-01-01.2034-12-31": 239600,
            "2035-01-01.2035-12-31": 244450,
        },
        "gov.irs.deductions.qbi.phase_out.start.JOINT": {
            "2024-01-01.2024-12-31": 396450,
            "2025-01-01.2025-12-31": 400575,
            "2026-01-01.2026-12-31": 409800,
            "2027-01-01.2027-12-31": 418100,
            "2028-01-01.2028-12-31": 426175,
            "2029-01-01.2029-12-31": 434225,
            "2030-01-01.2030-12-31": 442775,
            "2031-01-01.2031-12-31": 451525,
            "2032-01-01.2032-12-31": 460525,
            "2033-01-01.2033-12-31": 469750,
            "2034-01-01.2034-12-31": 479200,
            "2035-01-01.2035-12-31": 488900,
        },
        "gov.irs.deductions.qbi.phase_out.start.SEPARATE": {
            "2024-01-01.2024-12-31": 198225,
            "2025-01-01.2025-12-31": 200275,
            "2026-01-01.2026-12-31": 204900,
            "2027-01-01.2027-12-31": 209050,
            "2028-01-01.2028-12-31": 213075,
            "2029-01-01.2029-12-31": 217125,
            "2030-01-01.2030-12-31": 221375,
            "2031-01-01.2031-12-31": 225775,
            "2032-01-01.2032-12-31": 230275,
            "2033-01-01.2033-12-31": 234875,
            "2034-01-01.2034-12-31": 239600,
            "2035-01-01.2035-12-31": 244450,
        },
        "gov.irs.deductions.qbi.phase_out.start.SINGLE": {
            "2024-01-01.2024-12-31": 198225,
            "2025-01-01.2025-12-31": 200275,
            "2026-01-01.2026-12-31": 204900,
            "2027-01-01.2027-12-31": 209050,
            "2028-01-01.2028-12-31": 213075,
            "2029-01-01.2029-12-31": 217125,
            "2030-01-01.2030-12-31": 221375,
            "2031-01-01.2031-12-31": 225775,
            "2032-01-01.2032-12-31": 230275,
            "2033-01-01.2033-12-31": 234875,
            "2034-01-01.2034-12-31": 239600,
            "2035-01-01.2035-12-31": 244450,
        },
        "gov.irs.deductions.qbi.phase_out.start.SURVIVING_SPOUSE": {
            "2024-01-01.2024-12-31": 396450,
            "2025-01-01.2025-12-31": 400575,
            "2026-01-01.2026-12-31": 409800,
            "2027-01-01.2027-12-31": 418100,
            "2028-01-01.2028-12-31": 426175,
            "2029-01-01.2029-12-31": 434225,
            "2030-01-01.2030-12-31": 442775,
            "2031-01-01.2031-12-31": 451525,
            "2032-01-01.2032-12-31": 460525,
            "2033-01-01.2033-12-31": 469750,
            "2034-01-01.2034-12-31": 479200,
            "2035-01-01.2035-12-31": 488900,
        },
        "gov.irs.deductions.standard.amount.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2026-12-31": 22950,
            "2027-01-01.2027-12-31": 23425,
            "2028-01-01.2028-12-31": 23875,
            "2029-01-01.2029-12-31": 24325,
            "2030-01-01.2030-12-31": 24800,
            "2031-01-01.2031-12-31": 25300,
            "2032-01-01.2032-12-31": 25800,
            "2033-01-01.2033-12-31": 26300,
            "2034-01-01.2034-12-31": 26825,
            "2035-01-01.2035-12-31": 27375,
        },
        "gov.irs.deductions.standard.amount.JOINT": {
            "2026-01-01.2026-12-31": 30600,
            "2027-01-01.2027-12-31": 31225,
            "2028-01-01.2028-12-31": 31825,
            "2029-01-01.2029-12-31": 32425,
            "2030-01-01.2030-12-31": 33050,
            "2031-01-01.2031-12-31": 33725,
            "2032-01-01.2032-12-31": 34400,
            "2033-01-01.2033-12-31": 35075,
            "2034-01-01.2034-12-31": 35775,
            "2035-01-01.2035-12-31": 36500,
        },
        "gov.irs.deductions.standard.amount.SEPARATE": {
            "2026-01-01.2026-12-31": 15300,
            "2027-01-01.2027-12-31": 15600,
            "2028-01-01.2028-12-31": 15900,
            "2029-01-01.2029-12-31": 16225,
            "2030-01-01.2030-12-31": 16525,
            "2031-01-01.2031-12-31": 16850,
            "2032-01-01.2032-12-31": 17200,
            "2033-01-01.2033-12-31": 17550,
            "2034-01-01.2034-12-31": 17900,
            "2035-01-01.2035-12-31": 18250,
        },
        "gov.irs.deductions.standard.amount.SINGLE": {
            "2026-01-01.2026-12-31": 15300,
            "2027-01-01.2027-12-31": 15600,
            "2028-01-01.2028-12-31": 15900,
            "2029-01-01.2029-12-31": 16225,
            "2030-01-01.2030-12-31": 16525,
            "2031-01-01.2031-12-31": 16850,
            "2032-01-01.2032-12-31": 17200,
            "2033-01-01.2033-12-31": 17550,
            "2034-01-01.2034-12-31": 17900,
            "2035-01-01.2035-12-31": 18250,
        },
        "gov.irs.deductions.standard.amount.SURVIVING_SPOUSE": {
            "2026-01-01.2026-12-31": 30600,
            "2027-01-01.2027-12-31": 31225,
            "2028-01-01.2028-12-31": 31825,
            "2029-01-01.2029-12-31": 32425,
            "2030-01-01.2030-12-31": 33050,
            "2031-01-01.2031-12-31": 33725,
            "2032-01-01.2032-12-31": 34400,
            "2033-01-01.2033-12-31": 35075,
            "2034-01-01.2034-12-31": 35775,
            "2035-01-01.2035-12-31": 36500,
        },
        "gov.irs.income.amt.exemption.amount.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2026-12-31": 89925,
            "2027-01-01.2027-12-31": 91750,
            "2028-01-01.2028-12-31": 93525,
            "2029-01-01.2029-12-31": 95300,
            "2030-01-01.2030-12-31": 97150,
            "2031-01-01.2031-12-31": 99075,
            "2032-01-01.2032-12-31": 101050,
            "2033-01-01.2033-12-31": 103075,
            "2034-01-01.2034-12-31": 105150,
            "2035-01-01.2035-12-31": 107275,
        },
        "gov.irs.income.amt.exemption.amount.JOINT": {
            "2026-01-01.2026-12-31": 139850,
            "2027-01-01.2027-12-31": 142675,
            "2028-01-01.2028-12-31": 145425,
            "2029-01-01.2029-12-31": 148200,
            "2030-01-01.2030-12-31": 151100,
            "2031-01-01.2031-12-31": 154100,
            "2032-01-01.2032-12-31": 157150,
            "2033-01-01.2033-12-31": 160300,
            "2034-01-01.2034-12-31": 163525,
            "2035-01-01.2035-12-31": 166850,
        },
        "gov.irs.income.amt.exemption.amount.SEPARATE": {
            "2026-01-01.2026-12-31": 69925,
            "2027-01-01.2027-12-31": 71350,
            "2028-01-01.2028-12-31": 72725,
            "2029-01-01.2029-12-31": 74100,
            "2030-01-01.2030-12-31": 75550,
            "2031-01-01.2031-12-31": 77050,
            "2032-01-01.2032-12-31": 78575,
            "2033-01-01.2033-12-31": 80150,
            "2034-01-01.2034-12-31": 81775,
            "2035-01-01.2035-12-31": 83425,
        },
        "gov.irs.income.amt.exemption.amount.SINGLE": {
            "2026-01-01.2026-12-31": 89925,
            "2027-01-01.2027-12-31": 91750,
            "2028-01-01.2028-12-31": 93525,
            "2029-01-01.2029-12-31": 95300,
            "2030-01-01.2030-12-31": 97150,
            "2031-01-01.2031-12-31": 99075,
            "2032-01-01.2032-12-31": 101050,
            "2033-01-01.2033-12-31": 103075,
            "2034-01-01.2034-12-31": 105150,
            "2035-01-01.2035-12-31": 107275,
        },
        "gov.irs.income.amt.exemption.amount.SURVIVING_SPOUSE": {
            "2026-01-01.2026-12-31": 139850,
            "2027-01-01.2027-12-31": 142675,
            "2028-01-01.2028-12-31": 145425,
            "2029-01-01.2029-12-31": 148200,
            "2030-01-01.2030-12-31": 151100,
            "2031-01-01.2031-12-31": 154100,
            "2032-01-01.2032-12-31": 157150,
            "2033-01-01.2033-12-31": 160300,
            "2034-01-01.2034-12-31": 163525,
            "2035-01-01.2035-12-31": 166850,
        },
        "gov.irs.income.amt.exemption.phase_out.start.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2026-12-31": 639300,
            "2027-01-01.2027-12-31": 652250,
            "2028-01-01.2028-12-31": 664825,
            "2029-01-01.2029-12-31": 677425,
            "2030-01-01.2030-12-31": 690725,
            "2031-01-01.2031-12-31": 704400,
            "2032-01-01.2032-12-31": 718425,
            "2033-01-01.2033-12-31": 732825,
            "2034-01-01.2034-12-31": 747575,
            "2035-01-01.2035-12-31": 762675,
        },
        "gov.irs.income.amt.exemption.phase_out.start.JOINT": {
            "2026-01-01.2026-12-31": 1278575,
            "2027-01-01.2027-12-31": 1304475,
            "2028-01-01.2028-12-31": 1329675,
            "2029-01-01.2029-12-31": 1354850,
            "2030-01-01.2030-12-31": 1381475,
            "2031-01-01.2031-12-31": 1408825,
            "2032-01-01.2032-12-31": 1436875,
            "2033-01-01.2033-12-31": 1465650,
            "2034-01-01.2034-12-31": 1495150,
            "2035-01-01.2035-12-31": 1525375,
        },
        "gov.irs.income.amt.exemption.phase_out.start.SEPARATE": {
            "2026-01-01.2026-12-31": 639300,
            "2027-01-01.2027-12-31": 652250,
            "2028-01-01.2028-12-31": 664825,
            "2029-01-01.2029-12-31": 677425,
            "2030-01-01.2030-12-31": 690725,
            "2031-01-01.2031-12-31": 704400,
            "2032-01-01.2032-12-31": 718425,
            "2033-01-01.2033-12-31": 732825,
            "2034-01-01.2034-12-31": 747575,
            "2035-01-01.2035-12-31": 762675,
        },
        "gov.irs.income.amt.exemption.phase_out.start.SINGLE": {
            "2026-01-01.2026-12-31": 639300,
            "2027-01-01.2027-12-31": 652250,
            "2028-01-01.2028-12-31": 664825,
            "2029-01-01.2029-12-31": 677425,
            "2030-01-01.2030-12-31": 690725,
            "2031-01-01.2031-12-31": 704400,
            "2032-01-01.2032-12-31": 718425,
            "2033-01-01.2033-12-31": 732825,
            "2034-01-01.2034-12-31": 747575,
            "2035-01-01.2035-12-31": 762675,
        },
        "gov.irs.income.amt.exemption.phase_out.start.SURVIVING_SPOUSE": {
            "2026-01-01.2026-12-31": 1278575,
            "2027-01-01.2027-12-31": 1304475,
            "2028-01-01.2028-12-31": 1329675,
            "2029-01-01.2029-12-31": 1354850,
            "2030-01-01.2030-12-31": 1381475,
            "2031-01-01.2031-12-31": 1408825,
            "2032-01-01.2032-12-31": 1436875,
            "2033-01-01.2033-12-31": 1465650,
            "2034-01-01.2034-12-31": 1495150,
            "2035-01-01.2035-12-31": 1525375,
        },
        "gov.irs.income.bracket.rates.2": {"2026-01-01.2100-12-31": 0.12},
        "gov.irs.income.bracket.rates.3": {"2026-01-01.2100-12-31": 0.22},
        "gov.irs.income.bracket.rates.4": {"2026-01-01.2100-12-31": 0.24},
        "gov.irs.income.bracket.rates.5": {"2026-01-01.2100-12-31": 0.32},
        "gov.irs.income.bracket.rates.7": {"2026-01-01.2100-12-31": 0.37},
        "gov.irs.income.bracket.thresholds.3.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2026-12-31": 105475,
            "2027-01-01.2027-12-31": 107600,
            "2028-01-01.2028-12-31": 109700,
            "2029-01-01.2029-12-31": 111775,
            "2030-01-01.2030-12-31": 113950,
            "2031-01-01.2031-12-31": 116225,
            "2032-01-01.2032-12-31": 118525,
            "2033-01-01.2033-12-31": 120900,
            "2034-01-01.2034-12-31": 123350,
            "2035-01-01.2035-12-31": 125825,
        },
        "gov.irs.income.bracket.thresholds.3.JOINT": {
            "2026-01-01.2026-12-31": 210950,
            "2027-01-01.2027-12-31": 215225,
            "2028-01-01.2028-12-31": 219375,
            "2029-01-01.2029-12-31": 223525,
            "2030-01-01.2030-12-31": 227925,
            "2031-01-01.2031-12-31": 232425,
            "2032-01-01.2032-12-31": 237075,
            "2033-01-01.2033-12-31": 241825,
            "2034-01-01.2034-12-31": 246675,
            "2035-01-01.2035-12-31": 251675,
        },
        "gov.irs.income.bracket.thresholds.3.SEPARATE": {
            "2026-01-01.2026-12-31": 105475,
            "2027-01-01.2027-12-31": 107600,
            "2028-01-01.2028-12-31": 109700,
            "2029-01-01.2029-12-31": 111775,
            "2030-01-01.2030-12-31": 113950,
            "2031-01-01.2031-12-31": 116225,
            "2032-01-01.2032-12-31": 118525,
            "2033-01-01.2033-12-31": 120900,
            "2034-01-01.2034-12-31": 123350,
            "2035-01-01.2035-12-31": 125825,
        },
        "gov.irs.income.bracket.thresholds.3.SINGLE": {
            "2026-01-01.2026-12-31": 105475,
            "2027-01-01.2027-12-31": 107600,
            "2028-01-01.2028-12-31": 109700,
            "2029-01-01.2029-12-31": 111775,
            "2030-01-01.2030-12-31": 113950,
            "2031-01-01.2031-12-31": 116225,
            "2032-01-01.2032-12-31": 118525,
            "2033-01-01.2033-12-31": 120900,
            "2034-01-01.2034-12-31": 123350,
            "2035-01-01.2035-12-31": 125825,
        },
        "gov.irs.income.bracket.thresholds.3.SURVIVING_SPOUSE": {
            "2026-01-01.2026-12-31": 210950,
            "2027-01-01.2027-12-31": 215225,
            "2028-01-01.2028-12-31": 219375,
            "2029-01-01.2029-12-31": 223525,
            "2030-01-01.2030-12-31": 227925,
            "2031-01-01.2031-12-31": 232425,
            "2032-01-01.2032-12-31": 237075,
            "2033-01-01.2033-12-31": 241825,
            "2034-01-01.2034-12-31": 246675,
            "2035-01-01.2035-12-31": 251675,
        },
        "gov.irs.income.bracket.thresholds.4.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2026-12-31": 201350,
            "2027-01-01.2027-12-31": 205425,
            "2028-01-01.2028-12-31": 209400,
            "2029-01-01.2029-12-31": 213375,
            "2030-01-01.2030-12-31": 217550,
            "2031-01-01.2031-12-31": 221875,
            "2032-01-01.2032-12-31": 226275,
            "2033-01-01.2033-12-31": 230825,
            "2034-01-01.2034-12-31": 235475,
            "2035-01-01.2035-12-31": 240225,
        },
        "gov.irs.income.bracket.thresholds.4.JOINT": {
            "2026-01-01.2026-12-31": 402725,
            "2027-01-01.2027-12-31": 410875,
            "2028-01-01.2028-12-31": 418800,
            "2029-01-01.2029-12-31": 426725,
            "2030-01-01.2030-12-31": 435125,
            "2031-01-01.2031-12-31": 443725,
            "2032-01-01.2032-12-31": 452575,
            "2033-01-01.2033-12-31": 461650,
            "2034-01-01.2034-12-31": 470925,
            "2035-01-01.2035-12-31": 480450,
        },
        "gov.irs.income.bracket.thresholds.4.SEPARATE": {
            "2026-01-01.2026-12-31": 201350,
            "2027-01-01.2027-12-31": 205425,
            "2028-01-01.2028-12-31": 209400,
            "2029-01-01.2029-12-31": 213375,
            "2030-01-01.2030-12-31": 217550,
            "2031-01-01.2031-12-31": 221875,
            "2032-01-01.2032-12-31": 226275,
            "2033-01-01.2033-12-31": 230825,
            "2034-01-01.2034-12-31": 235475,
            "2035-01-01.2035-12-31": 240225,
        },
        "gov.irs.income.bracket.thresholds.4.SINGLE": {
            "2026-01-01.2026-12-31": 201350,
            "2027-01-01.2027-12-31": 205425,
            "2028-01-01.2028-12-31": 209400,
            "2029-01-01.2029-12-31": 213375,
            "2030-01-01.2030-12-31": 217550,
            "2031-01-01.2031-12-31": 221875,
            "2032-01-01.2032-12-31": 226275,
            "2033-01-01.2033-12-31": 230825,
            "2034-01-01.2034-12-31": 235475,
            "2035-01-01.2035-12-31": 240225,
        },
        "gov.irs.income.bracket.thresholds.4.SURVIVING_SPOUSE": {
            "2026-01-01.2026-12-31": 402725,
            "2027-01-01.2027-12-31": 410875,
            "2028-01-01.2028-12-31": 418800,
            "2029-01-01.2029-12-31": 426725,
            "2030-01-01.2030-12-31": 435125,
            "2031-01-01.2031-12-31": 443725,
            "2032-01-01.2032-12-31": 452575,
            "2033-01-01.2033-12-31": 461650,
            "2034-01-01.2034-12-31": 470925,
            "2035-01-01.2035-12-31": 480450,
        },
        "gov.irs.income.bracket.thresholds.5.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2026-12-31": 255700,
            "2027-01-01.2027-12-31": 260875,
            "2028-01-01.2028-12-31": 265925,
            "2029-01-01.2029-12-31": 270950,
            "2030-01-01.2030-12-31": 276275,
            "2031-01-01.2031-12-31": 281750,
            "2032-01-01.2032-12-31": 287375,
            "2033-01-01.2033-12-31": 293125,
            "2034-01-01.2034-12-31": 299025,
            "2035-01-01.2035-12-31": 305075,
        },
        "gov.irs.income.bracket.thresholds.5.JOINT": {
            "2026-01-01.2026-12-31": 511400,
            "2027-01-01.2027-12-31": 521775,
            "2028-01-01.2028-12-31": 531850,
            "2029-01-01.2029-12-31": 541925,
            "2030-01-01.2030-12-31": 552575,
            "2031-01-01.2031-12-31": 563500,
            "2032-01-01.2032-12-31": 574725,
            "2033-01-01.2033-12-31": 586250,
            "2034-01-01.2034-12-31": 598050,
            "2035-01-01.2035-12-31": 610125,
        },
        "gov.irs.income.bracket.thresholds.5.SEPARATE": {
            "2026-01-01.2026-12-31": 255700,
            "2027-01-01.2027-12-31": 260875,
            "2028-01-01.2028-12-31": 265925,
            "2029-01-01.2029-12-31": 270950,
            "2030-01-01.2030-12-31": 276275,
            "2031-01-01.2031-12-31": 281750,
            "2032-01-01.2032-12-31": 287375,
            "2033-01-01.2033-12-31": 293125,
            "2034-01-01.2034-12-31": 299025,
            "2035-01-01.2035-12-31": 305075,
        },
        "gov.irs.income.bracket.thresholds.5.SINGLE": {
            "2026-01-01.2026-12-31": 255700,
            "2027-01-01.2027-12-31": 260875,
            "2028-01-01.2028-12-31": 265925,
            "2029-01-01.2029-12-31": 270950,
            "2030-01-01.2030-12-31": 276275,
            "2031-01-01.2031-12-31": 281750,
            "2032-01-01.2032-12-31": 287375,
            "2033-01-01.2033-12-31": 293125,
            "2034-01-01.2034-12-31": 299025,
            "2035-01-01.2035-12-31": 305075,
        },
        "gov.irs.income.bracket.thresholds.5.SURVIVING_SPOUSE": {
            "2026-01-01.2026-12-31": 511400,
            "2027-01-01.2027-12-31": 521775,
            "2028-01-01.2028-12-31": 531850,
            "2029-01-01.2029-12-31": 541925,
            "2030-01-01.2030-12-31": 552575,
            "2031-01-01.2031-12-31": 563500,
            "2032-01-01.2032-12-31": 574725,
            "2033-01-01.2033-12-31": 586250,
            "2034-01-01.2034-12-31": 598050,
            "2035-01-01.2035-12-31": 610125,
        },
        "gov.irs.income.bracket.thresholds.6.HEAD_OF_HOUSEHOLD": {
            "2026-01-01.2026-12-31": 639300,
            "2027-01-01.2027-12-31": 652250,
            "2028-01-01.2028-12-31": 664825,
            "2029-01-01.2029-12-31": 677425,
            "2030-01-01.2030-12-31": 690725,
            "2031-01-01.2031-12-31": 704400,
            "2032-01-01.2032-12-31": 718425,
            "2033-01-01.2033-12-31": 732825,
            "2034-01-01.2034-12-31": 747575,
            "2035-01-01.2035-12-31": 762675,
        },
        "gov.irs.income.bracket.thresholds.6.JOINT": {
            "2026-01-01.2026-12-31": 767125,
            "2027-01-01.2027-12-31": 782650,
            "2028-01-01.2028-12-31": 797775,
            "2029-01-01.2029-12-31": 812875,
            "2030-01-01.2030-12-31": 828850,
            "2031-01-01.2031-12-31": 845250,
            "2032-01-01.2032-12-31": 862100,
            "2033-01-01.2033-12-31": 879350,
            "2034-01-01.2034-12-31": 897050,
            "2035-01-01.2035-12-31": 915200,
        },
        "gov.irs.income.bracket.thresholds.6.SEPARATE": {
            "2026-01-01.2026-12-31": 383550,
            "2027-01-01.2027-12-31": 391325,
            "2028-01-01.2028-12-31": 398875,
            "2029-01-01.2029-12-31": 406450,
            "2030-01-01.2030-12-31": 414425,
            "2031-01-01.2031-12-31": 422625,
            "2032-01-01.2032-12-31": 431050,
            "2033-01-01.2033-12-31": 439675,
            "2034-01-01.2034-12-31": 448525,
            "2035-01-01.2035-12-31": 457600,
        },
        "gov.irs.income.bracket.thresholds.6.SINGLE": {
            "2026-01-01.2026-12-31": 639300,
            "2027-01-01.2027-12-31": 652250,
            "2028-01-01.2028-12-31": 664825,
            "2029-01-01.2029-12-31": 677425,
            "2030-01-01.2030-12-31": 690725,
            "2031-01-01.2031-12-31": 704400,
            "2032-01-01.2032-12-31": 718425,
            "2033-01-01.2033-12-31": 732825,
            "2034-01-01.2034-12-31": 747575,
            "2035-01-01.2035-12-31": 762675,
        },
        "gov.irs.income.bracket.thresholds.6.SURVIVING_SPOUSE": {
            "2026-01-01.2026-12-31": 767125,
            "2027-01-01.2027-12-31": 782650,
            "2028-01-01.2028-12-31": 797775,
            "2029-01-01.2029-12-31": 812875,
            "2030-01-01.2030-12-31": 828850,
            "2031-01-01.2031-12-31": 845250,
            "2032-01-01.2032-12-31": 862100,
            "2033-01-01.2033-12-31": 879350,
            "2034-01-01.2034-12-31": 897050,
            "2035-01-01.2035-12-31": 915200,
        },
        "gov.irs.income.exemption.amount": {"2026-01-01.2100-12-31": 0},
    }

def get_baseline_data(baseline_sim, year):
    """Get baseline income and decile data"""
    baseline_income = baseline_sim.calculate(
        "income_tax", map_to="household", period=year
    )
    decile = baseline_sim.calculate("household_income_decile", period=year)

    # Filter out negative decile values
    valid_indices = decile >= 0

    people = baseline_sim.calc("household_count_people", period=year)

    net_income = baseline_sim.calc("household_net_income", period=year)

    return {
        "income": baseline_income[valid_indices],
        "unfiltered_decile": decile,
        "decile": decile[valid_indices],
        "full_income": baseline_income,
        "people": people,
        "net_income": net_income,
    }

def calculate_metrics(reform_sim, baseline_data, year):
    """Calculate average income changes by decile and people better/worse off"""
    # Get reform income tax (unfiltered)
    reform_tax = reform_sim.calculate("income_tax", map_to="household", period=year)
    baseline_tax = baseline_data["full_income"]

    # Use baseline filtered data for decile calculations
    baseline_tax_filtered = baseline_data["income"]
    decile = baseline_data["decile"]
    reform_tax_filtered = reform_tax[decile.index]

    # Calculate filtered tax change for decile calculations
    tax_change_filtered = reform_tax_filtered - baseline_tax_filtered

    # Calculate unfiltered total tax change
    total_tax_change = reform_tax.sum() - baseline_tax.sum()

    # Get number of people per decile (using baseline)
    counts_by_decile = baseline_tax_filtered.groupby(decile).count()

    # Calculate average change by decile
    avg_change_by_decile = tax_change_filtered.groupby(decile).sum() / counts_by_decile

    # Convert to dictionary
    decile_dict = avg_change_by_decile.to_dict()
    decile_values = {
        f"avg_income_change_p{k}0_{k+1}0": v for k, v in decile_dict.items()
    }

    baseline_net_income = baseline_data["net_income"]
    reform_net_income = reform_sim.calc("household_net_income", period=year)
    # Calculate percent better/worse off using absolute changes
    absolute_change = reform_net_income - baseline_net_income

    capped_baseline_net_income = np.maximum(baseline_net_income, 1)
    capped_reform_net_income = np.maximum(reform_net_income, 1) + absolute_change
    income_change = (
        capped_reform_net_income - capped_baseline_net_income
    ) / capped_baseline_net_income

    people = baseline_data["people"]
    # Ensure all series have the same index
    income_change = pd.Series(income_change, index=people.index)
    decile = baseline_data["unfiltered_decile"]

    decile_results = []

    for i in range(1, 11):
        in_decile = decile == i
        # Better off: gained more than 0.1%
        better_off = (income_change > 1e-3) & in_decile
        # Worse off: lost more than 0.1%
        worse_off = (income_change < -1e-3) & in_decile

        print(f"Decile {i}:")
        print(f"  Number in decile: {sum(in_decile)}")
        print(f"  Number better off: {sum(better_off)}")
        print(f"  Number worse off: {sum(worse_off)}")

        # Use loc to ensure index alignment
        better_off_sum = people.loc[better_off].sum()
        worse_off_sum = people.loc[worse_off].sum()
        in_decile_sum = people.loc[in_decile].sum()

        decile_results.append(
            {
                "better_off": better_off_sum / in_decile_sum,
                "worse_off": worse_off_sum / in_decile_sum,
            }
        )

    pct_better_off = sum(d["better_off"] for d in decile_results) / 10
    pct_worse_off = sum(d["worse_off"] for d in decile_results) / 10

    return {
        "total_income_change": total_tax_change,
        "percent_better_off": pct_better_off * 100,
        "percent_worse_off": pct_worse_off * 100,
        **decile_values,
    }

# Generate and sort combinations
combinations = generate_policy_combinations()
# Sort combinations so that behavioral=False comes first, then behavioral=True
combinations.sort(key=lambda x: (x[2], x[0]["name"], x[1]["name"]))

def load_progress():
    """
    Load existing progress from CSV files, or return empty structures if no progress exists
    """
    try:
        impacts_df = pd.read_csv("../data/impacts.csv")
        impacts = impacts_df.to_dict("records")
        # Extract completed reforms from impacts
        completed_reforms = {
            reform.replace("_vs_current_law", "").replace("_vs_current_policy", "")
            for reform in impacts_df["reform"]
        }
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # If file doesn't exist or is empty, return empty structures
        impacts = []
        completed_reforms = set()

    return impacts, completed_reforms

def calculate_impacts(year: int = 2026):
    """Calculate impacts for all policy combinations"""
    # Set up baseline simulations first
    year = str(year)
    print("Setting up baseline simulations...")
    baseline_sim = Microsimulation(dataset="enhanced_cps_2024")
    baseline_data = get_baseline_data(baseline_sim, year)

    # Set up TCJA baseline
    tcja_reform = Reform.from_dict(get_tcja_extension_reform(), country_id="us")
    tcja_baseline_sim = Microsimulation(reform=tcja_reform, dataset="enhanced_cps_2024")
    tcja_baseline_data = get_baseline_data(tcja_baseline_sim, year)

    # Load existing progress
    impacts, completed_reforms = load_progress()

    combinations = generate_policy_combinations()
    total_combinations = len(combinations)

    print(f"Processing {total_combinations} policy combinations...")
    print(f"Already completed: {len(completed_reforms)} combinations")

    for i, (salt_config, amt_config, behavioral, tcja_other_extended) in enumerate(
        combinations, 1
    ):
        scenario_name = (
            f"salt_{salt_config['name']}_amt_{amt_config['name']}_"
            f"behavioral_responses{'_yes' if behavioral else '_no'}_"
            f"other_tcja_provisions_extended{'_yes' if tcja_other_extended else '_no'}"
        )

        if scenario_name in completed_reforms:
            print(
                f"Skipping completed combination {i}/{total_combinations}: {scenario_name}"
            )
            continue

        print(f"Processing combination {i}/{total_combinations}: {scenario_name}")

        try:
            # Create and run reform
            reform_dict = create_reform_dict(
                salt_config, amt_config, behavioral, tcja_other_extended
            )
            reform = Reform.from_dict(reform_dict, country_id="us")
            reformed = Microsimulation(reform=reform, dataset="enhanced_cps_2024")

            # Calculate metrics vs current law baseline
            current_law_metrics = calculate_metrics(reformed, baseline_data, year)
            current_law_impacts = {
                "reform": f"{scenario_name}_vs_current_law",
                "baseline": "Current Law",
                **current_law_metrics,
            }
            impacts.append(current_law_impacts)

            # Calculate metrics vs current policy baseline
            current_policy_metrics = calculate_metrics(
                reformed, tcja_baseline_data, year
            )
            current_policy_impacts = {
                "reform": f"{scenario_name}_vs_current_policy",
                "baseline": "Current Policy",
                **current_policy_metrics,
            }
            impacts.append(current_policy_impacts)

            # Save checkpoint
            impacts_df = pd.DataFrame(impacts)
            impacts_df.to_csv(f"impacts_{year}.csv", index=False)
            upload(
                f"impacts_{year}.csv",
                "nikhilwoodruff/salt-amt-data",
                f"impacts_{year}.csv",
            )
            completed_reforms.add(scenario_name)
            print(f"Saved checkpoint after {scenario_name}")

        except Exception as e:
            print(f"Error processing {scenario_name}: {str(e)}")
            with open("../data/error_log.txt", "a") as f:
                f.write(f"\nError on {scenario_name}: {str(e)}")
            continue

    return pd.DataFrame(impacts)

if __name__ == "__main__":
    print("Starting tax reform impact calculations...")

    year = os.environ["YEAR"]

    try:
        # Calculate impacts
        impacts_df = calculate_impacts(args.year)
        print("\nCalculation complete!")
        print(f"Results saved to impacts.csv")

        # Print summary statistics
        print("\nSummary:")
        print(
            f"Total scenarios processed: {len(impacts_df) // 2}"
        )  # Divide by 2 because each scenario has two baselines
        print(f"Total baseline comparisons: {len(impacts_df)}")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        with open("../data/error_log.txt", "a") as f:
            f.write(f"\nError in main execution: {str(e)}")
        raise