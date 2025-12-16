# heatmap analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = 'heatmap_results/heatmap_results.csv'

independent_variables = [
    # modified in random generator
    'n_programs',
    'n_positions',
    'n_applicants',
    'interviews_per_spot',
    'max_applications'
]

calculated_constants = [
    'applicants_per_position',
    'minimum_unmatched',
    'spots_per_program',
    'simulations_per_s',
    'study_min_signal',
    'study_max_signal',
    'result_file_prefix'
]

# outcomes
dependent_variables = [
    'best_signal_rpp',
    'reviews_per_program_best_rpp',
    'unmatched_applicants_best_rpp',
    'unfilled_spots_best_rpp',
    'pct_interview_given_signal_best_rpp',
    'pct_interview_given_nosignal_best_rpp',
    'pct_interview_given_application_best_rpp',
    # best signal UA
    'best_signal_ua',
    'reviews_per_program_best_ua',
    'unmatched_applicants_best_ua',
    'unfilled_spots_best_ua',
    'pct_interview_given_signal_best_ua',
    'pct_interview_given_nosignal_best_ua',
    'pct_interview_given_application_best_ua',
    # best signal UFS
    'best_signal_ufs',
    'reviews_per_program_best_ufs',
    'unmatched_applicants_best_ufs',
    'unfilled_spots_best_ufs',
    'pct_interview_given_signal_best_ufs',
    'pct_interview_given_nosignal_best_ufs',
    'pct_interview_given_application_best_ufs',
    # best signal pigs 
    'best_signal_pigs',
    'reviews_per_program_best_pigs',
    'unmatched_applicants_best_pigs',
    'unfilled_spots_best_pigs',
    'pct_interview_given_signal_best_pigs',
    'pct_interview_given_nosignal_best_pigs',
    'pct_interview_given_application_best_pigs',
    # best signal pigns
    'best_signal_pigns',
    'reviews_per_program_best_pigns',
    'unmatched_applicants_best_pigns',
    'unfilled_spots_best_pigns',
    'pct_interview_given_signal_best_pigns',
    'pct_interview_given_nosignal_best_pigns',
    'pct_interview_given_application_best_pigns',
    # best signal piga
    'best_signal_piga',
    'reviews_per_program_best_piga',
    'unmatched_applicants_best_piga',
    'unfilled_spots_best_piga',
    'pct_interview_given_signal_best_piga',
    'pct_interview_given_nosignal_best_piga',
    'pct_interview_given_application_best_piga'
]

df = pd.read_csv(INPUT_PATH)