'''
Residency signaling project: for full methods see manuscript. In short,
applicants choose programs, programs choose applicants, and the match algorithm
is run :). 
'''

import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import random
import os
from collections import deque
from scipy.stats import gamma

# Gamma settings

GAMMA_SHAPE = 8.0
GAMMA_SCALE = 30 / 8  # mean of 30 applications

# there is a better way to do this but for now keep
# will have to manually add of these optimals if we add more...
HEATMAP_RESULTS_COLUMNS = [
    # same as generated from files
    # RANDOMIZED CONSTANTS
    'n_programs',
    'n_positions',
    'n_applicants',
    'interviews_per_spot',
    'max_applications',
    # CALCULATED CONSTANTS
    'applicants_per_position',
    'minimum_unmatched',
    'spots_per_program',
    'simulations_per_s',
    'study_min_signal',
    'study_max_signal',
    'result_file_prefix',
    # PROCESSED DATA PER CONSTANT SET
    # DERIVED FROM GET_SIMULATION_OPTIMALS
    # best signal RPP
    'best_signal_rpp',
    'reviews_per_program_best_rpp',
    'unmatched_applicants_best_rpp',
    'unfilled_spots_best_rpp',
    'pct_interview_given_signal_best_rpp',
    'pct_interview_given_nosignal_best_rpp',
    'pct_interview_given_application_best_rpp',
    'pct_of_app_match_via_signal_given_matched_best_rpp',
    # best signal UA
    'best_signal_ua',
    'reviews_per_program_best_ua',
    'unmatched_applicants_best_ua',
    'unfilled_spots_best_ua',
    'pct_interview_given_signal_best_ua',
    'pct_interview_given_nosignal_best_ua',
    'pct_interview_given_application_best_ua',
    'pct_of_app_match_via_signal_given_matched_best_ua',
    # best signal UFS
    'best_signal_ufs',
    'reviews_per_program_best_ufs',
    'unmatched_applicants_best_ufs',
    'unfilled_spots_best_ufs',
    'pct_interview_given_signal_best_ufs',
    'pct_interview_given_nosignal_best_ufs',
    'pct_interview_given_application_best_ufs',
    'pct_of_app_match_via_signal_given_matched_best_ufs',
    # best signal pigs 
    'best_signal_pigs',
    'reviews_per_program_best_pigs',
    'unmatched_applicants_best_pigs',
    'unfilled_spots_best_pigs',
    'pct_interview_given_signal_best_pigs',
    'pct_interview_given_nosignal_best_pigs',
    'pct_interview_given_application_best_pigs',
    'pct_of_app_match_via_signal_given_matched_best_pigs',
    # best signal pigns
    'best_signal_pigns',
    'reviews_per_program_best_pigns',
    
    'unmatched_applicants_best_pigns',
    'unfilled_spots_best_pigns',
    'pct_interview_given_signal_best_pigns',
    'pct_interview_given_nosignal_best_pigns',
    'pct_interview_given_application_best_pigns',
    'pct_of_app_match_via_signal_given_matched_best_pigns',
    # best signal piga
    'best_signal_piga',
    'reviews_per_program_best_piga',
    'unmatched_applicants_best_piga',
    'unfilled_spots_best_piga',
    'pct_interview_given_signal_best_piga',
    'pct_interview_given_nosignal_best_piga',
    'pct_interview_given_application_best_piga',
    'pct_of_app_match_via_signal_given_matched_best_piga',
    # best signal pamgs
    'best_signal_pamgs',
    'reviews_per_program_best_pamgs',
    'unmatched_applicants_best_pamgs',
    'unfilled_spots_best_pamgs',
    'pct_interview_given_signal_best_pamgs',
    'pct_interview_given_nosignal_best_pamgs',
    'pct_interview_given_application_best_pamgs',
    'pct_of_app_match_via_signal_given_matched_best_pamgs'
]


def read_existing_simulation_data(
    RESULTS_PATH: str,
    WIPE_DATA: bool) -> pd.DataFrame:
    
    # PATH WILL EXIST because it is checked in the viable_analysis and
    # check analysis directories
    
    if WIPE_DATA:
        print('Wiping analysis data.')
    
    if WIPE_DATA or not os.path.isfile(RESULTS_PATH):
        df = pd.DataFrame(columns = HEATMAP_RESULTS_COLUMNS)
        df.to_csv(RESULTS_PATH, index=False)
    else:
        df = pd.read_csv(RESULTS_PATH)
    return df

def read_constants(CONSTANTS_PATH: str) -> pd.DataFrame:
    df = pd.read_csv(CONSTANTS_PATH)
    not_integers = [
        'applicants_per_position',
        'result_file_prefix'
    ]
    for col in df.columns:
        if col not in not_integers:
            df[col] = df[col].astype(int)
    return df

def quartile_from_index(idx: int, n: int) -> int:
    q = n / 4
    b1, b2, b3 = int(q), int(2*q), int(3*q)
    if idx < b1:
        return 1
    elif idx < b2:
        return 2
    elif idx < b3:
        return 3
    else:
        return 4

class Applicant:
    
    # NOTE TO SELF - REMEMBER TO ADD ALL CLASS LEVEL OBJECTS TO
    # _SIMULATE_FOR_SIGNAL OR ELSE PARALLEL WILL BREAK

    # add slots to prevent memory with __dict__ and many instances of
    # applicant and program
    __slots__ = (
        "id",
        "quartile",
        "matched_program",
        "signaled_programs",
        "non_signaled_programs",
        "final_rank_list",
        "signaled_interviews",
        "non_signaled_interviews",
        "interviews_received"
    )

    # placeholders
    n_applications = int(-1) 
    n_applicants = int(-1)
    n_signals = int(-1)
    n_non_signals = int(-1)
    gamma_max_applications = False
    no_quartile = False
    RAND_APP_RANK_LIST_ORDER = False

    # class method to update signal numbers
    @classmethod
    def update_signal_number(cls, signal_number):
        cls.n_signals = signal_number
        cls.n_non_signals = cls.n_applications - signal_number
        

    @classmethod
    def update_analysis_settings(cls, gamma: bool, no_quartile: bool, rand_app_list_order: bool):
        cls.gamma_max_applications = bool(gamma)
        cls.no_quartile = bool(no_quartile)
        cls.RAND_APP_RANK_LIST_ORDER = bool(rand_app_list_order)


    def get_quartile(self):
        return quartile_from_index(self.id, self.n_applicants)

    def pick_programs(self,
                      all_programs: list,
                      program_quartile_list: dict,
                      signals: bool,
                      gamma_simulation_data: dict = {}):
        '''
    The purpose of this function is to create the list of applications that
    each applicant will send. It will also update the respective program
    lists of [received signals] and [received no signals] accordingly.

    This is where the 50/25/25 split will occur, where applicants will do
    50% of their applications in their own quartile, 25% in the quartile above
    (or the same if top), and 25% in the quartile below (or the same if bottom).

    Have to do signals and non-signals separately to make sure that split is
    maintained.

    Will first see how many times 4 divides the number of signals/non-signals,
    the remainder will go to the same quartile. This is to handle signals or
    application numbers that are not divisible by 4.

    Applicant is the class defined above.
    
    NOTE: there is a potential problem which is when there are not enough
    programs in a quartile to satisfy the application needs of an applicant.
    
    For example, with the 50/25/25 rule, there is a chance to sample 75%
    of programs within your own quartile (1st or last).
    
    If there are no programs left in quartile, currently we just exit ...
    if less then what you want just pick all of them remaining
    
    Can also just sample other quartiles if needed
    '''
        applications = []
        length = 0
        if Applicant.gamma_max_applications:
            if signals:
                length = gamma_simulation_data['n_signals'][self.id]
            else:
                length = gamma_simulation_data['n_non_signals'][self.id]
        else:
            length = Applicant.n_signals if signals else Applicant.n_non_signals
            
        already_chosen_programs = set(
            self.signaled_programs + self.non_signaled_programs)

        maximum_remaining_programs = len(all_programs) - len(already_chosen_programs)

        # very easy to ignore quartiles. just apply randomly
        
        if Applicant.no_quartile:
            available_programs = [
                p.id for p in all_programs if p.id not in already_chosen_programs]
            # pick randomly from available
            # length = min(length, len(available_programs))
            if length > len(available_programs):
                length = len(available_programs)
            program_choices = np.random.choice(
                available_programs, size = length, replace = False)
            for choice in program_choices:
                applications.append(choice)
                # you need this already chosen program since you
                # need to call again for next quartile
                already_chosen_programs.add(choice)
                if signals:
                    all_programs[choice].received_signals.append(self.id)
                else:
                    all_programs[choice].received_no_signals.append(self.id)
            return applications
        
        divisions_of_4 = length//4  # divisions of 4 go to quartiles
        remainder = length % 4  # all remainder go to same quartile
        own_quartile = self.quartile
        quartile_above = (
            self.quartile - 1) if self.quartile > 1 else self.quartile
        quartile_below = (
            self.quartile + 1) if self.quartile < 4 else self.quartile

        # this creates a list of the quartiles sorted by their closeness to the applicant's quartile so that we can later step into quartiles by distance
        quartiles_by_closeness = sorted(
            program_quartile_list.keys(),
            key=lambda q: abs(q - self.quartile))

        def _add_to_application_list(num, quartile):
            if num <= 0:
                return

            # First: try to get programs from the target quartile
            program_choices = program_quartile_list[quartile]  # list of IDs
            available = [p for p in program_choices
                         if p not in already_chosen_programs]

            # If this quartile alone doesn't have enough distinct programs,
            # top up from other quartiles, starting with those closest
            # to the APPLICANT'S quartile.
            if len(available) < num:
                # note: this if statement never occurs in the
                # base case scenario and is only for edge cases
                # when there are not enough available programs in the
                # quartile of interest
                needed = num - len(available)

                for q in quartiles_by_closeness:
                    if q == quartile:
                        # we've already pulled from this quartile
                        continue

                    candidates_q = [
                        p for p in program_quartile_list[q]
                        if (p not in already_chosen_programs)
                        and (p not in available)
                    ]
                    if not candidates_q:
                        continue

                    if len(candidates_q) <= needed:
                        # take all of them
                        available.extend(candidates_q)
                        needed -= len(candidates_q)
                    else:
                        # randomly choose just enough from this quartile
                        extra = np.random.choice(
                            candidates_q, size=needed, replace=False
                        )
                        available.extend(extra.tolist())
                        needed = 0

                    if needed == 0:
                        break

            # If there are no programs left at all, we're done
            if not available:
                return

            # If, even after topping up, there are fewer than num programs,
            # clamp num to what we actually have.
            if len(available) < num:
                num = len(available)

            # Final sample without replacement from `available`
            choices = np.random.choice(available, size=num, replace=False)
            for choice in choices:
                applications.append(choice)
                # you need this already chosen program since you
                # need to call again for next quartile
                already_chosen_programs.add(choice)
                if signals:
                    all_programs[choice].received_signals.append(self.id)
                else:
                    all_programs[choice].received_no_signals.append(self.id)

        _add_to_application_list(divisions_of_4, quartile_above)
        _add_to_application_list(divisions_of_4, quartile_below)
        _add_to_application_list(2*divisions_of_4 + remainder, own_quartile)

        if len(applications) != length:
            print(
                f"Warning: Expected {length} applications but got {len(applications)} for applicant {self.id}. However, maximum remaining programs is {maximum_remaining_programs}.")

        return applications

    def __init__(self,
                 id_index: int,
                 programs: list,
                 program_quartile_list: dict,
                 gamma_simulation_data: dict = {}):
        self.id = id_index
        self.quartile = self.get_quartile()
        self.matched_program = None
        self.signaled_programs = []
        self.non_signaled_programs = []
        
        self.signaled_programs = self.pick_programs(
            programs, program_quartile_list, True, gamma_simulation_data
        )
        self.non_signaled_programs = self.pick_programs(
            programs, program_quartile_list, False, gamma_simulation_data
        )
                
        if Applicant.RAND_APP_RANK_LIST_ORDER:
            # shuffle in place
            random.shuffle(self.signaled_programs)
            random.shuffle(self.non_signaled_programs)
        else:
            self.signaled_programs.sort()  # sort in ascending order
            self.non_signaled_programs.sort()  # sort in ascending order
            
        # note that final rank list is
        # signaled programs followed by non-signaled programs, just to give
        # a sense of "reality" with signals being prioritized
        # NOTE: this final rank list 
        self.final_rank_list = (
            self.signaled_programs + self.non_signaled_programs
        )
        self.signaled_interviews = []
        self.non_signaled_interviews = []
        self.interviews_received = []


class Program:
    # again, use slots to save memory and prevent __dict__ creation
    __slots__ = (
        "id",
        "quartile",
        "received_signals",
        "received_no_signals",
        "reviewed_applications",
        "final_rank_list",
        "tentative_matches",
        "n_positions",
        "n_interviews",
        "rank_index",    # added in stable_match
    )

    # NOTE TO SELF - REMEMBER TO ADD ALL CLASS LEVEL OBJECTS TO
    # _SIMULATE_FOR_SIGNAL OR ELSE PARALLEL WILL BREAK

    RAND_PROG_RANK_LIST_ORDER = False
    

    @classmethod
    def update_analysis_settings(cls, random_prog_rank_list_order: bool):
        cls.RAND_PROG_RANK_LIST_ORDER = bool(random_prog_rank_list_order)


    def get_quartile(self, constants):
        return quartile_from_index(self.id, int(constants["n_programs"]))

    def create_final_rank_list_and_count_reviews(self):
        '''
        When this function is called, applicatants have already applied
        and signaled to programs. Now, the goal is to fill the program 
        interviews. So, the logic is:
        
        0. Reviewed applications = len(received signals)
        1. Sort the received signals and non-signals in ascending order
        2. If len(received signals) >= num_interviews:
            final_rank_list = received signals [0:num_interviews]
        3. If len(received signals) < num_interviews:
            reviewed applications += len(received no signals)
            final_rank_list = 
                received signals + received no signals [0:remaining spots]
                
        Note: also support for ignoring applicant ranking and randomly
        ranking people
                
        '''
        # must review all signals
        self.reviewed_applications = len(self.received_signals)
        signal_list = []
        
        # behavior for random program list orders
        
        if Program.RAND_PROG_RANK_LIST_ORDER:
            # shuffle IN PLACE
            random.shuffle(self.received_signals)
            signal_list = list(self.received_signals)
        else:
            signal_list = sorted(self.received_signals)
        
        if len(self.received_signals) >= self.n_interviews:
            self.final_rank_list = signal_list[0:self.n_interviews]
        else:
            remaining_spots = self.n_interviews - len(self.received_signals)
            # IMPORTANT: can relax this assumption of having to review everybody
            # if you don't fill with signals
            # can do some greedy optimization but really it doesn't
            # make a difference for local minimum
            self.reviewed_applications += len(self.received_no_signals)
            # final rank list always prioritizes signals
            no_signal_list = []
            if Program.RAND_PROG_RANK_LIST_ORDER:
                # randomize order they are added on
                random.shuffle(self.received_no_signals)
                no_signal_list = self.received_no_signals
            else:
                no_signal_list = sorted(self.received_no_signals)    
            self.final_rank_list = (
                signal_list +
                no_signal_list[0:remaining_spots])

    def __init__(self, id_index, constants, positions):
        self.id = id_index
        self.n_positions = positions
        self.n_interviews = self.n_positions * constants['interviews_per_spot']
        self.quartile = self.get_quartile(constants)
        self.received_signals = []
        self.received_no_signals = []
        self.reviewed_applications = 0
        self.final_rank_list = []  # this is length of interviews_per_program
        self.tentative_matches = []



def get_quartile_dict(n) -> dict:
    quartile_size = n/4
    quartiles = {
        1: [i for i in range(0, int(quartile_size))],
        2: [i for i in range(int(quartile_size), int(quartile_size*2))],
        3: [i for i in range(int(quartile_size*2), int(quartile_size*3))],
        4: [i for i in range(int(quartile_size*3), n)]
    }
    return quartiles


def stable_match(applicants: list, programs: list):
    """
    Implements the applicant-proposing deferred acceptance algorithm (NRMP style).
    
    Applicants propose to programs in order of their final_rank_list.
    Programs tentatively hold the best applicants (according to their final_rank_list)
    up to their capacity, rejecting others.
    
    The algorithm terminates when no applicant has a program left to propose to.
    
    Compared to last push, this version just precomputes program rankings
    and uses a queue for free applicants for speed.
    """

    # Precompute constant-time rank lookups for each program
    # used later to see if applicant is in the PROGRAM's rank list, fast lookup
    # with dictionary
    for program in programs:
        # app_id -> position in final_rank_list
        program.rank_index = {
            app_id: pos for pos, app_id in enumerate(program.final_rank_list)
        }
        # Ensure tentative_matches is empty at the start
        program.tentative_matches = []

    n_applicants = len(applicants)
    # deque for O(1) pops from left which we use often here
    free_applicants = deque(range(n_applicants))

    # Simple list, last time was dict this will be faster
    next_proposal_index = [0] * n_applicants

    while free_applicants:
        # fast pop for applicant ID to propose
        applicant_id = free_applicants.popleft()
        applicant = applicants[applicant_id]

        # If no programs left to propose to, they remain unmatched
        # already popped from free applicants
        if next_proposal_index[applicant_id] >= len(applicant.final_rank_list):
            continue

        # Next program on this applicant's list
        program_id = applicant.final_rank_list[next_proposal_index[applicant_id]]
        next_proposal_index[applicant_id] += 1
        program = programs[program_id]

        # Program does not rank this applicant -> effectively immediate rejection,
        # applicant will try again next time with the NEXT program
        if applicant_id not in program.rank_index:
            # remember to add back into free applicants
            free_applicants.append(applicant_id)
            continue

        # Program considers the proposal
        program.tentative_matches.append(applicant_id)

        # Sort by program’s preference
        # NOTE: For the case where the program randomly ranks applicants, we 
        # don't have to change anything since the program.final_rank_list
        # is already shuffled
        # HOWEVER IT does STILL prioritize signals over non-signals
        program.tentative_matches.sort(
            key=lambda aid: program.rank_index[aid]
        )

        # Keep only best spots_per_program applicants if over capacity
        # program always prefers best applicants
        if len(program.tentative_matches) > program.n_positions:
            # rejected is from n_postions : to end of list
            rejected = program.tentative_matches[program.n_positions:]
            # matches is start of list to : n positions
            program.tentative_matches = program.tentative_matches[:program.n_positions]

            # Rejected applicants become free again
            for rej in rejected:
                free_applicants.append(rej)

        # If the applicant is among program.tentative_matches, they are tentatively matched
        # and will only re-enter free_applicants if they are later rejected.

    # Finalize matches exactly as before
    for program in programs:
        for app_id in program.tentative_matches:
            applicants[app_id].matched_program = program.id

    return applicants, programs

def count_applicant_interview_rates(applicants: list):
    '''
    Counts the interview rates for signaled and non-signaled programs.
    
    Returns:
        signaled_interview_rate: fraction of signaled programs that gave interviews
        non_signaled_interview_rate: fraction of non-signaled programs that gave interviews
    '''
    
    results = {}
    
    total_signaled_interviews = 0
    total_signaled_programs = 0
    total_non_signaled_interviews = 0
    total_non_signaled_programs = 0

    for app in applicants:
        total_signaled_programs += len(app.signaled_programs)
        total_non_signaled_programs += len(app.non_signaled_programs)
        total_signaled_interviews += len(app.signaled_interviews)
        total_non_signaled_interviews += len(app.non_signaled_interviews)

    signaled_interview_rate = (
        total_signaled_interviews / total_signaled_programs
        if total_signaled_programs > 0 else 0.0
    )
    
    non_signaled_interview_rate = (
        total_non_signaled_interviews / total_non_signaled_programs
        if total_non_signaled_programs > 0 else 0.0
    )
    
    overall_interview_rate = (
        (total_signaled_interviews + total_non_signaled_interviews) / (
            total_signaled_programs + total_non_signaled_programs)
        )
    
    results['pct_interview_given_signal'] = signaled_interview_rate
    results['pct_interview_given_nosignal'] = non_signaled_interview_rate
    results['pct_interview_given_application'] = overall_interview_rate
    
    return results

def post_match_analysis(applicants: list, programs: list) -> dict:
    '''
    Counts the number of unmatched applicants and unfilled program spots.
    
    Returns:
        unmatched_applicants: number of applicants who did not match to any program
        unfilled_spots: total number of empty spots across all programs
    '''
    
    post_match_counts = {}
    
    post_match_counts['unmatched_applicants'] = sum(
        1 for app in applicants if app.matched_program is None)
    
    unfilled_spots = 0
    
    # for each applicant also calculate len(signaled_interviews)/
    # len(signaled_programs), len(signaled_interviews)/(len both)
    # len(non_signaled_interviews)/len(non_signaled_programs), 
    # len nonsignaled_interviews)/(len both)
    
    num_matched = 0
    num_matched_via_signal = 0
    for applicant in applicants:
        if applicant.matched_program is not None:
            num_matched += 1
            if applicant.matched_program in applicant.signaled_programs:
                num_matched_via_signal += 1

    post_match_counts['pct_of_app_match_via_signal_given_matched'] = (
        (num_matched_via_signal / num_matched) if num_matched > 0 else 0.0
    )
    for program in programs:
        filled_spots = len(program.tentative_matches)
        unfilled_spots += program.n_positions - filled_spots

    post_match_counts['unfilled_spots'] = unfilled_spots
    
    return post_match_counts


def _get_positions_per_program(constants):
    n_positions = constants['n_positions']
    n_programs = constants['n_programs']
    base = n_positions // n_programs
    remainder = n_positions - base*n_programs
    positions_per_program = [base for _ in range(n_programs)]
    if remainder == 0:
        return positions_per_program
    # select remainder number of ids from program_ids without replacement
    # sample does a unique pull
    programs_to_add_one = random.sample(
        range(n_programs), k = remainder)
    for program in programs_to_add_one:
        positions_per_program[program] += 1
    return positions_per_program
    

def run_simulation(s, constants, gamma_simulation_data={}):
    results = {}
    program_quartile_list = get_quartile_dict(constants['n_programs'])
    # in the refactoring for each program to have an independent number of
    # positions and thus interviews to offer, we must
    # change the initialization here

    positions_per_program = _get_positions_per_program(constants)     
    programs = [
        Program(
            j, constants, positions_per_program[j] # the index gets to how many
            # positions that program has
            ) for j in range(constants['n_programs'])]
    applicants = [
        Applicant(i, 
                  programs, 
                  program_quartile_list, 
                  gamma_simulation_data) for i in range(
            constants['n_applicants'])]
    for program in programs:
        program.create_final_rank_list_and_count_reviews()
    total_reviews = sum(
        [program.reviewed_applications for program in programs])
    avg_reviews_per_program = total_reviews / constants['n_programs']
    results['reviews_per_program'] = avg_reviews_per_program
    
    # now also calculate the applicant interviews received
    # TODO: One day maybe it is worth recording this data
    # but we do some calculations on it 
    # ultimately could probably JUST store matched programs, signals,
    # and non-signals, and that would tell all results ...
    for program in programs:
        for app_id in program.final_rank_list:
            applicants[app_id].interviews_received.append(program.id)
            if program.id in applicants[app_id].signaled_programs:
                applicants[app_id].signaled_interviews.append(program.id)
            else:
                applicants[app_id].non_signaled_interviews.append(program.id)
    
    # calculate average applicant interview statistics
    
    applicant_statistics = count_applicant_interview_rates(applicants)
    results['pct_interview_given_signal'] = applicant_statistics[
        'pct_interview_given_signal']
    results['pct_interview_given_nosignal'] = applicant_statistics[
        'pct_interview_given_nosignal']
    results['pct_interview_given_application'] = applicant_statistics[
        'pct_interview_given_application']
            
    # last step is for the matching algorithm, which is stable matching
    applicants_matched, programs_matched = stable_match(applicants, programs)
    
    # post matching counts
    post_match_counts = post_match_analysis(
        applicants_matched, programs_matched)
    
    results['unmatched_applicants'] = post_match_counts['unmatched_applicants']
    results['unfilled_spots'] = post_match_counts['unfilled_spots']
    results['pct_of_app_match_via_signal_given_matched'] = post_match_counts[
        'pct_of_app_match_via_signal_given_matched']
    
    return results

def get_simulation_optimals(
    simulation_results, signal_values):
    
    parameter_names_tails = {
        'reviews_per_program': 'rpp',
        'unmatched_applicants': 'ua',
        'unfilled_spots': 'ufs',
        'pct_interview_given_signal': 'pigs',
        'pct_interview_given_nosignal': 'pigns',
        'pct_interview_given_application': 'piga',
        'pct_of_app_match_via_signal_given_matched': 'pamgs'
    }
    
    parameter_optimization = {
        'reviews_per_program': 'min',
        'unmatched_applicants': 'min',
        'unfilled_spots': 'min',
        'pct_interview_given_signal': 'max',
        'pct_interview_given_nosignal': 'max',
        'pct_interview_given_application': 'max',
        'pct_of_app_match_via_signal_given_matched': 'max'
    }
    
    # get the best signal for each parameter optimum
    results = {}
    
    # helpers to get max and min
    def _get_max_parameter_mean(param_df, signal_values):
        max_value = float('-inf')
        best_s = None
        for s in signal_values:
            col = str(s)
            values = param_df[col].values
            mean = np.nanmean(values)
            if mean > max_value:
                max_value = mean
                best_s = s        
        return best_s
    
    def _get_min_parameter_mean(param_df, signal_values):
        min_value = float('inf')
        best_s = None
        for s in signal_values:
            col = str(s)
            values = param_df[col].values
            mean = np.nanmean(values)
            if mean < min_value:
                min_value = mean
                best_s = s        
        return best_s
    
    for parameter in parameter_names_tails.keys():
        if parameter_optimization[parameter] is None:
            continue
        best_s = None
        param_df = simulation_results[
            simulation_results['Parameter'] == parameter]
        if parameter_optimization[parameter] == 'max':
            best_s = _get_max_parameter_mean(param_df, signal_values)
        else:
            best_s = _get_min_parameter_mean(param_df, signal_values)
        result_string = f"best_signal_{parameter_names_tails[parameter]}"
        results[result_string] = best_s
    
    # now should have results filled out for the best signal 
    # with each key being the best_signal_{tail}
    # now add on the mean reviews for each other parameter
    # at that value
    
    # it looks like a double loop but it's intentional, you first
    # get the best signal for each parameter, then you generate
    # the mean for each parameter AT that single best signal
    for parameter in parameter_names_tails.keys():
        # get that same best signal key
        tail = parameter_names_tails[parameter]
        result_string = f"best_signal_{tail}"
        optimal_signal_integer = results[result_string]
        optimal_signal_string = str(optimal_signal_integer)
        
        for parameter_name in parameter_names_tails.keys():
            # take the mean AT that signal key
            results[f"{parameter_name}_best_{tail}"] = np.nanmean(
                simulation_results[
                    simulation_results['Parameter'] == parameter_name
                ][optimal_signal_string].values)
            
    return results


def process_simulation_heatmap(
        final_simulation_dataframe: pd.DataFrame,
        constants: pd.Series,
        signal_values) -> pd.DataFrame:
    
    # the additional data we need now is
    # best signal value, mean reviews per program, mean unmatched applicants
    results = get_simulation_optimals(
        final_simulation_dataframe,
        signal_values)
    
    final_dataframe = {}
    for key in HEATMAP_RESULTS_COLUMNS:
        if key in constants.index:
            final_dataframe[key] = constants[key]
        elif key in results.keys():
            final_dataframe[key] = results[key]
        else:
            print(f"Warning: Key {key} not found in constants or results.")
            final_dataframe[key] = np.nan        
    
    return pd.DataFrame([final_dataframe])


def _simulate_for_signal(
        constants: pd.Series,
        signal_value: int,
        seed: int,
        analysis_settings: dict):
    """
     Single simulation run for a given signal_value.
    Did move applicant.update_signal_number inside so that each process
    is self-contained.
    
    It's actually real important to give them a random seed here because
    there can be a risk with parallelizaition that multiple processes
    get the same random seed and thus produce correlated results.
    """
    
    Applicant.update_analysis_settings(
        analysis_settings['GAMMA_MAX_APPLICATIONS'],
        analysis_settings['NO_QUARTILE'],
        analysis_settings['RAND_APP_RANK_LIST_ORDER']
       )
    Program.update_analysis_settings(
        analysis_settings['RAND_PROG_RANK_LIST_ORDER']
    )
    
    np.random.seed(seed)
    random.seed(seed)

    Applicant.n_applicants = constants['n_applicants']
    
    # base case scenario OK to set max applications and
    # also to update signal number
    
    # otherwise this occurs dynamically for each applicant

    gamma_n_applications = []
    gamma_n_signals = []
    gamma_n_non_signals = []

    if Applicant.gamma_max_applications:
        # choose gamma values up to the number of applicants
        while len(gamma_n_applications) < constants['n_applicants']:
            sample = gamma.rvs(
                a=GAMMA_SHAPE,
                scale=GAMMA_SCALE,
                size=1)
            sample = int(sample)
            # MUST have at least 5 applications
            # see manuscript for details
            if sample >= 5:
                gamma_n_applications.append(sample)
                # choose s if we can, otherwise go to max supported signals
                # which would be number of applications
                n_signals = min(signal_value, sample)
                n_non_signals = sample - n_signals
                gamma_n_signals.append(n_signals)
                gamma_n_non_signals.append(n_non_signals)
    else:
        Applicant.n_applications = constants['max_applications']
        Applicant.update_signal_number(signal_value)
    
    gamma_simulation_data = {
        'n_applications': gamma_n_applications,
        'n_signals': gamma_n_signals,
        'n_non_signals': gamma_n_non_signals
    }
    
    simulation_results = run_simulation(
        signal_value, 
        constants,
        gamma_simulation_data=gamma_simulation_data)
    
    return simulation_results


def run_simulation_heatmap(
    CONSTANTS: pd.Series,
    analysis_settings: dict) -> dict:
    '''
    Runs a full heatmap simulation for a given set of CONSTANTS.
    Returns a DataFrame with the results for this constant set.
    '''
    min_s = int(CONSTANTS["study_min_signal"])
    max_s = int(CONSTANTS["study_max_signal"])
    signal_values = list(range(min_s, max_s + 1))
    signal_range = [str(v) for v in signal_values]
    df_simulation = pd.DataFrame(columns=['Parameter'] + signal_range)
    
    # to build array
    n_signals = len(signal_values)
    n_runs = CONSTANTS['simulations_per_s']
    
    # this dataframe holds all simulation results for each signal range
    # and will append to the df_simulation later where first column is
    # parameter and rest are the values at the signal values
    simulation_results_names = [
        'unmatched_applicants',
        'unfilled_spots',
        'reviews_per_program',
        'pct_interview_given_signal',
        'pct_interview_given_nosignal',
        'pct_interview_given_application',
        'pct_of_app_match_via_signal_given_matched'
    ]
    
    # initialize array to hold all of simulation results
    # for each parameter
    # array is [run][signal_value]
    simulation_results = {
        m: np.empty((n_runs, n_signals)) for m in simulation_results_names}


    # -------- PARALLEL ------
    from concurrent.futures import ProcessPoolExecutor

    # One process pool reused across all batches
    with ProcessPoolExecutor() as executor:
        for i in range(CONSTANTS['simulations_per_s']):
            print(
                f"Starting simulation batch {i+1} of "
                f"{CONSTANTS['simulations_per_s']}"
            )

            # Give each (batch, signal_value) combo its own seed to
            # avoid accidental correlation between runs
            seeds = np.random.randint(
                0, 2**32 - 1, size=len(signal_values), dtype=np.uint64
            )

            # Run this batch in parallel across signal values
            # each call to simulate for run_simulation(s) is independent
            # embarrassingly parallel ...
            # pickles simulate for signal and sends payload to
            # idle worker process -> returns future object
            # each worker does the job and returns results
            futures = [
                executor.submit(
                    _simulate_for_signal, 
                    CONSTANTS, 
                    s_val, 
                    int(seed), 
                    analysis_settings)
                for s_val, seed in zip(signal_values, seeds)
            ]

            # Collect results, should each be a dictionary
            # with all of the following values
            results = [f.result() for f in futures]
            
            for j, sim in enumerate(results):
                for m in simulation_results_names:
                    # i is defined above as the "run" number
                    # j is the signal value index
                    simulation_results[m][i,j] = sim[m]
                        
    dataframe_rows = []
    for m in simulation_results_names:
        for run in range(n_runs):
            row = [m] + simulation_results[m][run].tolist()
            dataframe_rows.append(row)
    
    df_simulation = pd.DataFrame(
        dataframe_rows,
        columns=['Parameter'] + signal_range
    )        

    processed_results = process_simulation_heatmap(
        df_simulation, CONSTANTS, signal_values)

    all_results = {
        'all_data': df_simulation,
        'processed_results': processed_results
    }

    return all_results


def _makedirs_with_msg(path: str, label: str) -> bool:
    """Create directory if needed. Return True if it was created."""
    if not path:
        path = "."
    if os.path.isdir(path):
        return False
    os.makedirs(path, exist_ok=True)
    print(f"Created {label} directory: {path}")
    return True


def _empty_dir_with_msg(dir_path: str, label: str) -> bool:
    """
    Deletes all contents of dir_path (files + subdirs), keeps the directory.
    Returns True if anything was deleted.
    """
    p = Path(dir_path)
    if not p.exists():
        return False

    deleted_any = False
    for child in p.iterdir():
        deleted_any = True
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    if deleted_any:
        print(f"Emptied {label}: {dir_path}")
    return deleted_any


def check_analysis_directories(analysis_settings) -> int:
    created_any = False

    # RESULTS_PATH is a file -> ensure parent dir exists
    results_csv = analysis_settings["RESULTS_PATH"]
    created_any |= _makedirs_with_msg(
        os.path.dirname(results_csv), "RESULTS_PATH parent")

    # ALL_DATA_EXPORT_PATH is a directory -> ensure it exists
    export_dir = analysis_settings["ALL_DATA_EXPORT_PATH"]
    created_any |= _makedirs_with_msg(export_dir, "ALL_DATA_EXPORT_PATH")

    # If WIPE_DATA: empty export directory contents
    if analysis_settings.get("WIPE_DATA", False):
        _empty_dir_with_msg(export_dir, "ALL_DATA_EXPORT_PATH contents")

    # CONSTANTS_PATH is a file -> ensure parent dir exists, then require file exists
    constants_file = analysis_settings["CONSTANTS_PATH"]
    created_any |= _makedirs_with_msg(os.path.dirname(
        constants_file), "CONSTANTS_PATH parent")

    if not created_any:
        print("Analysis directories already exist.")

    if os.path.isfile(constants_file):
        return 1

    print(
        f"CONSTANTS_PATH file does not exist for analysis: "
        f"{analysis_settings.get('ANALYSIS_NAME', 'UNKNOWN')}\n"
        f"Missing: {constants_file}"
    )
    return 0

def run_analysis(analysis_settings):
    
    viable_analysis = check_analysis_directories(analysis_settings)
    if viable_analysis == 0:
        return    
    constants = read_constants(analysis_settings['CONSTANTS_PATH'])
    # check if existing data
    simulation_data = read_existing_simulation_data(
        analysis_settings['RESULTS_PATH'],
        analysis_settings['WIPE_DATA'])
    
    unique_constants = set(constants['result_file_prefix'].values)
    print(f"Total unique constants provided: {len(unique_constants)}.")
    already_run_heatmaps = set(simulation_data['result_file_prefix'].values)
    print(f"Number of already run/saved heatmaps: {len(already_run_heatmaps)}.")

    heatmaps_not_in_constants = already_run_heatmaps - unique_constants
    print(
        f"Number of heatmaps not in current constants file: {len(heatmaps_not_in_constants)}.")
    
    to_run_constants = unique_constants - already_run_heatmaps
    
    print(f"Number of constants left to run: {len(to_run_constants)}.")
    to_run_constants_df = constants[
       constants['result_file_prefix'].isin(to_run_constants)]
    
    print('Number of cores:', os.cpu_count())

    iteration = 0
    for _, CONSTANTS in to_run_constants_df.iterrows():
        iteration += 1
        all_results = run_simulation_heatmap(CONSTANTS, analysis_settings)
        simulation_results = all_results['all_data']
        processed_results = all_results['processed_results']
        
        # print all results to csv for this constant set, in case 
        # we need to dissect down each signal later
        # will be easy to process individual files using 
        # existing scripts

        simulation_results.to_csv(
            f"{analysis_settings['ALL_DATA_EXPORT_PATH']}{CONSTANTS['result_file_prefix']}.csv", 
            index=False)
        
        # Avoid concat-with-empty/all-NA behavior that pandas is changing
        # this is the summary simulation data that we export to
        # heatmap_results.csv
        if simulation_data.empty:
            simulation_data = processed_results.copy()
        else:
            simulation_data = pd.concat(
                [simulation_data, processed_results], 
                ignore_index=True)
        if iteration % 5 == 0:
            print(f"Completed {iteration} constant sets. Exporting interim results.")
            simulation_data.to_csv(analysis_settings['RESULTS_PATH'], index=False)
    simulation_data.to_csv(analysis_settings['RESULTS_PATH'], index=False)


def to_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n", ""}:
        return False
    # fallback: treat anything else as True
    return True

if __name__ == "__main__":
    
    analysis_settings = pd.read_csv('analysis_settings.csv')
    
    # NOTE: HAVE TO MANUALLY UPDATE IF ANY BINARY FLAGS ARE ADDED
    for col in [
        "GAMMA_MAX_APPLICATIONS", 
        "NO_QUARTILE",
        "WIPE_DATA", 
        "RUN_ANALYSIS",
        "RAND_APP_RANK_LIST_ORDER",
        "RAND_PROG_RANK_LIST_ORDER"]:
        analysis_settings[col] = analysis_settings[col].map(to_bool)
    
    # the other variables will be strings by default
    
    for index, row in analysis_settings.iterrows():
        settings = row.to_dict()
        if settings['RUN_ANALYSIS'] is False:
            print(f"Will not run {settings['ANALYSIS_NAME']} as specified in analysis_settings.csv")
            continue
        print(
            f"Running analysis: {settings['ANALYSIS_NAME']}.")        
        run_analysis(settings)