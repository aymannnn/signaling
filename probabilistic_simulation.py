'''
Residency signaling project: for full methods see manuscript. In short,
applicants choose programs, programs choose applicants, and the match algorithm
is run :). 
'''

import numpy as np
import pandas as pd
import os
from collections import deque
import os
import pyarrow.parquet as pq
import probabilistic_simulation_helpers as helpers
from concurrent.futures import ProcessPoolExecutor

# can modify here to leave whatever number of cores free
CORES_TO_USE = max(1, os.cpu_count()-4)

def calculate_results(
    applicants: list, 
    programs: list,
    sim_settings: dict):

    results = {
        'sim_id': sim_settings['sim_id'],
        'program': sim_settings['program'],
        'signals': sim_settings['signals'],
        'n_programs': sim_settings['n_programs'],
        'n_positions': sim_settings['n_positions'],
        'n_applicants': sim_settings['n_applicants']
    }
    

    total_signaled_interviews = 0
    total_signaled_programs = 0
    total_non_signaled_interviews = 0
    total_non_signaled_programs = 0

    unmatched_applicants = 0
    matched_at_signal = 0
    matched_no_signal = 0
    program_decile_matches = {i: [0]*10 for i in range(1, 11)}

    for app in applicants:
        total_signaled_programs += len(app.signaled_programs)
        total_non_signaled_programs += len(app.non_signaled_programs)
        total_signaled_interviews += len(app.signaled_interviews)
        total_non_signaled_interviews += len(app.non_signaled_interviews)
        # did not match
        if app.matched_program is None:
            unmatched_applicants += 1
        # matched
        else:
            if app.matched_program in app.signaled_programs:
                matched_at_signal += 1
            else:
                matched_no_signal += 1
            pro = app.matched_program
            # decile is from 1 to 10 inclusive so index is -1 ... care
            program_decile_matches[programs[pro].decile][app.decile-1] += 1
                
    total_matches = 0
    total_positions = 0
    total_reviews = 0
    unfilled_spots = 0
    
    for pro in programs:
        matches_at_pro = len(pro.tentative_matches)
        total_matches += matches_at_pro
        total_reviews += pro.reviewed_applications
        total_positions += pro.n_positions
        unfilled_spots += pro.n_positions-matches_at_pro

    results['p_int_given_signal'] = (
        total_signaled_interviews / total_signaled_programs 
        if total_signaled_programs > 0 else 0.0
    )
    results['p_int_given_nosignal'] = (
        total_non_signaled_interviews / total_non_signaled_programs
        if total_non_signaled_interviews > 0 else 0.0
    )
    
    results['pct_matches_from_signal'] = (
            matched_at_signal / total_matches if total_matches > 0 else 0.0)
    results['pct_match_from_nosignal'] = 1 - results['pct_matches_from_signal']
    results['reviews_per_program'] = total_reviews / sim_settings['n_programs']
    results['unfilled_positions'] = sim_settings['n_positions'] - total_matches
    
    if results['unfilled_positions'] != unfilled_spots:
        print(f"Warning: Unfilled positions calculated differently: {results['unfilled_positions']} vs {unfilled_spots}")
    
    # add on the decile plots
    for i in range(1, 11):
        total = sum(program_decile_matches[i])
        for j in range(10):
            if total > 0:
                program_decile_matches[i][j] = (
                    program_decile_matches[i][j] / total)
            results[f'p{i}_a{j+1}'] = program_decile_matches[i][j]

    return results


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
        # note: final_rank_list now is only the programs that the applicant
        # received an interview at, so it's actually a final rank list
        # they may not be on the programs rank list but this models real life
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


class Applicant:

    # add slots to prevent memory with __dict__ and many instances of
    # applicant and program
    __slots__ = (
        "id",
        "quartile",
        "decile",
        "matched_program",
        "signaled_programs",
        "non_signaled_programs",
        "final_rank_list",
        "signaled_interviews",
        "non_signaled_interviews",
        "interviews_received",
        "n_applications",
        "n_signals",
        "n_non_signals",
        "no_quartile",
        "random_rank_list_order",
    )

    def create_final_rank_list(self):
        if self.random_rank_list_order:
            # shuffle in place
            np.random.shuffle(self.signaled_interviews)
            np.random.shuffle(self.non_signaled_interviews)
        else:
            self.signaled_interviews.sort()  # sort in ascending order
            self.non_signaled_interviews.sort()  # sort in ascending order
        self.final_rank_list = self.signaled_interviews + self.non_signaled_interviews
        
    def pick_programs(self,
                      all_programs: list,
                      position_lookups: dict,
                      fill_signals: bool):
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
        length = self.n_signals if fill_signals else self.n_non_signals

        already_chosen_programs = set(
            self.signaled_programs + self.non_signaled_programs)

        maximum_remaining_programs = len(
            all_programs) - len(already_chosen_programs)

        # very easy to ignore quartiles. just apply randomly
        
        if self.no_quartile:
            available_programs = [
                p.id for p in all_programs if p.id not in already_chosen_programs]
            # pick randomly from available
            n_to_pick = min(length, len(available_programs))
            program_choices = np.random.choice(
                available_programs, 
                size=n_to_pick, 
                replace=False).tolist()
            for choice in program_choices:
                choice = int(choice)  # proper type casting for later json
                applications.append(choice)
                # you need this already chosen program since you
                # need to call again for next quartile
                already_chosen_programs.add(choice)
                if fill_signals:
                    all_programs[choice].received_signals.append(self.id)
                else:
                    all_programs[choice].received_no_signals.append(self.id)
            return applications
        
        # for not applying randomly ....

        divisions_of_4 = length//4  # divisions of 4 go to quartiles
        remainder = length % 4  # all remainder go to same quartile
        own_quartile = self.quartile
        quartile_above = (
            self.quartile - 1) if self.quartile > 1 else self.quartile
        quartile_below = (
            self.quartile + 1) if self.quartile < 4 else self.quartile

        # this creates a list of the quartiles 
        # sorted by their closeness to the applicant's quartile 
        # so that we can later step into quartiles by distance
        quartiles_by_closeness = sorted(
            position_lookups['program_quartile_to_id_list'].keys(),
            key=lambda q: abs(q - self.quartile))

        def _add_to_application_list(num, quartile):
            if num <= 0:
                return

            # First: try to get programs from the target quartile
            available = [
                p for p in position_lookups[
                    'program_quartile_to_id_list'][quartile]
                if p not in already_chosen_programs
            ]

            initial_pull = min(num, len(available))

            # IMPORTANT: convert to list so concatenation works correctly
            program_choices = (
                np.random.choice(available, size=initial_pull,
                                 replace=False).tolist()
                if initial_pull > 0 else []
            )

            chosen_set = set(program_choices)

            # Top up if needed, pulling from other quartiles
            # maybe have to check to see if this biases a certain way...
            if initial_pull < num:
                needed = num - initial_pull
                for q in quartiles_by_closeness:
                    if q == quartile:  # already checked
                        continue
                    if needed == 0:
                        break

                    candidates_q = [
                        p for p in position_lookups[
                            'program_quartile_to_id_list'][q]
                        if (p not in already_chosen_programs) and (p not in chosen_set)
                    ]
                    if not candidates_q:
                        continue

                    if len(candidates_q) <= needed:
                        program_choices.extend(candidates_q)
                        chosen_set.update(candidates_q)
                        needed -= len(candidates_q)
                    else:
                        additional_choices = np.random.choice(
                            candidates_q, size=needed, replace=False
                        ).tolist()
                        program_choices.extend(additional_choices)
                        chosen_set.update(additional_choices)
                        needed = 0

            for choice in program_choices:
                choice = int(choice)  # proper type casting for later json
                applications.append(choice)
                already_chosen_programs.add(choice)
                if fill_signals:
                    all_programs[choice].received_signals.append(self.id)
                else:
                    all_programs[choice].received_no_signals.append(self.id)

        _add_to_application_list(divisions_of_4, quartile_above)
        _add_to_application_list(divisions_of_4, quartile_below)
        _add_to_application_list(2*divisions_of_4 + remainder, own_quartile)

        return applications

    def __init__(
        self,
        id_index: int,
        programs: list,
        position_lookups: dict,
        n_applications: int,
        n_signals: int,
        no_quartile = False,
        random_rank_list_order = False):
        
        self.id = id_index
        self.quartile = position_lookups['applicant_id_to_quartile'][id_index]
        self.decile = position_lookups['applicant_id_to_decile'][id_index]
        self.matched_program = None
        self.n_applications = min(len(programs), n_applications)
        # in case signals > applications
        self.n_signals = min(n_signals, n_applications)
        self.n_non_signals = n_applications - self.n_signals
        self.no_quartile = no_quartile
        self.random_rank_list_order = random_rank_list_order

        
        self.signaled_programs = []
        self.non_signaled_programs = []
        
        self.signaled_programs = self.pick_programs(
            programs, position_lookups, True)
        
        self.non_signaled_programs = self.pick_programs(
            programs, position_lookups, False)
        
        self.signaled_interviews = []
        self.non_signaled_interviews = []
        self.interviews_received = []
        self.final_rank_list = []

class Program:
    # use slots to save memory and prevent __dict__ creation
    __slots__ = (
        "id",
        "quartile",
        "decile",
        "received_signals",
        "received_no_signals",
        "reviewed_applications",
        "final_rank_list",
        "tentative_matches",  # note, tentative matches becomes final matches
        "n_positions",
        "n_interviews",
        "rank_index",    # added in stable_match
        "random_rank_list_order"
    )

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

        if self.random_rank_list_order:
            # shuffle IN PLACE
            np.random.shuffle(self.received_signals)
            signal_list = list(self.received_signals)
        else:
            signal_list = sorted(self.received_signals)

        if len(self.received_signals) >= self.n_interviews:
            self.final_rank_list = signal_list[0:self.n_interviews]
        else:
            remaining_spots = self.n_interviews - len(self.received_signals)
            # Can one day relax this assumption of having to review everybody
            # if you don't fill with signals
            # can do some greedy optimization but really it doesn't
            # make a difference for local minimum
            self.reviewed_applications += len(self.received_no_signals)
            # final rank list always prioritizes signals
            no_signal_list = []
            if self.random_rank_list_order:
                # randomize order they are added on
                np.random.shuffle(self.received_no_signals)
                no_signal_list = list(self.received_no_signals)
            else:
                no_signal_list = sorted(self.received_no_signals)
            
            # signals always prioritized
            self.final_rank_list = (
                signal_list +
                no_signal_list[0:remaining_spots])

    def __init__(
        self, 
        id_index,
        interviews_per_position,  
        positions,
        position_lookups,
        random_rank_list_order = False):
        
        self.id = id_index
        self.n_positions = positions
        self.n_interviews = self.n_positions * interviews_per_position
        self.quartile = position_lookups['program_id_to_quartile'][id_index]
        self.decile = position_lookups['program_id_to_decile'][id_index]
        self.random_rank_list_order = random_rank_list_order
        self.received_signals = []
        self.received_no_signals = []
        self.reviewed_applications = 0
        self.final_rank_list = []
        self.tentative_matches = []


def _get_positions_per_program(constants):
    n_positions = constants['n_positions']
    n_programs = constants['n_programs']
    base = n_positions // n_programs
    remainder = n_positions - base*n_programs
    positions = [base] * n_programs
    if remainder > 0:
        programs_to_add_one = np.random.choice(
            n_programs, size=remainder, replace=False)
        for program in programs_to_add_one:
            positions[program] += 1
    return positions

def run_sim(sim_settings:dict) -> dict:
    '''
    Worker function for parallel execution, runs one full simulation given
    settings dictionary
    '''

    # hash it to the sim id which is always unique
    seed = abs(hash(str(sim_settings['sim_id']))) % (2**32)
    np.random.seed(seed)
    
    positions_per_program = _get_positions_per_program(sim_settings)
    position_lookups = helpers.get_lookups(sim_settings)
    
    programs = [
        Program(
            j, 
            sim_settings['interviews_per_pos_list'][j], 
            positions_per_program[j],
            position_lookups,
            random_rank_list_order=False # TODO: incorporate later
        ) for j in range(sim_settings['n_programs'])
    ]
    
    # create each applicant, also applies to programs
    applicants = [
        Applicant(
            i,
            programs,
            position_lookups,
            sim_settings['applications_list'][i],
            sim_settings['signals'],
            no_quartile=False, # TODO: incorporate later
            random_rank_list_order=False # TODO: incorporate later
        ) for i in range(sim_settings['n_applicants'])
    ]
    
    # review applications, offer interviews
    for program in programs:
        program.create_final_rank_list_and_count_reviews()
        # Update applicants with interview offers
        for app_id in program.final_rank_list:
            applicants[app_id].interviews_received.append(program.id)
            if program.id in applicants[app_id].signaled_programs:
                applicants[app_id].signaled_interviews.append(program.id)
            else:
                applicants[app_id].non_signaled_interviews.append(program.id)

    # now create final rank lists based on interviews
    for applicant in applicants:
        applicant.create_final_rank_list()

    # match
    applicants, programs = stable_match(applicants, programs)

    return calculate_results(applicants, programs, sim_settings)


def read_run_simulations(RESULTS_PATH: str, WIPE_DATA: bool) -> pd.DataFrame:
    if WIPE_DATA:
        print('Wiping analysis data.')
        if os.path.exists(RESULTS_PATH):
            os.remove(RESULTS_PATH)

    if not os.path.isfile(RESULTS_PATH):
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        return pd.DataFrame()
    else:
        return pd.read_csv(RESULTS_PATH)

if __name__ == "__main__":
    
    results_path = "results/probabilistic/results.csv"
    sim_results = read_run_simulations(
        RESULTS_PATH=results_path,
        WIPE_DATA=False)
    sims_run = set(sim_results['sim_id'].values) if not sim_results.empty else set()
    
    simulation_constants = pq.ParquetFile("constants/constants.parquet")
    # each simulation contains huge amounts of data (gamma dists) so 
    # read in 
    
    print(f"Starting probabilistic simulation on {CORES_TO_USE} cores.")

    BATCH_SIZE = 100

    # Process file in chunks
    for batch_idx, sim_batch in enumerate(simulation_constants.iter_batches(batch_size=BATCH_SIZE)):
        chunk_df = sim_batch.to_pandas()
        tasks = []

        for index, sim_constants in chunk_df.iterrows():
            sim_config = sim_constants.to_dict()
            if sim_config['sim_id'] in sims_run:
                continue
            tasks.append(sim_config)

        if not tasks:
            continue

        # Execute batch in parallel
        with ProcessPoolExecutor(max_workers=CORES_TO_USE) as executor:
            future_results = executor.map(run_sim, tasks)

            new_results = []
            for res in future_results:
                new_results.append(res)

            if new_results:
                batch_df = pd.DataFrame(new_results)

                if sim_results.empty:
                    sim_results = batch_df
                else:
                    sim_results = pd.concat(
                        [sim_results, batch_df], ignore_index=True)

                sims_run.update(batch_df['sim_id'].values)

        print(
            f"Batch {batch_idx + 1} processed. Total simulations: {len(sim_results)}")

        # Save interim results
        if (batch_idx + 1) % 5 == 0:
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            sim_results.to_csv(results_path, index=False)
            print(f"Interim save to {results_path}")

    # Final save
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    sim_results.to_csv(results_path, index=False)
    print("Simulation complete.")