'''
Residency signaling project ||

Key assumptions:
1. We do not consider paired applicants
2. We limit the total number of programs applicants may apply to
3. Assume that the list of applicants and programs is RANKED, i.e. 
if applicants is from 1, 2, …. N and programs from 1, 2, …. J, 
then 1 is the best applicant/program and N/J are the worst
4. Assume that the DISTANCE between this ordered list is equally distributed, 
i.e. the difference between 1 and 2 is the same as 29 and 30
5. Assume number of positions per program is uniform AND the number of
interviews per program is fixed.
6. Assume that programs and applicants will always rank their choices by the 
rank of the other, i.e. if i>j, then programs will rank applicant i > applicant j 
and applicants will rank program i > j
6a. Note: signals priority over non-signals, regardless of rank
7. Assume applicants will apply to programs in a 50/25/25 quartile split: in other
words, 50% of applications go to programs in their own quartile, 25% to the quartile above
(if applicable), and 25% to the quartile below (if applicable). If cannot
apply above or below OR if there is a remainder from division by 4, those applications
go to their own quartile.

## NOTE: Difference in this file is that it works by having applicants apply
# via a gamma distribution to simulate real-world behavior better.
'''

import numpy as np
import pandas as pd
import os
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import gamma

# variables you can change
SHOW_GAMMA_DISTRIBUTION = True

# if you wanted could update path here but don't recommend it
constants = pd.read_csv("constants_gamma.csv")
constants = constants.drop(columns=['Description'])
constants = constants.drop(columns=['Type'])
result_print_prefix = constants.loc[
    constants['Variable'] == 'result_file_prefix', 'Value'].values[0]
RANDOMIZE_PICK = constants.loc[
    constants['Variable'] == 'completely_randomize_program_pick', 'Value'].values[0
]
RANDOMIZE_PICK = False if RANDOMIZE_PICK.lower().strip() == 'false' else True
drop_constants = [
    'result_file_prefix',
    'completely_randomize_program_pick'
]
constants = constants[~constants['Variable'].isin(drop_constants)]
CONSTANTS = constants.set_index(
    'Variable')['Value'].astype(int).to_dict()

def print_constants():
    print("In this simulation, we will be running:\n")
    print(f"Residency programs: {CONSTANTS['n_programs']}")
    print(f"Residency applicants: {CONSTANTS['n_applicants']}")
    print(f"Spots per program: {CONSTANTS['spots_per_program']}, Interviews per spot: {CONSTANTS['interviews_per_spot']}")
    print(f"Simulations per signal value: {CONSTANTS['simulations_per_s']}")
    print(f"Studying signal values from {CONSTANTS['study_min_signal']} to {CONSTANTS['study_max_signal']}")
    print(f"Gamma distribution parameters: alpha = {CONSTANTS['gamma_alpha']}, beta = {CONSTANTS['gamma_beta']}")
    print(f"Completely randomize program pick: {RANDOMIZE_PICK}\n")
    

def _simulate_for_signal(signal_value: int, seed: int | None = None):
    """
    Pure worker - does NOT depend on globally mutable state.
    Single simulation run for a given signal_value.
    Did move applicant.update_signal_number inside so that each process
    is self-contained.
    
    It's actually real important to give them a random seed here because
    there can be a risk with parallelizaition that multiple processes
    get the same random seed and thus produce correlated results.
    """
    if seed is not None:
        np.random.seed(seed)

    Applicant.update_signal_number(signal_value)
    app, spot, review = run_simulation(signal_value)
    return app, spot, review

class Applicant:

    # add slots to prevent memory with __dict__ and many instances of
    # applicant and program
    __slots__ = (
        "id",
        "quartile",
        "matched_program",
        "signaled_programs",
        "non_signaled_programs",
        "final_rank_list",
        "n_applications",
        "n_signals",
        "n_non_signals"
    )

    n_applicants = CONSTANTS['n_applicants']
    class_n_signals = int(-1) # placeholder
    application_values = np.array(n_applicants * [0]) # placeholder
    
    # class method to update signal numbers
    @classmethod
    def update_signal_number(cls, signal_number):
        # also updates gamma distribution application values
        cls.class_n_signals = signal_number
        cls.application_values = gamma.rvs(
            a=CONSTANTS['gamma_alpha'],
            scale=CONSTANTS['gamma_beta'],
            size=cls.n_applicants)
    
    def get_quartile(self):
        quartile_size = self.n_applicants / 4
        if self.id < quartile_size:
            return 1
        elif self.id < quartile_size * 2:
            return 2
        elif self.id < quartile_size * 3:
            return 3
        else:
            return 4
    
    def _completely_random_pick(
        self, all_programs: list, length: int, signals: bool):
        applications = []
        already_chosen_programs = set(
            self.signaled_programs + self.non_signaled_programs)
        available_programs = [p.id for p in all_programs
                              if p.id not in already_chosen_programs]
        if length > len(available_programs):
            print(f"Warning: Applicant {self.id} requested {length} applications but only {len(available_programs)} available. This will only occur in extreme edge cases where n programs < n applications")
        choices = np.random.choice(
            available_programs, size=length, replace=False)
        for choice in choices:
            applications.append(choice)
            # add applicant to programs received applications
            if signals:
                all_programs[choice].received_signals.append(self.id)
            else:
                all_programs[choice].received_no_signals.append(self.id
        )
        return applications
        
    def pick_programs(self, 
                      all_programs: list, 
                      program_quartile_list: dict,
                      signals: bool,
                      completely_randomize: bool):
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
        if completely_randomize:
            return self._completely_random_pick(
                all_programs,
                Applicant.class_n_signals if signals else Applicant.n_non_signals,
                signals
            )
            
        applications = []
        length = self.n_signals if signals else self.n_non_signals
        if length == 0:
            # empty list if length is 0, rare but can happen in this scenario
            # since we are choosing application numbers from 
            # a gamma distribtuion and therefore there can be 0 non-signals
            return applications
        
        divisions_of_4 = length//4 # divisions of 4 go to quartiles
        remainder = length % 4 # all remainder go to same quartile
        own_quartile = self.quartile
        quartile_above = (self.quartile - 1) if self.quartile > 1 else self.quartile
        quartile_below = (self.quartile + 1) if self.quartile < 4 else self.quartile
        
        # this creates a list of the quartiles sorted by their closeness to the applicant's quartile so that we can later step into quartiles by distance
        quartiles_by_closeness = sorted(
            program_quartile_list.keys(),
            key=lambda q: abs(q - self.quartile))

    
        already_chosen_programs = set(
            self.signaled_programs + self.non_signaled_programs)
            
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
            print(f"Warning: Expected {length} applications but got {len(applications)} for applicant {self.id}")
    
        return applications
    
    def assign_signal_nonsignal_counts(self):
        raw = Applicant.application_values[self.id]
        self.n_applications = int(max(0, int(round(raw))))
        self.n_signals = min(self.n_applications, Applicant.class_n_signals)
        self.n_non_signals = self.n_applications - self.n_signals
    
    def __init__(self, 
                 id_index: int, 
                 programs: list, 
                 program_quartile_list: dict):
        self.id = id_index
        self.quartile = self.get_quartile()
        self.matched_program = None
        self.signaled_programs = []
        self.non_signaled_programs = []
        self.assign_signal_nonsignal_counts()
        self.signaled_programs = self.pick_programs(
            programs, program_quartile_list, True, RANDOMIZE_PICK
        )
        self.non_signaled_programs = self.pick_programs(
            programs, program_quartile_list, False, RANDOMIZE_PICK
        )
        self.signaled_programs.sort() # sort in ascending order
        self.non_signaled_programs.sort() # sort in ascending order
        # note that final rank list is 
        # signaled programs followed by non-signaled programs, just to give
        # a sense of "reality" with signals being prioritized
        self.final_rank_list = (
            self.signaled_programs + self.non_signaled_programs
        )

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
        "rank_index",    # added in stable_match
    )
    
    spots_per_program = CONSTANTS['spots_per_program']
    num_interviews = (
        CONSTANTS['interviews_per_spot']*CONSTANTS['spots_per_program'])
    def get_quartile(self):
        quartile_size = CONSTANTS['n_programs'] / 4
        if self.id < quartile_size:
            return 1
        elif self.id < quartile_size * 2:
            return 2
        elif self.id < quartile_size * 3:
            return 3
        else:
            return 4
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
                
        TODO: If you want a greedy parallel optimization of non-signal review
        then this is where to do it.
        '''
        # must review all signals
        self.reviewed_applications = len(self.received_signals)
        sorted_signals = sorted(self.received_signals)
        if len(self.received_signals) >= self.num_interviews:
            self.final_rank_list = sorted_signals[0:self.num_interviews]
        else:
            remaining_spots = self.num_interviews - len(self.received_signals)
            # TO nt DO: can relax this assumption of having to review everybody
            # if you don't fill with signals with what Bruce and I chatted
            # about regarding some greedy parallel optimization
            self.reviewed_applications += len(self.received_no_signals)
            # final rank list always prioritizes signals
            self.final_rank_list = (
                sorted_signals + 
                sorted(self.received_no_signals)[0:remaining_spots])
        # if len(self.final_rank_list) != self.num_interviews:
        #             print(f"Warning: Program {self.id} has {len(self.final_rank_list)} in rank list but expected {self.num_interviews}. Can be due to not enough applications and is not always abnormal for edge cases (low number of application cap).")
                    
    def __init__(self, id_index):
        self.id = id_index
        self.quartile = self.get_quartile()
        self.received_signals = []
        self.received_no_signals = []
        self.reviewed_applications = 0
        self.final_rank_list = [] # this is length of interviews_per_program
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
        program.tentative_matches.sort(
            key=lambda aid: program.rank_index[aid]
        )

        # Keep only best spots_per_program applicants if over capacity
        # program always prefers best applicants
        if len(program.tentative_matches) > program.spots_per_program:
            rejected = program.tentative_matches[program.spots_per_program:]
            program.tentative_matches = program.tentative_matches[:program.spots_per_program]

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

def count_unmatched(applicants: list, programs: list):
    '''
    Counts the number of unmatched applicants and unfilled program spots.
    
    Returns:
        unmatched_applicants: number of applicants who did not match to any program
        unfilled_spots: total number of empty spots across all programs
    '''
    unmatched_applicants = sum(
        1 for app in applicants if app.matched_program is None)
    unfilled_spots = 0
    
    for program in programs:
        filled_spots = len(program.tentative_matches)
        unfilled_spots += program.spots_per_program - filled_spots
    
    return unmatched_applicants, unfilled_spots

def run_simulation(s):
    program_quartile_list = get_quartile_dict(CONSTANTS['n_programs'])
    programs = [Program(j) for j in range(CONSTANTS['n_programs'])]
    applicants = [
        Applicant(i, programs, program_quartile_list) for i in range(
            CONSTANTS['n_applicants'])]
    for program in programs:
        program.create_final_rank_list_and_count_reviews()
    total_reviews = sum(
        [program.reviewed_applications for program in programs])
    total_reviews_per_program = total_reviews / CONSTANTS['n_programs']
    
    # last step is for the matching algorithm, which is stable matching
    applicants_matched, programs_matched = stable_match(applicants, programs)
    unmatched_applicants, unfilled_spots = count_unmatched(
        applicants_matched, programs_matched)
    
    return unmatched_applicants, unfilled_spots, total_reviews_per_program

def display_gamma_distribution():

    alpha = CONSTANTS['gamma_alpha']  # shape
    beta = CONSTANTS['gamma_beta']    # scale

    x = np.linspace(0, 200, 1000)
    y = gamma.pdf(x, a=alpha, scale=beta)

    plt.plot(x, y, label=f'Gamma Distribution (α={alpha}, β={beta})')
    plt.title('Gamma Distribution of Applications per Applicant')
    plt.xlabel('Number of Applications')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid()
    plt.savefig(f'simulation_results/gamma_distribution_{CONSTANTS["gamma_alpha"]}_{CONSTANTS["gamma_beta"]}.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    
    print_constants()

    if SHOW_GAMMA_DISTRIBUTION:
        display_gamma_distribution()
        
    signal_values = list(range(
        CONSTANTS['study_min_signal'],
        CONSTANTS['study_max_signal'] + 1
    ))
    
    signal_range = [str(v) for v in signal_values]

    final_dataframe = pd.DataFrame(columns=['Parameter'] + signal_range)

    print('Number of cores:', os.cpu_count())

    # -------- PARALLEL VERSION (normal fast run) --------
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
            # embarrassingly parallel in my research ... very rude name
            # pickles simulate for signal and sends payload to
            # idle worker process -> returns future object
            # each worker does the job and returns results
            futures = [
                executor.submit(_simulate_for_signal, s_val, int(seed))
                for s_val, seed in zip(signal_values, seeds)
            ]

            # Collect results (same order as signal_values)
            results = [f.result() for f in futures]

            # Rebuild rows in the same order as the original code
            unmatched_applicants = ['Unmatched_Applicants']
            unfilled_spots = ['Unfilled_Spots']
            reviews_per_program = ['Reviews_Per_Program']

            for (app, spot, review) in results:
                unmatched_applicants.append(app)
                unfilled_spots.append(spot)
                reviews_per_program.append(review)

            final_dataframe.loc[len(final_dataframe)
                                ] = unmatched_applicants
            final_dataframe.loc[len(final_dataframe)] = unfilled_spots
            final_dataframe.loc[len(final_dataframe)] = reviews_per_program

    prefix = 'simulation_results/' + result_print_prefix
    final_dataframe.to_csv(prefix +'_simulation_results.csv', index=False)