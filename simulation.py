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
'''

import numpy as np
import pandas as pd

DEBUG = True
constants = pd.read_csv("constants.csv") if not DEBUG else pd.read_csv(
    "constants_debug_profile.csv")

constants = constants.drop(columns=['Description'])
CONSTANTS = constants.set_index(
    'Variable')['Value (Must be INTEGERS)'].astype(int).to_dict()

def print_constants():
    print("In this simulation, we will be running:\n")
    print(f"Residency programs: {CONSTANTS['n_programs']}")
    print(f"Residency applicants: {CONSTANTS['n_applicants']}")
    print(f"Spots per program: {CONSTANTS['spots_per_program']}, Interviews per spot: {CONSTANTS['interviews_per_spot']}")
    print(f"Maximum applications per applicant: {CONSTANTS['max_applications']}")
    print(f"Simulations per signal value: {CONSTANTS['simulations_per_s']}")
    print(f"Studying signal values from {CONSTANTS['study_min_signal']} to {CONSTANTS['study_max_signal']}\n")

class Applicant:
    n_applications = CONSTANTS['max_applications']
    n_applicants = CONSTANTS['n_applicants']
    n_signals = int(-1) # placeholder
    n_non_signals = int(-1) # placeholder
    
    # class method to update signal numbers
    @classmethod
    def update_signal_number(cls, signal_number):
        cls.n_signals = signal_number
        cls.n_non_signals = cls.n_applications - signal_number
    
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
        
    def pick_programs(self, 
                      all_programs: list, 
                      program_quartile_list: dict,
                      signals: bool):
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
    '''
        applications = []
        length = Applicant.n_signals if signals else Applicant.n_non_signals
        divisions_of_4 = length//4 # divisions of 4 go to quartiles
        remainder = length % 4 # all remainder go to same quartile
        own_quartile = self.quartile
        quartile_above = (self.quartile - 1) if self.quartile > 1 else self.quartile
        quartile_below = (self.quartile + 1) if self.quartile < 4 else self.quartile
        
        def _add_to_application_list(num, quartile):
            for _ in range(num):
                program_choices = program_quartile_list[quartile]
                attempts = 0
                while True:
                    attempts += 1
                    if attempts > 1000:
                        raise RuntimeError(
                            f"Applicant {self.id}: Stuck in loop trying to pick from quartile {quartile}. "
                            f"Already picked {len(applications)} programs: {applications}. " f"Need {num} from this quartile. Available: {len(program_choices)}")
                    # I suppose this could break if you run out of programs but
                    # that would mean > 80 applications per applicant in our scenario
                    # will write it as a condition in docs
                    choice = np.random.choice(program_choices) # picks program index/id
                    # make sure that they don't double-choose a program
                    # that they have already picked, either in 
                    # signals or non-signals
                    if (choice in applications) or (
                        choice in self.signaled_programs) or (
                        choice in self.non_signaled_programs
                        ):
                        # pick again, program already chosen
                        continue
                    else:
                        applications.append(choice) # just need the index
                        # add applicant to program's received signals/no signals
                        if signals:
                            all_programs[choice].received_signals.append(self.id) 
                        else:
                            all_programs[choice].received_no_signals.append(self.id)
                        break
        
        _add_to_application_list(divisions_of_4, quartile_above)
        _add_to_application_list(divisions_of_4, quartile_below)
        _add_to_application_list(2*divisions_of_4 + remainder, own_quartile)
        
        if len(applications) != length:
            print(f"Warning: Expected {length} applications but got {len(applications)} for applicant {self.id}")
    
        return applications
    
    def __init__(self, 
                 id_index: int, 
                 programs: list, 
                 program_quartile_list: dict):
        self.id = id_index
        self.quartile = self.get_quartile()
        self.matched_program = None
        self.signaled_programs = []
        self.non_signaled_programs = []
        self.signaled_programs = self.pick_programs(
            programs, program_quartile_list, True
        )
        self.non_signaled_programs = self.pick_programs(
            programs, program_quartile_list, False
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
            # TODO: can relax this assumption of having to review everybody
            # if you don't fill with signals with what Bruce and I chatted
            # about regarding some greedy parallel optimization
            self.reviewed_applications += len(self.received_no_signals)
            # final rank list always prioritizes signals
            self.final_rank_list = (
                sorted_signals + 
                sorted(self.received_no_signals)[0:remaining_spots])
        if len(self.final_rank_list) != self.num_interviews:
                    print(f"Warning: Program {self.id} has {len(self.final_rank_list)} in rank list but expected {self.num_interviews}")
                    
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
    '''
    Implements the applicant-proposing deferred acceptance algorithm (NRMP style).
    
    Applicants propose to programs in order of their final_rank_list.
    Programs tentatively hold the best applicants (according to their final_rank_list)
    up to their capacity, rejecting others.
    
    The algorithm terminates when no applicant has a program left to propose to.
    '''
    # Initialize tracking structures
    free_applicants = set(range(len(applicants)))
    # Track next program to propose to for each applicant
    # starting index is 0, checks first program on applicant final rank list
    # maximum index is len(final_rank_list) - 1 which = (max applications -1)
    next_proposal_index = {i: 0 for i in range(len(applicants))}

    while free_applicants:
        # Get an applicant who hasn't been matched and has programs left to propose to
        applicant_id = None
        for app_id in free_applicants:
            if next_proposal_index[app_id] < len(applicants[app_id].final_rank_list):
                applicant_id = app_id
                break

        if applicant_id is None:
            # No more proposals possible
            # All free applicants have exhausted their rank lists, end function
            break

        applicant = applicants[applicant_id]

        # Get next program on applicant's rank list
        program_id = applicant.final_rank_list[next_proposal_index[applicant_id]]
        next_proposal_index[applicant_id] += 1
        program = programs[program_id]
        
        if applicant_id not in program.final_rank_list:
            # Program does not consider this applicant
            # don't need to do anything, applicant remains in free_applicants
            continue

        # Program considers the proposal
        program.tentative_matches.append(applicant_id)
        
        # return the INDEX position of APP ID in the PROGRAMS final rank list
        # and sort based on THAT key
        # remember program and applicant rank lists prioritize signals first
        # so it's not always in order of index
        program.tentative_matches.sort(
            key = lambda app_id: program.final_rank_list.index(app_id))
                
        # Keep only spots_per_program applicants
        if len(program.tentative_matches) > program.spots_per_program:
            rejected = program.tentative_matches[program.spots_per_program:]
            program.tentative_matches = program.tentative_matches[:program.spots_per_program]

            # Rejected applicants become free again
            for rejected_id in rejected:
                # remember it is a set so it doesn't matter if we add
                # the applicant back to it
                free_applicants.add(rejected_id)
            
            if applicant_id in program.tentative_matches:
                # remember that remove raises a key error if not in set
                # discard will not ... but I want a key error if 
                # this somehow removes someone that wasn't in it
                free_applicants.remove(applicant_id)
            
        else:
            # Applicant was accepted (tentatively)
            free_applicants.remove(applicant_id)

    # Finalize matches
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

if __name__ == "__main__":
    print_constants()
    
    signal_range = [str(i) for i in range(CONSTANTS['study_min_signal'], 
                                          CONSTANTS['study_max_signal'] + 1)]
    final_dataframe = pd.DataFrame(columns=['Parameter'] + signal_range)
    
    for i in range(CONSTANTS['simulations_per_s']):
        print(f"Starting simulation batch {i+1} of {CONSTANTS['simulations_per_s']}")
        unmatched_applicants = ['Unmatched_Applicants']
        unfilled_spots = ['Unfilled_Spots']
        reviews_per_program = ['Reviews_Per_Program']
        for signal_value in range(CONSTANTS['study_min_signal'], 
                                  CONSTANTS['study_max_signal'] + 1):
            Applicant.update_signal_number(signal_value)
            app, spot, review = run_simulation(signal_value)
            unmatched_applicants.append(app)
            unfilled_spots.append(spot)
            reviews_per_program.append(review)
        # Append results to final dataframe
        final_dataframe.loc[len(final_dataframe)] = unmatched_applicants
        final_dataframe.loc[len(final_dataframe)] = unfilled_spots
        final_dataframe.loc[len(final_dataframe)] = reviews_per_program
    
    # Export combined results
    final_dataframe.to_csv('simulation_results.csv', index=False)