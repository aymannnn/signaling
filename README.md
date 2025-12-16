# Signaling Simulation Documentation

## Quickstart

1. Install requirements with: 

```bash
pip install -r requirements.txt
```

2. Set your constants in constants.csv. Make sure to set the result file prefix. Results will be generated in the folder: simulation_results/
3. Run:
    
```python
python simulation.py
```

4. To graph, first make sure you don't modify constants.csv, as the graphing functions pull the file path defined by the file prefix in constants.csv.  Simply run the following to generate graphs, will will have the same prefix as defined above:

```python
python data_analysis.py
```

## Repository Structure

1. constants_settings/: simple folder to save your constant settings if you'd like. The base case settings are stored here
2. simulation_results/: the simulation and graphs will default to this location. all other folders are just for study results of specific subsets.
3. profiling.txt: instructions on how to profile the code/what commands to run
4. constants.csv: constants for your simulation run, will always read from here
5. simulation.py/data_analysis.py: simulation and graphing of results, respectively

## Simulation

The entire simulation is contained in simulation.py. Here we will list out key assumptions and how the simulator works.

Assumptions: 

1. We do not consider paired applicants
2. We limit the total number of programs applicants may apply to
3. Assume that the list of applicants and programs is RANKED, i.e. 
if applicants is from 1, 2, …. N and programs from 1, 2, …. J, 
then 1 is the best applicant/program and N/J are the worst
4. Assume that the DISTANCE between this ordered list is equally distributed, 
i.e. the difference between 1 and 2 is the same as 29 and 30. Note that assumption this doesn't make a difference in this methodology.
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

Simulator Mechanics:

The simulator functions with the following simplified flow:

1. Have every applicant choose programs to apply to. This occurs with a 50/25/25 quartile split to fill up the number of signals and number of non-signals. 
2. Programs now offer interviews to applicants. First, they review all signaled applications, order them, and select the best ranked applicants until the interview spots are filled. If they are unable to fill the interviews with signals alone, then:
   1. Then proceed to review non-signaled applications. Here, we review all, which is heavily penalizing. Then, rank the non-signaled applications and add to interviews. Note: a non-signaled will never bump a signal.
3. Perform the stable matching algorithm. Applications and programs both rank in ascending order. The algorithm is applicant-proposing, like the NRMP, and terminates when no applicant has a program left to apply to.
4. Count the number of total reviewed applications by programs, the number of unmatched applicants, and number of unfilled program spots.

Repeat for every simulation in "simulations_per_s", and repeat for every s. Total simulations are (study_max_signal-study_min_signal) * simulations_per_s.

## Base Case Scenario

The base case scenario is modeled of the general surgery 2025 match. 367 programs, 3300 applicants, 5 spots per program (estimate), 12 interviews per spot, 1000 simulations per signal estimate, 40 maximum applications, signals between 5 and 20. 

In the base case, we keep the completely_randomize_program_pick flag off - if on, then it does not do a 50/25/25 quartile split and instead applicants randomly pick programs to apply to.

## Sensitivity Analyses

1. First we randomzied every program that applicants applied to without the 50/25/25 split.
2. Next, we tested a cap on 10 programs, 20 programs, 30 programs, and 50 programs and found the optimal signal number for each simulation. 

