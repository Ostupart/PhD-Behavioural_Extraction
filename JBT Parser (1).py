#!/usr/bin/env python
# coding: utf-8

# # JBT Data Analysis
# 
# Extracts Judgement Bias Task (JBT) data from K-Limbic Software datafiles.
# 
# If you're running this then I assume you know what you're doing with Python and packages, etc..
# 
# Written by Peter Einarsson Nielsen (pe296) and edited by Olivia Stupart (osrps2)
# 

# In[8]:


from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import dateutil
from typing import List, Optional
from enum import Enum
import csv


class Side(Enum):
    left = 'L'
    right = 'R'


class DayTable(Enum):
    T3 = 'Training 3'
    T2a = 'Training 2    2 KHz - 8 KHz'
    T2b = 'Training 2    8 KHz - 2 KHz'


class ColumnIdx(Enum):
    outcome = 1
    tone = 2 #item index; 0: 2kHz, 1: 8kHz
    
    s4entry = 11  # timestamp: stimulus presented
    s4exit = 12  # timestamp: lever touched

    s4L1 = 13 # left lever pressed
    s4L2 = 14 #right lever pressed

    s7 = 18  # correct reward
    s8 = 19 #correct reward if dispense 2

    s10entry = 20  # timestamp: timeout
    s10exit = 21  # timestamp: timeout - always 10s
    s10L1 = 22 #premature left
    s10L2 = 23 # premature right

    s13entry = 29  # timestamp: ITI - always 5s
    s13exit = 30  # timestamp: ITI
    s13L1 = 31 # premature left
    s13L2 = 32 # premature right
    
    s14entry = 36 #timestamp: entry to premature time out 
    s14L1 = 38 # premature  left
    s14L2 = 39 # premature right
    

@dataclass
class ExperimentInfo():
    datetime: datetime
    subject_id: int
    box_id: int
    day_table: DayTable
    duration: int
    pellet_count: int


@dataclass
class TrialResult():
    #is_reversal: bool  # True if 'trial' is a reversal event. Otherwise False.
    choice_correct: Optional[bool] = None  # was their choice correct?
    chosen_side: Optional[Side] = None  # which side was chosen?
    correct_side: Optional[Side] = None  # which side was correct?
    #stuck_choice: Optional[bool] = None  # was their choice the same as for the last trial? (None for first run)
    reward_given: Optional[bool] = None  # were they given a reward?
    #reward_misleading: Optional[bool] = None  # was it a misleading reward?
    latency_choice: Optional[int] = None  # how long did it take to choose a side? (ms)
    #latency_collect: Optional[int] = None  # how long did it take to collect the reward?  (ms, None if no reward given)
    #latency_initiate: Optional[int] = None  # how long did it take to initiate the next trial? (ms, None if reward given)
    premature: Optional[int] = None #total number of prematures 
    missed: Optional[bool] = None # missed trials 

@dataclass
class ExperimentFindings():

    num_trials: int = 0
    #num_reversals: int = 0
    num_correct: int = 0
    #num_misleading_rewards: int = 0
    #num_misleading_loss: int = 0
    num_premature: Optional[int] = None
    num_left: int = 0
    num_right: int = 0
    num_left_correct: Optional[int] = None
    num_right_correct: Optional[int] = None
    num_missed: Optional[int] = None

    #num_trials_to_first_reversal: Optional[int] = None  # TODO
    #mean_perseverative_responses: Optional[float] = None  # TODO

    perc_correct: float = 0.0
    perc_left_correct: Optional[int] = None
    perc_right_correct: Optional[int] = None
    #perc_misleading_rewards: float = 0.0
    #perc_misleading_loss: float = 0.0
    perc_premature: Optional[int] = None

    mean_latency_choice: float = 0.0
    #mean_latency_collect: float = 0.0
    #mean_latency_initiate: float = 0.0

    # Only useful when Day Table is PRL_L/R
    #stwc: int = 0  # Stay on win on correct.
    #stwi: int = 0  # Stay on win on incorrect.
    #stlc: int = 0  # Stay on loss on correct.
    #stli: int = 0  # Stay on loss on incorrect.
    #shwc: int = 0  # Shift on win on correct.
    #shwi: int = 0  # Shift on win on incorrect.
    #shlc: int = 0  # Shift on loss on correct.
    #shli: int = 0  # Shift on loss on incorrect.
    

@dataclass
class Experiment():
    info: ExperimentInfo
    results: List[TrialResult]
    findings: ExperimentFindings

    def analyse(self):
        self.findings.num_trials = sum([1 for trial in self.results])
        self.findings.num_left = sum([1 for trial in self.results if trial.chosen_side == Side.left])
        self.findings.num_right = sum([1 for trial in self.results if trial.chosen_side == Side.right])
        self.findings.num_correct = sum([1 for trial in self.results if trial.choice_correct])
        self.findings.num_left_correct = sum([1 for trial in self.results if trial.choice_correct and trial.chosen_side == Side.left])
        self.findings.num_right_correct = sum([1 for trial in self.results if trial.choice_correct and trial.chosen_side == Side.right])
        #self.findings.num_misleading_rewards = sum([1 for trial in self.results if trial.reward_misleading])
        #self.findings.num_misleading_loss = sum([1 for trial in self.results if trial.choice_correct and trial.reward_given == False])
        self.findings.perc_correct = self.findings.num_correct / self.findings.num_trials
        self.findings.perc_left_correct = (self.findings.num_left_correct / self.findings.num_left) if self.findings.num_left > 0 else None
        self.findings.perc_right_correct = (self.findings.num_right_correct / self.findings.num_right) if self.findings.num_right > 0 else None
        #self.findings.perc_misleading_rewards = self.findings.num_misleading_rewards / self.findings.num_trials
        #self.findings.perc_misleading_loss = self.findings.num_misleading_loss / self.findings.num_trials
        self.findings.mean_latency_choice = sum([trial.latency_choice for trial in self.results if trial.latency_choice]) / self.findings.num_trials
        #self.findings.mean_latency_collect = sum([trial.latency_collect for trial in self.results if trial.latency_collect]) / self.findings.num_trials
        #self.findings.mean_latency_initiate = sum([trial.latency_initiate for trial in self.results if trial.latency_initiate]) / self.findings.num_trials
        self.findings.num_premature = sum([trial.premature for trial in self.results if trial.premature])
        self.findings.perc_premature = self.findings.num_premature / (self.findings.num_trials + self.findings.num_premature)
       
        self.findings.num_missed = sum([1 for trial in self.results if trial.missed])

        #self.findings.stwc = sum([1 for trial in self.results if trial.stuck_choice == True and trial.reward_given and trial.choice_correct])
        #self.findings.stwi = sum([1 for trial in self.results if trial.stuck_choice == True and trial.reward_given and trial.choice_correct == False])
        #self.findings.stlc = sum([1 for trial in self.results if trial.stuck_choice == True and trial.reward_given == False and trial.choice_correct])
        #self.findings.stli = sum([1 for trial in self.results if trial.stuck_choice == True and trial.reward_given == False and trial.choice_correct == False])
        #self.findings.shwc = sum([1 for trial in self.results if trial.stuck_choice == False and trial.reward_given and trial.choice_correct])
        #self.findings.shwi = sum([1 for trial in self.results if trial.stuck_choice == False and trial.reward_given and trial.choice_correct == False])
        #self.findings.shlc = sum([1 for trial in self.results if trial.stuck_choice == False and trial.reward_given == False and trial.choice_correct])
        #self.findings.shli = sum([1 for trial in self.results if trial.stuck_choice == False and trial.reward_given == False and trial.choice_correct == False])

        #self.findings.num_reversals = sum([1 for trial in self.results if trial.is_reversal])
        #self.findings.num_trials_to_first_reversal = next(i for i, trial in enumerate(self.results) if trial.is_reversal) if self.findings.num_reversals > 0 else None
        
        # Calculating the mean number of perseverative responses.
        #idx_of_reversals = [i for i, trial in enumerate(self.results) if trial.is_reversal]
        #num_persp_resps = []
        #for idx in idx_of_reversals:
         #   num_persp_resp = 0
          #  for trial in self.results[idx+1:]:
           #     if trial.choice_correct == False:
            #        num_persp_resp += 1
             #   else:
              #      break
           # if num_persp_resp == len(self.results[idx+1:]):
            #    continue
           # num_persp_resps.append(num_persp_resp)
        #self.findings.mean_perseverative_responses = sum(num_persp_resps) / len(num_persp_resps) if num_persp_resps else None
        

    def export_to_csv(self, file: Path):
        '''
        Output info and findings to csv file.
        If file exists: append to file.
        If file !exists: create file, write header, then write info and findings.
        '''

        if not file.is_file():
            # create file, add header
            header = [*vars(self.info), *vars(self.findings)]
            with open(file, 'w') as f:
                csv.writer(f).writerow(header)

        row = []
        for subobj in [self.info, self.findings]:
            for item in [*vars(subobj)]:
                if isinstance(getattr(subobj, item), DayTable):
                    row.append(f'{getattr(subobj, item).name}')
                    continue
                row.append(f'{getattr(subobj, item)}')

        with open(file, 'a') as f:
            csv.writer(f).writerow(row)
            




# In[9]:


def get_runs(datafile):
    '''Identify all STARTDATA, ENDDATA chunks in a datafile.'''
    all_data = []

    with open(datafile, 'r') as ro:
        reader = csv.reader(ro)
        for row in reader:
            all_data.append(row)

    start_indices = [i for i, row in enumerate(all_data) if 'STARTDATA' in row]
    end_indices = [i for i, row in enumerate(all_data) if 'ENDDATA' in row]

    return [
        all_data[i:j] for i,j in zip(start_indices, end_indices)
    ]


# In[10]:


def check_header_row(run: List, header: str):
    return [row[1] for row in run if row and row[0] == header][0]


def get_main_row_idx(run, search_term):
    return [i for i, row in enumerate(run) if search_term in row][0]


def get_run_info(run):
    header = run[1:get_main_row_idx(run, 'AC Comment')]

    # Extract pertinent header information
    return ExperimentInfo(
        datetime = dateutil.parser.parse(
            f"{check_header_row(header, 'Date')} {check_header_row(header, 'Time')}"
        ),
        subject_id = check_header_row(header, 'Subject Id'),
        box_id = check_header_row(header, 'Box Index'),
        day_table = DayTable(check_header_row(header, 'Day Table')),
        duration = check_header_row(header, 'Duration'),
        pellet_count = check_header_row(header, 'Pellet Count'),
    )


# In[11]:


# Extract pertinent trial information

def get_trials(run):
    # Identify just the trials section of the datafile and convert the data to integers.
    trials = run[get_main_row_idx(run, 'Stage (3)')+3:get_main_row_idx(run, 'ACTIVITYLOG')-1]
    trials = [[int(el) for el in trial if el] for trial in trials]
    # Remove 'test-is-ready' and incomplete trials.
    real_trials = [trial for trial in trials if trial[1] != 1000 and trial[1] != 128 and trial[1] != 150]
    # skip trials indicating end of run; trial[1]
    # == 1000 if ...
    # == 128 if run finishes prematurely
    # == 150 if run finishes (i.e. full 100 trials in run)
    return real_trials



def get_trial_info(trial: List, run_info: ExperimentInfo):
    #no reversals in JBT
    # If a 'trial' is a reversal event then there is nothing to analyse.
#     missed = True if trial[ColumnIdx.outcome.value] == 2 else False
    if trial[ColumnIdx.outcome.value] == 2:
        missed = True
    elif trial[ColumnIdx.outcome.value] == 150:
        missed = True
    else:
        missed = False 
    
    if missed:
        return TrialResult(
            missed = missed,
        )

    choice_correct = True if trial[ColumnIdx.outcome.value] == 0 else False
    

    if trial[ColumnIdx.s4L1.value] == 1 and trial[ColumnIdx.s4L2.value] == 0:
        chosen_side = Side.left
    elif trial[ColumnIdx.s4L1.value] == 0 and trial[ColumnIdx.s4L2.value] == 1:
        chosen_side = Side.right
    elif trial[ColumnIdx.s4L1.value] == 0 and trial[ColumnIdx.s4L2.value] == 1:
        missed = True
    #else:
     #   raise Exception('Miss Trial', trial)

    if choice_correct:
        correct_side = chosen_side
    elif not choice_correct and chosen_side == Side.left:
        correct_side = Side.right
    elif not choice_correct and chosen_side == Side.right:
        correct_side = Side.left

    #if prev_chosen_side:
     #   stuck_choice = True if chosen_side == prev_chosen_side else False
    #else:
     #   stuck_choice = None

    #if run_info.day_table not in [DayTable.T3, DayTable.PRL_R]:
     #   reward_given = True if choice_correct == True else False
      #  reward_misleading = False
    
    reward_given = True if trial[ColumnIdx.s7.value] == 1 or trial[ColumnIdx.s8.value] == 1 else False
     #   reward_misleading = True if trial[ColumnIdx.s14.value] == 1 else False 

    latency_choice = trial[ColumnIdx.s4exit.value] - trial[ColumnIdx.s4entry.value]

    #latency_collect = trial[ColumnIdx.s15exit.value] - trial[ColumnIdx.s15entry.value] if reward_given else None

    #latency_initiate = trial[ColumnIdx.s20entry.value] - trial[ColumnIdx.s19entry.value] if not reward_given else None
    
    premature = trial[ColumnIdx.s10L1.value] + trial[ColumnIdx.s10L2.value] + trial[ColumnIdx.s13L1.value] + trial[ColumnIdx.s13L2.value] + trial[ColumnIdx.s14L1.value] + trial[ColumnIdx.s14L2.value]

    return TrialResult(
        #is_reversal = is_reversal,
        choice_correct = choice_correct,
        chosen_side = chosen_side,
        correct_side = correct_side,
        #stuck_choice = stuck_choice,
        reward_given = reward_given,
        #reward_misleading = reward_misleading,
        latency_choice = latency_choice,
        #latency_collect = latency_collect,
        #latency_initiate = latency_initiate,
        premature = premature,
    )


# In[12]:


# Put it all together

def get_experiments(datafile) -> List[Experiment]:

    runs = get_runs(datafile)

    print(f'NUMBER OF RUNS IN {datafile}: {len(runs)}')

    experiments = []

    for run in runs:
        run_info = get_run_info(run)  # ExperimentInfo
        if run_info.day_table in [DayTable.T2a,DayTable.T2b]:
            print('Run has daytable Touch Training: ignoring.')
            continue
        
        # num_reversals, real_trials = get_trials(run)
        real_trials = get_trials(run)

        trial_results = []

        previous_choice = None
        for trial in real_trials:
            trial_info = get_trial_info(trial, run_info)
            #previous_choice = trial_info.chosen_side
            trial_results.append(trial_info)

        # Ignore a run if the number of trials is zero.
        # This is to account for a particular issue where the participant did not complete the trial.
        if len(trial_results) == 0:
            print(f'Ignoring run. No trials in run: {run_info}')
            continue

        experiments.append(
            Experiment(
                info = run_info,
                results = trial_results,
                # findings = ExperimentFindings(num_reversals=num_reversals)
                findings = ExperimentFindings()
            )
        )

    return experiments


# # Running the script
# Each datafile contains multiple experiments.
# 
# get_experiments(df) parses a datafile and returns list of experiment objects.
# 
# Each experiment can be analysed by running exp.analyse() where exp is an experiment object.

# In[14]:


FINDINGS_FILE = Path('./JBT/findings_JBT.csv')
DATAFOLDER = Path('./JBT/Animal_Data/')

datafiles = ([
    p for p in DATAFOLDER.iterdir() if p.is_file
    and p.suffix == '.csv' and 'Combined' not in p.name
])

#DATAFOLDERS = [
 #    Path('./MS_Cohort_1/Animal_Data/Male/Corticosterone/'),
 #   Path('./MS_Cohort_1/Animal_Data/Female/Corticosterone/'),
# ]


#datafiles1 = ([
#     p for p in DATAFOLDERS[0].iterdir() if p.is_file
#     and p.suffix == '.csv'# and 'Combined' not in p.name
# ])

#datafiles2 = ([
#     p for p in DATAFOLDERS[1].iterdir() if p.is_file
#     and p.suffix == '.csv'# and 'Combined' not in p.name
# ])

#datafiles = datafiles1 + datafiles2



print(len(datafiles))

experiments = [exp for df in datafiles for exp in get_experiments(df)]

print(f'TOTAL NUMBER OF EXPERIMENTS: {len(experiments)}')

for exp in experiments:
    try:
        exp.analyse()
    except ZeroDivisionError as err:
        raise Exception('Division by zero', exp.info)

    exp.export_to_csv(FINDINGS_FILE)


# In[ ]:




