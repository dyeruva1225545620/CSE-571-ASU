import os
import sys
import json
import time
import datetime
import threading
import subprocess
import numpy as np
import pandas as pd
import multiprocessing as mp

# Determines the no of processes to be launched for each of
#  TrueOnlineSARSA and ApproximateQLearning Agent. Recommended value is 1 when run locally.
ITER_COUNT = 1

# The value to be passed to -x in pacman.py
TRAINING_COUNT = 100

# TRAINING_COUNT + TESTING_COUNT is passed to -n in pacman.py
TESTING_COUNT = 20

# Value of ALPHA that the agents should use, this is passed to pacman.py via -agentArgs
ALPHA = 0.1
EPSILON = 0.05

# Layout to use, passed via -l to pacman.py
GRID = "originalClassic_2"

# Ghost to use, passed via -g to pacman.py, this can take the value RandomGhost, DirectionalGhost, HeterogenousGhost
GHOST_TYPE = "HeterogenousGhost"
WINDOW_SIZE = 20

# Folder to store results locally
LOCAL_FOLDER = "local_results"

# Whether to log Weights and Q values in the resulting excel files, recommended value is False
# This was used initially to plot Weights of features over iterations. Enabling this
# will result in huge amounts of data being logged and is not recommended.
LOG_WEIGHTS_AND_Q = False

# Whether to run the SARSA and Q Learning Agent respectively. Enable both of them.
SARSA_AGENT = True
Q_LEARNING_AGENT = True

status_update_lock = threading.Lock()

def find_q_scores(proc_stdout):
    start_idx = 0
    q_scores = []
    while(proc_stdout.find("Q_value:", start_idx) != -1):
        q_score_start_idx = proc_stdout.find("Q_value:", start_idx)
        q_score_end_idx = proc_stdout.find("\n", q_score_start_idx)
        q_score = float(proc_stdout[q_score_start_idx: q_score_end_idx].split(":")[-1].strip(" \n"))
        start_idx = q_score_end_idx
        q_scores.append(q_score)
    return q_scores

def find_game_scores(proc_stdout):
    start_idx = 0
    game_scores = []
    while(proc_stdout.find("Game Score:", start_idx) != -1):
        game_score_start_idx = proc_stdout.find("Game Score:", start_idx)
        game_score_end_idx = proc_stdout.find("\n", game_score_start_idx)
        game_score = float(proc_stdout[game_score_start_idx: game_score_end_idx].split(":")[-1].strip(" \n"))
        start_idx = game_score_end_idx
        game_scores.append(game_score)
    return game_scores

def find_testing_victories(proc_stdout):
    start_idx = 0
    game_victories = []
    while(proc_stdout.find("Pacman", start_idx) != -1):
        game_score_start_idx = proc_stdout.find("Pacman", start_idx)
        game_score_end_idx = proc_stdout.find("\n", game_score_start_idx)
        result_line = proc_stdout[game_score_start_idx: game_score_end_idx]
        if ("died" in result_line):
            game_victories.append(False)
        elif ("victorious" in result_line):
            game_victories.append(True)
        else:
            assert False, f"Unknown result {result_line}"
        start_idx = game_score_end_idx
    return game_victories

def find_weights(proc_stdout):
    start_idx = 0
    weights = {"bias": [], "#-of-ghosts-1-step-away": [], "eats-food": [], "closest-food": []}
    while(proc_stdout.find("Print Weights:", start_idx) != -1):
        weights_start = proc_stdout.find("Print Weights:", start_idx)
        weights_end = proc_stdout.find("\n", weights_start)
        weights_in_iter = json.loads(proc_stdout[weights_start + len("Print Weights:") + 1: weights_end])
        for feature_name in weights_in_iter:
            weights[feature_name].append(weights_in_iter[feature_name])
        start_idx = weights_end
    return weights


def find_win_rate(proc_stdout):
    win_start_idx = proc_stdout.find("Win Rate:")
    win_end_idx = proc_stdout.find("\n", win_start_idx)
    win_score = int(proc_stdout[win_start_idx: win_end_idx].split(" ")[-2].split("/")[0])
    return win_score

def init_shared_value(val, ):
    global tasks_tracker
    tasks_tracker = val

def run_agent(agent_name):
    global tasks_tracker
    start_time = time.time()

    cmd = ["python3", "pacman.py", "-p", agent_name, "-q", "-x", str(TRAINING_COUNT),
           "-n", str(TRAINING_COUNT+TESTING_COUNT), "-l", GRID, "-g", GHOST_TYPE,
            "--agentArgs", f"alpha={ALPHA},epsilon={EPSILON}", "-k", "1000",
            "--timeout", "2"]
    print(f"starting {cmd}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE)
    print(f"done {cmd}")
    stdout = proc.stdout.decode()
    game_scores = find_game_scores(stdout)
    game_victories = find_testing_victories(stdout)
    win_rate = find_win_rate(stdout)
    if LOG_WEIGHTS_AND_Q:
        weights = find_weights(stdout)
        q_scores = find_q_scores(stdout)
    else:
        weights = None
        q_scores = None

    end_time = time.time()
    #with tasks_tracker.get_lock():
    #    print(f"Task done in {end_time - start_time}")
    #    tasks_tracker.value += 1
    #    if (tasks_tracker.value % 10 == 0):
    #        print(f"{tasks_tracker.value} tasks completed")
    return win_rate, game_scores, game_victories, weights, q_scores

if __name__ == "__main__":
    if (len(sys.argv) == 1):
        job_id = 1
    else:
        job_id = sys.argv[1]

    os.makedirs(LOCAL_FOLDER, exist_ok=True)
    timestamp = int(time.time())
    alpha_st = "{:.0e}".format(ALPHA)
    epsilon_st = "{:.0e}".format(EPSILON)
    file_name = f"{LOCAL_FOLDER}/iter: {ITER_COUNT} grid: {GRID} train: {TRAINING_COUNT} test: {TESTING_COUNT} ghost: {GHOST_TYPE} alpha: {alpha_st} epsilon: {epsilon_st} job_id: {job_id} time: {timestamp}.xlsx"
    print(file_name)
    print(f"Iter: {ITER_COUNT}, Grid: {GRID}, Ghost: {GHOST_TYPE}, Train: {TRAINING_COUNT}, Test: {TESTING_COUNT}, Alpha: {alpha_st}, Epsilon: {epsilon_st}")

    agents_to_run = []
    if (SARSA_AGENT):
        agents_to_run += ["TrueOnlineSARSA" for _ in range(0, ITER_COUNT)]

    if (Q_LEARNING_AGENT):
        agents_to_run += ["ApproximateQAgent" for _ in range(0, ITER_COUNT)]

    tasks_tracker = mp.Value('i', 0)
    with mp.Pool(initializer=init_shared_value, initargs=(tasks_tracker,)) as pool:
        agent_scores = pool.map(run_agent, agents_to_run)

    if (os.path.exists(file_name)):
        os.remove(file_name)
    excel_writer = pd.ExcelWriter(file_name)
    MAX_ROWS_IN_SHEET = 1048000

    if (SARSA_AGENT):
        sarsa_mov_avg_data = {}
        sarsa_raw_data = {}
        for idx, (win_rate, game_scores, game_victories, weights, q_scores) in enumerate(agent_scores[0:ITER_COUNT]):
            game_scores = np.array(game_scores)
            game_victories = np.array([False for _ in range(0, TRAINING_COUNT)] + game_victories)

            sarsa_mov_avg_data[f"game_scores_mov_avg_{idx}"] = np.convolve(game_scores, np.ones(WINDOW_SIZE), 'valid')/WINDOW_SIZE
            sarsa_raw_data[f"game_scores_{idx}"] = game_scores
            sarsa_raw_data[f"test_game_victories_{idx}"] = game_victories

            if (LOG_WEIGHTS_AND_Q):
                weights_df = pd.DataFrame(weights)
                q_scores_df = pd.DataFrame({"q_scores": np.cumsum(np.array(q_scores))})
                if (len(weights_df.index) > MAX_ROWS_IN_SHEET):
                    no_of_rows_to_delete = len(weights_df.index) - MAX_ROWS_IN_SHEET
                    weights_df.drop(weights_df.sample(n=no_of_rows_to_delete).index, inplace=True)
                weights_df.to_excel(excel_writer, f"SARSA weights {idx}", index=False)
                q_scores_df.to_excel(excel_writer, f"S Q_Scores {idx}", index=False)

        sorted_col_names = list(sarsa_raw_data.keys())
        sorted_col_names.sort()
        pd.DataFrame(sarsa_mov_avg_data).to_excel(excel_writer, "SARSA Mov Avg,", index=False)
        pd.DataFrame(sarsa_raw_data).to_excel(excel_writer, "SARSA Raw", index=False, columns=sorted_col_names)

    if (Q_LEARNING_AGENT):
        q_learning_mov_avg_data = {}
        q_learning_raw_data = {}
        for idx, (win_rate, game_scores, game_victories, weights, q_scores) in enumerate(agent_scores[ITER_COUNT: ]):
            game_scores = np.array(game_scores)
            game_victories = np.array([False for _ in range(0, TRAINING_COUNT)] + game_victories)

            q_learning_mov_avg_data[f"game_scores_mov_avg_{idx}"] = np.convolve(game_scores, np.ones(WINDOW_SIZE), 'valid')/WINDOW_SIZE
            q_learning_raw_data[f"game_scores_{idx}"] = game_scores
            q_learning_raw_data[f"test_game_victories_{idx}"] = game_victories

            if (LOG_WEIGHTS_AND_Q):
                weights_df = pd.DataFrame(weights)
                q_scores_df = pd.DataFrame({"q_scores": np.cumsum(np.array(q_scores))})
                if (len(weights_df.index) > MAX_ROWS_IN_SHEET):
                    no_of_rows_to_delete = len(weights_df.index) - MAX_ROWS_IN_SHEET
                    weights_df.drop(weights_df.sample(n=no_of_rows_to_delete).index, inplace=True)
                weights_df.to_excel(excel_writer, f"QLearn weights {idx}", index=False)
                q_scores_df.to_excel(excel_writer, f"QL Q_Scores {idx}", index=False)

        sorted_col_names = list(q_learning_raw_data.keys())
        sorted_col_names.sort()
        pd.DataFrame(q_learning_mov_avg_data).to_excel(excel_writer, "Q_Learning Mov Avg", index=False)
        pd.DataFrame(q_learning_raw_data).to_excel(excel_writer, "Q_Learning Raw", index=False, columns=sorted_col_names)

    excel_writer.close()
