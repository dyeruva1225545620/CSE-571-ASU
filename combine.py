import os
import pandas as pd

#Folder where automate.py results are stored
FOLDER_PATH = "local_results"

#Summary file name
FILE_NAME = "Grid: originalClassic_2 new_features train: 100 test: 20 alpha 1e-01 epsilon 5e-02.xlsx"

#Only use files which have the following strings in them
STRINGS_IN_NAME = ["train: 100", "test: 20", "grid: originalClassic_2", "alpha: 1e-01", "epsilon: 5e-02"]

sarsa_raw_data = []
q_raw_data = []
file_no = 0

for file in os.listdir(FOLDER_PATH):
    file_path = os.path.join(FOLDER_PATH, file)
    if (os.path.isfile(file_path) and all([_ in file for _ in STRINGS_IN_NAME])):
        sarsa_raw = pd.read_excel(file_path, sheet_name="SARSA Raw")
        q_raw = pd.read_excel(file_path, sheet_name="Q_Learning Raw")

        sarsa_raw_data.append(sarsa_raw)
        q_raw_data.append(q_raw)

sarsa_raw = pd.concat(sarsa_raw_data, axis=1)
q_raw = pd.concat(q_raw_data, axis=1)

score_col_names = []
victory_col_names = []
for col_name in sarsa_raw.columns:
    if "scores" in col_name:
        score_col_names.append(col_name)
    elif "victories" in col_name:
        victory_col_names.append(col_name)
    else:
        assert False, f"Unknown column {col_name}"

raw_dfs = {"s": sarsa_raw, "q": q_raw}
summary_dfs = {"ss": pd.DataFrame(), "qq": pd.DataFrame()}
for algo in raw_dfs:
    df = raw_dfs[algo]
    dff = summary_dfs[f"{algo}{algo}"]
    dff[f"{algo}_score_avg"] = df[score_col_names].mean(axis=1)
    dff[f"{algo}_score_avg_cumu"] = dff[f"{algo}_score_avg"].cumsum()
    dff[f"{algo}_vict_avg"] = df[victory_col_names].mean(axis=1)
    dff[f"{algo}_vict_avg_cumu"] = dff[f"{algo}_vict_avg"].cumsum()

sarsa_raw = summary_dfs["ss"]
q_raw = summary_dfs["qq"]
summary_df = pd.concat((sarsa_raw, q_raw), axis=1)

excel_writer = pd.ExcelWriter(FILE_NAME)
sarsa_raw.to_excel(excel_writer, "SARSA", index=False)
q_raw.to_excel(excel_writer, "QLearn", index=False)
summary_df.to_excel(excel_writer, "summary", index=False)
excel_writer.close()
