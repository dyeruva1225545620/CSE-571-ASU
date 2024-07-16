import pandas as pd
import scipy.stats as stats

#Path to excel file containing convergence speed of SARSA and Q Learning
df=pd.read_excel('final iter.xlsx')
df.dropna(inplace=True)
print(stats.ttest_ind(df['SARSA'],df['Q']))
