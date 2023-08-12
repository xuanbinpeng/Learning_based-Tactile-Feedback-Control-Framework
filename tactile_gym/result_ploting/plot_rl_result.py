import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

data_path = './result_push/'
rewards_all = []
for filename in os.listdir(data_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_path, filename)
        with open(file_path, 'r') as file:
            if len(rewards_all) == 0:
                rewards_all = np.transpose(np.array(pd.read_csv(file, usecols=['Value'])))
            else:
                new_rewards = np.transpose(np.array(pd.read_csv(file, usecols=['Value'])))
                rewards_all = np.vstack([rewards_all,new_rewards])
#print(rewards_all)
df = pd.DataFrame(rewards_all).melt(var_name='episode',value_name='reward')

g = sns.lineplot(x="episode", y="reward", data=df)
xlabels = ['{:.0f}'.format(x) +'K' for x in g.get_xticks()*10]

ylabels = ['reward']
g.set_xticklabels(xlabels)
g.set_xticklabels(ylabels)
plt.show()

