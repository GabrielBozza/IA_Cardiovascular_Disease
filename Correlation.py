import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\g-boz\Desktop\Trabalho IA\datasets\cleveland.csv")

df.columns = ['idade', 'sexo', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

df['thal'] = df.thal.fillna(df.thal.mean())#SUBSITUI OS VALORES NULOS PELA MEDIA
df['ca'] = df.ca.fillna(df.ca.mean())
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

corr = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, len(df.columns), 1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)
plt.show()