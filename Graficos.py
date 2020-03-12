import pandas as pd
from warnings import simplefilter
import numpy as np
import matplotlib.pyplot as plt

simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv(r"C:\Users\g-boz\Desktop\Trabalho IA\datasets\cleveland.csv", header=None)

df.columns = ['idade', 'sexo', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(100, 100))
plt.subplots_adjust(wspace=0.20, hspace=0.60, top=0.955)
plt.suptitle("Dados Heart Disease", fontsize=15)

bins = np.linspace(20, 80, 10)
axes[0, 0].hist(df[df.target > 0].idade.tolist(), bins, color=["crimson"], histtype="step", label="Presença de cardiopatia", density=True)
axes[0, 0].hist(df[df.target == 0].idade, bins, color=["chartreuse"], histtype="step", label="Ausência de cardiopatia", density=True)
axes[0, 0].set_xlabel("Idade (anos)", fontsize=10)
axes[0, 0].set_ylim(0.0, 0.080)
axes[0, 0].legend(prop={'size': 8}, loc="upper left")

bins = np.arange(2)
width = 0.5
heights1 = df[df.target > 0]["sexo"].groupby(df["sexo"]).count()
heights2 = df[df.target == 0]["sexo"].groupby(df["sexo"]).count()
heights1 = heights1 / sum(heights1)
heights2 = heights2 / sum(heights2)
axes[0, 1].bar(bins + 0.025, heights1, width, align="center", edgecolor=["crimson"], color=["none"], label="Presença de cardiopatia")
axes[0, 1].bar(bins, heights2, width, align="center", edgecolor=["chartreuse"], color=["none"], label="Ausência de cardiopatia")
axes[0, 1].set_xlabel("Sexo", fontsize=10)
axes[0, 1].set_xticks(bins)
axes[0, 1].set_xticklabels(["Mulher", "Homem"], ha="center")

bins = np.arange(4)
width = 0.5
heights1 = df[df.target > 0]["cp"].groupby(df["cp"]).count()
heights2 = df[df.target == 0]["cp"].groupby(df["cp"]).count()
heights1 = heights1 / sum(heights1)
heights2 = heights2 / sum(heights2)
axes[0, 2].bar(bins + 0.025, heights1, width, align="center", edgecolor=["crimson"], color=["none"], label="Presença de cardiopatia")
axes[0, 2].bar(bins, heights2, width, align="center", edgecolor=["chartreuse"], color=["none"], label="Ausência de cardiopatia")
axes[0, 2].set_xlabel("Tipo de dor torácica", fontsize=10)
axes[0, 2].set_xticks(bins)
axes[0, 2].set_xticklabels(["típica", "atípica", "não ang.", "assintomática"], ha="right", rotation=45.,fontsize=7)

bins = np.linspace(80, 200, 15)
axes[0, 3].hist(df[df.target > 0].trestbps.tolist(), bins, color=["crimson"], histtype="step", label="Presença de cardiopatia", density=True)
axes[0, 3].hist(df[df.target == 0].trestbps, bins, color=["chartreuse"], histtype="step", label="Ausência de cardiopatia", density=True)
axes[0, 3].set_xlabel("Pressão arterial de repouso (mm Hg)", fontsize=10)

axes[1, 0].hist(df[df.target > 0].chol.tolist(), color=["crimson"], histtype="step", label="Presença de cardiopatia", density=True)
axes[1, 0].hist(df[df.target == 0].chol, color=["chartreuse"], histtype="step", label="Ausência de cardiopatia", density=True)
axes[1, 0].set_xlabel("Soro Colesterol (mg/dl)", fontsize=10)

bins = np.arange(2)
width = 0.5
heights1 = df[df.target > 0]["fbs"].groupby(df["fbs"]).count()
heights2 = df[df.target == 0]["fbs"].groupby(df["fbs"]).count()
heights1 = heights1 / sum(heights1)
heights2 = heights2 / sum(heights2)
axes[1, 1].bar(bins + 0.025, heights1, width, align="center", edgecolor=(0.917, 0.083, 0, 0.75), color=["none"], label="Presença de cardiopatia")
axes[1, 1].bar(bins, heights2, width, align="center", edgecolor=(0.467, 0.533, 0, 0.75), color=["none"], label="Ausência de cardiopatia")
axes[1, 1].set_xlabel("Açúcar no sangue em jejum", fontsize=10)
axes[1, 1].set_xticks(bins)
axes[1, 1].set_xticklabels(["< 120 mg/dl", "> 120 mg/dl"], ha="center")

bins = np.arange(3)
width = 0.5
heights1 = df[df.target > 0]["restecg"].groupby(df["restecg"]).count()
heights2 = df[df.target == 0]["restecg"].groupby(df["restecg"]).count()
heights1 = heights1 / sum(heights1)
heights2 = heights2 / sum(heights2)
axes[1, 2].bar(bins + 0.025, heights1, width, align="center", edgecolor=["crimson"], color=["none"], label="Presença de cardiopatia")
axes[1, 2].bar(bins, heights2, width, align="center", edgecolor=["chartreuse"], color=["none"], label="Ausência de cardiopatia")
axes[1, 2].set_xlabel("ECG em repouso", fontsize=10)
axes[1, 2].set_xticks(bins)
axes[1, 2].set_xticklabels(["Normal", "Anomalia ST-T ", "Hipert. ventric. esq."], ha="right", rotation=45.,fontsize=7)

axes[1, 3].hist(df[df.target > 0].thalach.tolist(), color=["crimson"], histtype="step", label="Presença de cardiopatia", density=True)
axes[1, 3].hist(df[df.target == 0].thalach, color=["chartreuse"], histtype="step", label="Ausência de cardiopatia", density=True)
axes[1, 3].set_xlabel("Max. freq. cardíaca", fontsize=10)

bins = np.arange(2)
width = 0.5
heights1 = df[df.target > 0]["exang"].groupby(df["exang"]).count()
heights2 = df[df.target == 0]["exang"].groupby(df["exang"]).count()
heights1 = heights1 / sum(heights1)
heights2 = heights2 / sum(heights2)
axes[2, 0].bar(bins + 0.025, heights1, width, align="center", edgecolor=["crimson"], color=["none"], label="Presença de cardiopatia")
axes[2, 0].bar(bins, heights2, width, align="center", edgecolor=["chartreuse"], color=["none"], label="Ausência de cardiopatia")
axes[2, 0].set_xlabel("Angina induzida por exercício", fontsize=10)
axes[2, 0].set_xticks(bins)
axes[2, 0].set_xticklabels(["Não", "Sim"], ha="center")

axes[2, 1].hist(df[df.target > 0].oldpeak.tolist(), color=["crimson"], histtype="step", label="Presença de cardiopatia", density=True)
axes[2, 1].hist(df[df.target == 0].oldpeak, color=["chartreuse"], histtype="step", label="Ausência de cardiopatia", density=True)
axes[2, 1].set_xlabel("Depressão ST Induzida por exercício", fontsize=10)

bins = np.arange(3)
width = 0.5
heights1 = df[df.target > 0]["slope"].groupby(df["slope"]).count()
heights2 = df[df.target == 0]["slope"].groupby(df["slope"]).count()
heights1 = heights1 / sum(heights1)
heights2 = heights2 / sum(heights2)
axes[2, 2].bar(bins + 0.025, heights1, width, align="center", edgecolor=["crimson"], color=["none"], label="Presença de cardiopatia")
axes[2, 2].bar(bins, heights2, width, align="center", edgecolor=["chartreuse"], color=["none"], label="Ausência de cardiopatia")
axes[2, 2].set_xlabel("Inclinação do segmento ST no pico do exercício", fontsize=10)
axes[2, 2].set_xticks(bins)
axes[2, 2].set_xticklabels(["Positiva", "Plana", "Negativa"], ha="right", rotation=45.,fontsize=7)

'''''
bins = np.arange(4)
width = 0.5
heights1 = df[df.target > 0]["ca"].groupby(df["ca"]).count()
heights2 = df[df.target == 0]["ca"].groupby(df["ca"]).count()
heights1 = heights1 / sum(heights1)
heights2 = heights2 / sum(heights2)
axes[2, 3].bar(bins + 0.025, heights1, width, align="center", edgecolor=["crimson"], color=["none"], label="Presença de cardiopatia")
axes[2, 3].bar(bins, heights2, width, align="center", edgecolor=["chartreuse"], color=["none"], label="Ausência de cardiopatia")
axes[2, 3].set_xlabel("Número de vasos coloridos por Fluoroscopia", fontsize=10)
axes[2, 3].set_xticks(bins)
axes[2, 3].set_xticklabels(["0", "1", "2", "3"], ha="center")
'''''

bins = np.arange(3)
width = 0.5
heights1 = df[df.target > 0]["thal"].groupby(df["thal"]).count()
heights2 = df[df.target == 0]["thal"].groupby(df["thal"]).count()
heights1 = heights1 / sum(heights1)
heights2 = heights2 / sum(heights2)
axes[2, 3].bar(bins + 0.025, heights1, width, align="center", edgecolor=["crimson"], color=["none"], label="Presença de cardiopatia")
axes[2, 3].bar(bins, heights2, width, align="center", edgecolor=["chartreuse"], color=["none"], label="Ausência de cardiopatia")
axes[2, 3].set_xlabel("Resultado do teste de Thallium", fontsize=10)
axes[2, 3].set_xticks(bins)
axes[2, 3].set_xticklabels(["Normal", "Defeito fixo", "Defeito reversível"], ha="right", rotation=45.,fontsize=7)
axes[2, 3].set_ylim(0.0, 1.0)


plt.show()