import os
import matplotlib.pyplot as plt
import pandas as pd
path = "/Users/praba/Desktop/Audio-Class/logs"
log_csvs = sorted(os.listdir(path))
print(log_csvs)
labels = ['Conv 1D', 'Conv 2D']
colors = ['r', 'm', 'c']
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16,5))

for i, (fn, label, c) in enumerate(zip(log_csvs, labels, colors)):
    csv_path = os.path.join('/Users/praba/Desktop/Audio-Class', 'logs', fn)
    df = pd.read_csv(csv_path)
    ax[i].set_title(label, size=16)
    ax[i].plot(df.accuracy, color=c, label='train')
    ax[i].plot(df.val_accuracy, ls='--', color=c, label='test')
    ax[i].legend(loc='upper left')
    ax[i].tick_params(axis='both', which='major', labelsize=12)
    ax[i].set_ylim([0,1.0])

fig.text(0.5, 0.02, 'Epochs', ha='center', size=14)
fig.text(0.08, 0.5, 'Accuracy', va='center', rotation='vertical', size=14)
plt.show()
