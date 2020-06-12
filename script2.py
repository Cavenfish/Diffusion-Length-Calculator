import os
import numpy as np
import pandas as pd
import compile_data as cd
import matplotlib.pyplot as plt
import statistics as st

#Define Data Directory
dDir = './06-08-2020/'
rDir = dDir+'June8/'

#Make plots dir
try:
    os.mkdir(rDir)
except:
    print('Folder exists')

#Read in sp and fp DataFrame
pdf = pd.read_csv(dDir+'sp-fp.csv')

#Initialize dictionary
data = {}
for file in pdf['File Name']:
    if file is np.nan:
        break
    var = file[0:len(file)-2]

    if var not in data:
        data[var] = []

for file in os.listdir(dDir):
    if '.txt' not in file:
        continue

    fname = dDir + file
    op    = pdf[pdf['File Name'] == file.strip('.txt')]['op'].values[0]
    sp    = pdf[pdf['File Name'] == file.strip('.txt')]['sp'].values[0]
    fp    = pdf[pdf['File Name'] == file.strip('.txt')]['fp'].values[0]
    tp    = pdf[pdf['File Name'] == file.strip('.txt')]['tp'].values[0]
    np    = pdf[pdf['File Name'] == file.strip('.txt')]['np'].values[0]

    print(file)
    dl    = cd.fit_data(fname, op, sp, fp, tp,
                        save_name=rDir+file.replace('.txt', '.png'),
                        neg=np)

    data[file[0:len(file)-6]].append(dl)

"""
tmps = [78.75, 123.15, 173.15, 223.15, 293.15]
dls  = [st.mean(n194),
        st.mean(n150),
        st.mean(n100),
        st.mean(n050),
        st.mean(rt)]
sig  = [st.stdev(n194),
        st.stdev(n150),
        st.stdev(n100),
        st.stdev(n050),
        st.stdev(rt)]


print('-194C: L = ' + str(st.mean(n194)) + '± ' + str(st.stdev(n194)) )
print('-150C: L = ' + str(st.mean(n150)) + '± ' + str(st.stdev(n150)) )
print('-100C: L = ' + str(st.mean(n100)) + '± ' + str(st.stdev(n100)) )
print('-50C: L = ' + str(st.mean(n050)) + '± ' + str(st.stdev(n050)) )
print('20C: L = ' + str(st.mean(rt)) + '± ' + str(st.stdev(rt)) )

cd.fit_LvT(dls, tmps, sig, save_name=rDir+'LvT.png')
"""
means = []
sigs  = []
times = []
for key in data:
    u = st.mean(data[key])
    s = st.stdev(data[key])
    t = int(key.split()[0].strip('s'))

    means.append(u)
    sigs.append(s)

    if '100uA' in key:
        t += 200
    if '150uA' in key:
        t += 250

    times.append(t)


plt.vlines(200, 0, 0.5, linestyles='--', color='red')
plt.vlines(250, 0, 0.5, linestyles='--', color='orange')
plt.errorbar(times, means, sigs, label='Data', fmt='o')
plt.title('Diffusion Length versus Injection Time')
plt.xlabel('Injection Time (s)')
plt.ylabel('Diffusion Length (μm)')
plt.show()
plt.close()
