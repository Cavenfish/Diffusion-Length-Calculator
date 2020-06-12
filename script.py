import os
import numpy as np
import pandas as pd
import openpyxl as op
import compile_data as cd
from openpyxl import load_workbook

#Define Data Directory
dDir = './test-data/'
rDir = './results-and-plots/'

#Initialize Excel Writer
writer = pd.ExcelWriter(rDir+'Test_Sample.xlsx', engine='openpyxl')

#Read in sp and fp DataFrame
pdf = pd.read_csv('./sp-fp.csv')

for file in os.listdir(dDir):
    fname = dDir + file
    sp    = pdf[pdf['File Name'] == file.strip('.txt')]['sp'].values[0]
    fp    = pdf[pdf['File Name'] == file.strip('.txt')]['fp'].values[0]
    r     = np.arange(0,1,0.1)

    #Initialize DataFrame
    df    = pd.DataFrame(index=r+sp, columns=r+fp)

    #Make and Save plot at initial points
    try:
        cd.fit_data(fname, sp+0.5, fp+0.5, save_name=rDir+file.replace('.txt', '.png'))
    except:
        print('Yo, dude this shit dont want to converge')
        print(file)

    #Fill in DataFrame with Diffusion Lengths
    for i in r:
        for j in r:
            try:
                dl = cd.fit_data(fname, sp+i, fp+j, save_name=False)
            except:
                dl = np.NAN
            df.loc[sp+i, fp+j] = dl

    #Make Excel File
    df.to_excel(writer, sheet_name=file.strip('.txt'))
    img = op.drawing.image.Image(rDir+file.replace('.txt', '.png'))
    ws  = writer.sheets[file.strip('.txt')]
    ws.add_image(img, 'A13')

writer.save()
