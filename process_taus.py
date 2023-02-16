# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 22:31:04 2022

@author: Felipe Tapia
"""
import os
import numpy as np

def long_number(x):
    num = str(x)
    while len(num) < 3:
         num = '0' + num
    return num

fold = r'C:\...'
subdirs = next(os.walk(fold))[1]

overall = []

for subfold in subdirs:
    thisfold = os.path.join(fold, subfold)
    
    filedata = open(os.path.join(thisfold, 'files.txt'), 'r')
    cell_list = filedata.readlines()
    filedata.close()
    
    for fil in os.listdir(thisfold):
        if '.abf' in fil:
            file_name = fil[:-7]
            break
    
    cumulative = [[] for _ in range(len(cell_list[0].split('|')))]
    
    for cell in cell_list:
        cell = cell.strip()
        states = [x.split(',') for x in cell.split('|')]
        
        for state in range(len(states)):
            this_taus = []
            
            for fil in states[state]:    
                this_file = os.path.join(thisfold, file_name + long_number(fil) + '_events.txt')
                
                event_file = open(this_file, 'r')
                event_data = event_file.readlines()
                event_file.close()
                
                taus = [float(x.split('\t')[2].strip()) for x in event_data]
                this_taus = np.mean(taus)
                
            cumulative[state].append(np.mean(this_taus))
    
    if len(overall) == 0:
        overall = cumulative
    else:
        for item in range(len(overall)):
            [overall[item].append(x) for x in cumulative[item]]

out = open(os.path.join(fold, 'tau_summary.txt'), 'w')

for item in range(len(overall[0])):
    for col in range(len(overall)):
        out.write(str(overall[col][item]) + '\t')
    out.write('\n')

out.close()
