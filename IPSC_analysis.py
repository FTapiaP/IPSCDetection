# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:59:54 2020
Last edited on 18-02-22 18:19
@author: Felipe Tapia
"""

import os
import locale
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from neo.io import AxonIO
from scipy.ndimage import gaussian_filter
from scipy.stats import iqr
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import find_peaks #, peak_widths
from scipy.integrate import trapezoid
from statsmodels.distributions.empirical_distribution import ECDF


def shortNumber(x):
    return str(round(x, 2))

def long_number(x):
    num = str(x)
    while len(num) < 3:
         num = '0' + num
    return num

def make_states(times):
    return [[] for _ in range(times)]

def write_pairs(filename, dataA, dataB):
    data_file = open(filename, 'w')
    
    for value in range(len(dataA)):
        data_file.write(str(dataA[value]) + '\t' + str(dataB[value]))
        data_file.write('\n')
        
    data_file.close()

locale.setlocale(locale.LC_ALL, 'es_CL')

#Parameters==================================
fold = r'C:\...'
subdirs = next(os.walk(fold))[1]

aq_rate = 10000 #Hz
distance = 10 * (aq_rate/1000) #ms 10
baseline = int(1.5 * aq_rate) #s 1
stimulus_start = 10 * aq_rate #s
minimum_width = 0.7 * (aq_rate/1000) #ms 0.7
trim_window = 500 #samples
lower_samples = 150 #samples
lowcut = 100.0 #Hz
plot = True
#============================================

plt.rcParams['interactive'] == False
matplotlib.use('Agg')

def process_cell(folder, file_sample, celldata):
    states = [x.split(',') for x in celldata.split('|')]
    all_files = [item for sublist in states for item in sublist]
    
    processed_files = {}
    
    frequencies = make_states(len(states))
    amplitudes = make_states(len(states))
    
    amp_agg = make_states(len(states)) #Amplitudes
    tau_agg = make_states(len(states)) #Decay time
    auc_agg = make_states(len(states)) #Area under the curve
    iei_agg = make_states(len(states)) #Inter-event interval
    
    
    for file in all_files:
        full_path = os.path.join(folder, file_sample + long_number(file) + '.abf')
        
        r = AxonIO(filename = full_path)
        bl = r.read_block(lazy=False)
        
        data = bl.segments[0].analogsignals[0]
        
        b, a = butter(3, lowcut, fs=aq_rate)
        data = filtfilt(b, a, data, axis=0)
        
        base = float(np.mean(data[:baseline]))
        data = data-base
        base = 0
        
        print('Filtered-Zeroed')
        
        noise = float(iqr(data[:baseline]))
        noise2 = float(iqr(data[len(data) - baseline:]))
        
        noise = (noise + noise2) / 2
        
        processed_files[file] = [file_sample + long_number(file), data, noise]
        
        print('IQR: ' + str(round(noise, 2)) + ' pA')
    

    for state in range(len(states)):
        for file in states[state]:
            trace = processed_files[file]
            
            trimmed_data = []
            data = trace[1]
            base_noise = trace[2]
            
            threshold = base_noise * 2
            limit = base_noise + threshold
            
            for x in range(int(len(data)/trim_window)):
                this_data = data[x*trim_window: (x*trim_window)+trim_window]
                mins = np.mean(np.sort(this_data, axis=None)[:lower_samples])
            
                this_limitU = mins + threshold
                this_limitL = mins
                
                for y in this_data:
                    if y[0] > this_limitU:
                        trimmed_data.append(this_limitU)
                    elif y[0] < this_limitL:
                        trimmed_data.append(this_limitL)
                    else:
                        trimmed_data.append(y[0])
            
            filt = gaussian_filter(trimmed_data, sigma=int(aq_rate/5))

            print('Baseline')
            
            trim = [x[0] if x > y+limit else y for x, y in zip(data, filt)]
            
            print('Trimmed')
            
            flatline = data.flatten() - filt
            
            peaks, _ = find_peaks(trim, height=filt + limit, distance=distance, width=minimum_width)
            
            events = []
            event_data = []

            for peak in peaks:
                target = float(data[peak]) - float(filt[peak])
                target *= np.exp(-1)

                left_side = 0
                right_side = 0
                rightf = False
                leftf = False
                
                for window in range(10000):
                    next_peak = peak - window
                    if next_peak >= 0:
                        if flatline[next_peak] <= target and not leftf:
                            left_side = peak - window
                            leftf = True
                    
                    next_peak = peak + window
                    if next_peak < len(flatline):
                        if flatline[next_peak] <= target and not rightf:
                            right_side = peak + window
                            rightf = True
                        
                    if leftf and rightf:
                        break
            
                if leftf and rightf and right_side-left_side > 0:
                    event = flatline[left_side:right_side]
                    int_peaks, _ = find_peaks(event.flatten())
                    
                    if len(int_peaks) <= 1:
                        events.append(event - event[0])
                        
                        peak_time = peak / aq_rate #s
                        AUC = trapezoid(event, np.linspace(0, len(event)/aq_rate, len(event)))
                        decay_time = (right_side - peak) / 10 #ms
                        
                        event_data.append([peak_time, AUC, decay_time])
                        
                        
            [tau_agg[state].append(x[2]) for x in event_data]
            [auc_agg[state].append(x[1]) for x in event_data]
            
        
            if len(peaks)>0:
                freq = len(peaks)/(len(data)/aq_rate)
            else:
                freq=0
            
            ev_amplitudes = [float(data[x]) - float(filt[x]) for x in peaks]
            [amp_agg[state].append(x) for x in ev_amplitudes]
                
            ieis = []
            
            for item in range(1, len(peaks)):
                ieis.append((peaks[item] - peaks[item-1]) / aq_rate)
            
            [iei_agg[state].append(x) for x in ieis]

            amplitudes[state].append(np.mean(ev_amplitudes))
            frequencies[state].append(freq)
        
            print('')
            print('=====Results=====')
            print('Events: ' + str(len(peaks)))
            print('Frequency (events/time): ' + shortNumber(freq) + ' Hz')
            

            if plot:
                plt.figure(figsize=(12,10), dpi=80)
                plt.plot(data, linewidth=0.5, color='gray')
                plt.plot(filt, color='w')
                plt.plot(filt+base_noise, color='k')
                plt.plot(filt-base_noise, color='k')
                plt.plot(filt+limit, linestyle='--', color='r')
                plt.plot(peaks, [data[x] for x in peaks], "x", color='b')
                plt.axhline(y=base)
                plt.xlim(0, len(data)-1)
                plt.ylim(min(data)-5, max(data)+5)
        
                os.makedirs(os.path.join(thisfold, 'graphs'), exist_ok=True)
                plt.savefig(os.path.join(thisfold, 'graphs' , trace[0] + '.png'))
        
                plt.clf()
                
                for item in events:
                    plt.plot(item)
                plt.savefig(os.path.join(thisfold, 'graphs' , trace[0] + '_events.png'))
                np.savetxt(os.path.join(thisfold, trace[0] + '_events.txt'), event_data, delimiter='\t', fmt='%s')
                
                plt.clf()
                
                plt.hist([x[2] for x in event_data], 20, (0.0, 20.0), True)
                plt.savefig(os.path.join(thisfold, 'graphs' , trace[0] + '_hist.png'))
               
    return [frequencies, amplitudes, tau_agg, amp_agg, auc_agg, iei_agg]

    
cum_ecdf_amp = []

parameters = [[], [], [], []]
param_labels = ['Tau', 'Amplitude', 'AUC', 'IEI']

state_count = 0

for subfold in subdirs:
    thisfold = os.path.join(fold, subfold)
    
    filedata = open(os.path.join(thisfold, 'files.txt'), 'r')
    cell_list = filedata.readlines()
    filedata.close()
    file_name = ''
    
    for fil in os.listdir(thisfold):
        if '.abf' in fil:
            file_name = fil[:-7]
            break
    
    cumulative_frequencies = []
    cumulative_amplitudes = []
    
    for cell in cell_list:
        result = process_cell(thisfold, file_name, cell.strip())
        cumulative_frequencies.append(result[0])
        cumulative_amplitudes.append(result[1])
        
        state_count = len(result[2])
        
        if len(parameters[0]) == 0:
            for param in range(len(parameters)):
                parameters[param] = make_states(state_count)
            
        for state in range(state_count):
            for param in range(len(parameters)):
                [parameters[param][state].append(x) for x in result[param + 2][state]]
                
    freq_file = open(os.path.join(thisfold, 'resultsFreq.txt'), 'w')
    
    for state in range(len(cumulative_frequencies[0])):
        for cell in cumulative_frequencies:
            for value in cell[state]:
                freq_file.write(str(value) + '\t')
            freq_file.write('\n')
        freq_file.write('\n')
        
    freq_file.close()
    
    
    amplitude_file = open(os.path.join(thisfold, 'resultsAmp.txt'), 'w')
    
    for state in range(len(cumulative_amplitudes[0])):
        for cell in cumulative_amplitudes:
            for value in cell[state]:
                amplitude_file.write(str(value) + '\t')
            amplitude_file.write('\n')
        amplitude_file.write('\n')
        
    amplitude_file.close()
        
bin_list = make_states(state_count)

trimmed_params = [make_states(state_count), make_states(state_count), make_states(state_count), make_states(state_count)]

for param in range(len(parameters)):
    this_param = parameters[param]
    
    for state in range(len(this_param)):
        st_av = np.mean(this_param[state])
        st_sd = np.std(this_param[state])
        
        trimmed_params[param][state] = [x for x in this_param[state] if np.abs((x - st_av) / st_sd) < 3]

parameters = trimmed_params

for param in range(len(parameters)):
    this_param = parameters[param]
    
    for state in range(len(this_param)):
        #HISTOGRAM PARAMETERS---------------------------------
        edges = np.histogram_bin_edges(this_param[state], bins = 'auto')
        bins = len(edges) - 1
        bin_list[state].append(bins)
        
        #CDF---------------------------------
        cdf = ECDF(this_param[state])
        write_file = os.path.join(fold, 'resultsCDF_' + param_labels[param] + '_' + str(state) + '.txt')
        write_pairs(write_file, cdf.x, cdf.y)
        
        #KDE---------------------------------
        kde = sm.nonparametric.KDEUnivariate(this_param[state])
        kde.fit(kernel='epa', bw='scott', fft=False)
        write_file = os.path.join(fold, 'resultsKDE_' + param_labels[param] + '_' + str(state) + '.txt')
        write_pairs(write_file, kde.support, kde.density)
        
        #RAW---------------------------------
        write_file = os.path.join(fold, 'resultsRAW_' + param_labels[param] + '_' + str(state) + '.txt')
        raw_file = open(write_file, 'w')
        for value in this_param[state]:
            raw_file.write(str(value) + '\n')
        raw_file.close()


for param in range(len(parameters)):
    this_param = parameters[param]
    for state in range(len(this_param)):
        #HISTOGRAM---------------------------------
        bin_number = np.max([x[param] for x in bin_list])
        minimum = np.min(np.min(this_param))
        maximum = np.max(np.max(this_param))
        
        hist, edges = np.histogram(this_param[state], int(bin_number), (minimum, maximum), density=True)

        full_edges = [str(edges[x]) + ';' + str(edges[x+1]) for x in range(len(edges) - 1)]
        write_file = os.path.join(fold, 'resultsHist_' + param_labels[param] + '_' + str(state) + '.txt')
        write_pairs(write_file, hist, full_edges)
