# -*- coding: utf-8 -*-
"""
@author: Felipe Tapia
"""

import os
import locale
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from neo.io import AxonIO
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import find_peaks #, peak_widths
from scipy.integrate import trapezoid
from statsmodels.distributions.empirical_distribution import ECDF
from pybaselines import Baseline


def short_number(x):
    """
    Return number rounded to 2 positions as a string.
    """
    return str(round(x, 2))

def long_number(x):
    """
    Return padded integer to 3 spaces as a string.
    """
    num = str(x)
    return ('0'*(3-len(num))) + num

def make_conditions(times):
    """
    Return a list of 'times' empty lists.
    """
    return [[] for _ in range(times)]

def write_pairs(filename, dataA, dataB):
    """
    Writes a list data pairs (A and B) as TAB separated values to a text file.
    """
    with open(filename, 'w') as data_file:
        for value in range(len(dataA)):
            data_file.write(str(dataA[value]) + '\t' + str(dataB[value]) + '\n')
    
def get_baselinenoise(data, file_path):
    """
    Attempts to read a baseline data file,
    calculates baseline and creates a new one if not found
    Baseline is calculated using noise_median, the lower bound of the trace
    is calculated using ipsa over the straigtened data and taken as noise threshold
    """
    baseline_file = file_path.replace('.abf', '_baseline.npy')
    thresh_file = file_path.replace('.abf', '_noise.npy')
    
    if os.path.exists(baseline_file):
        baseline = np.load(baseline_file)
        noise_thres = float(np.load(thresh_file))
    else:
        baseline_fitter = Baseline(check_finite=False)
        baseline = baseline_fitter.noise_median(data, half_window=2000, smooth_half_window=5000)[0]
        flat_data = data.flatten() - baseline
        lower_bound = baseline_fitter.ipsa(flat_data)
        noise_thres = np.abs(np.quantile(lower_bound[0], 0.95))
        np.save(baseline_file, baseline)
        np.save(thresh_file, noise_thres)
    
    return(baseline, noise_thres)

#Sets locale for data files
locale.setlocale(locale.LC_ALL, 'es_CL')

#=============================================================
#Parameters
#=============================================================
#Files are read from the base folder, the structure should be:
#base folder
#    -experiment folder
#        -daily folder
#            -abf files
#            -"files.txt" list of abf file numbers grouped by cell and condition

fold = r'C:\...'
subdirs = next(os.walk(fold))[1]

aq_rate = 10000 #Hz
distance = 10 * (aq_rate/1000) #ms
baseline_time = int(1.5 * aq_rate) #s
stimulus_start = 10 * aq_rate #s
minimum_width = 0.7 * (aq_rate/1000) #ms
trim_window = 500 #samples
lower_samples = 150 #samples
lowcut = 100.0 #Hz
times_noise_thresh = 2.75
plot = True
#=============================================================


plt.rcParams['interactive'] == False
matplotlib.use('Agg')

def process_cell(folder, file_sample, celldata):
    """
    Process abf files for a subset of cells inside the daily "folder",
    "file_sample" is the invariant part of the file names for the current day,
    "celldata" is the text data file for the day, contains the file numbers
    for the abf files, each line is a cell, conditions for each cell are separated
    by "|", for example for 2 cells in a day, with two conditions recorded for each
    and 3 consecutive recordings for condition, the file would look like this:
    1,2,3|4,5,6
    7,8,9|10,11,12
    """
    
    conditions = [x.split(',') for x in celldata.split('|')]
    all_files = [item for sublist in conditions for item in sublist]
    
    processed_files = {}
    
    frequencies = make_conditions(len(conditions))
    amplitudes = make_conditions(len(conditions))
    
    amp_agg = make_conditions(len(conditions)) #Amplitudes
    tau_agg = make_conditions(len(conditions)) #Decay time
    auc_agg = make_conditions(len(conditions)) #Area under the curve
    iei_agg = make_conditions(len(conditions)) #Inter-event interval
    
    for file in all_files:
        #Preprocess files, calculate and apply baselines and thresholds
        #Filter signals using a digital butterwoth filter
        
        full_path = os.path.join(folder, file_sample + long_number(file) + '.abf')
        
        r = AxonIO(filename = full_path)
        bl = r.read_block(lazy=False)
        
        data = bl.segments[0].analogsignals[0]
        
        b, a = butter(3, lowcut, fs=aq_rate)
        data = filtfilt(b, a, data, axis=0)
        
        baseline, noise = get_baselinenoise(data, full_path)
        
        base = float(np.mean(data[:baseline_time]))
        data = data - base
        baseline = baseline - base
               
        processed_files[file] = [file_sample + long_number(file), data, noise, baseline]
        
        print('Noise: ' + str(round(noise, 2)) + ' pA')
    
    for state in range(len(conditions)):
        for file in conditions[state]:
            #Read stored processed data and find peaks
            
            name, data, base_noise, baseline = processed_files[file]
           
            flatline = data.flatten() - baseline
            
            peaks, _ = find_peaks(flatline, height=base_noise * times_noise_thresh, distance=distance, width=minimum_width)
            
            events = []
            event_data = []

            for peak in peaks:
                target = float(flatline[peak])
                target *= np.exp(-1)

                left_side = 0
                right_side = 0
                rightf = False
                leftf = False
                
                #Searches 10000 samples in front and behind the current peak
                #until it finds an amplitude <= to e times the current peak,
                #this area is also used to find double peaks and exclude them
                #from event shape analysis
                #The AUC is also calculated from this area, so this parameter
                #can be tweaked if AUC measurement is to be used
                
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
            
            ev_amplitudes = [float(flatline[x]) for x in peaks]
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
            print('Frequency (events/time): ' + short_number(freq) + ' Hz')
            

            if plot:
                plt.figure(figsize=(12,10), dpi=80)
                plt.plot(data, linewidth=0.5, color='gray')
                plt.plot(baseline, color='w')
                plt.plot(baseline+base_noise, color='k')
                plt.plot(baseline-base_noise, color='k')
                plt.plot(baseline+(base_noise*times_noise_thresh), linestyle='--', color='r')
                plt.plot(peaks, [data[x] for x in peaks], "x", color='b')
                plt.axhline(y=base)
                plt.xlim(0, len(data)-1)
                plt.ylim(min(data)-5, max(data)+5)
        
                os.makedirs(os.path.join(thisfold, 'graphs'), exist_ok=True)
                plt.savefig(os.path.join(thisfold, 'graphs' , name + '.png'))
                print(os.path.join(thisfold, 'graphs' , name + '.png'))
                
                for item in events:
                    plt.plot(item)
                    
            np.savetxt(os.path.join(thisfold, name + '_events.txt'), event_data, delimiter='\t', fmt='%s')
               
    return [frequencies, amplitudes, tau_agg, amp_agg, auc_agg, iei_agg]

    
cum_ecdf_amp = []

parameters = [[], [], [], []]
param_labels = ['Tau', 'Amplitude', 'AUC', 'IEI']

state_count = 0

for subfold in subdirs:
    #Read the contents of the daily folders, create the file list from the
    #"files.txt" files inside and process and aggregate the data
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
                parameters[param] = make_conditions(state_count)
            
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
        
bin_list = make_conditions(state_count)

trimmed_params = [make_conditions(state_count), make_conditions(state_count), make_conditions(state_count), make_conditions(state_count)]

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
        with open(write_file, 'w') as raw_file:
            for value in this_param[state]:
                raw_file.write(str(value) + '\n')


for param in range(len(parameters)):
    this_param = parameters[param]
    for state in range(len(this_param)):
        #HISTOGRAM---------------------------------
        bin_number = np.max([x[param] for x in bin_list])
        minimum = np.min([np.min(x) for x in this_param])
        maximum = np.max([np.max(x) for x in this_param])
        
        hist, edges = np.histogram(this_param[state], int(bin_number), (minimum, maximum), density=True)

        full_edges = [str(edges[x]) + ';' + str(edges[x+1]) for x in range(len(edges) - 1)]
        write_file = os.path.join(fold, 'resultsHist_' + param_labels[param] + '_' + str(state) + '.txt')
        write_pairs(write_file, hist, full_edges)
        
amp_data = [[],[],[]]
freq_data = [[],[],[]]

for subfold in subdirs:
    #Read files with amplitude and frequency data
    #to create an overall summary
    thisfold = os.path.join(fold, subfold)
    
    with open(os.path.join(thisfold, 'resultsAmp.txt'), 'r') as file:
        full_text = file.readlines()
        counter=0
        for line in full_text:
            if line=='\n':
                counter += 1
                continue
            
            if counter > len(amp_data) - 1:
                break
            
            amp_data[counter].append(line)
            
    with open(os.path.join(thisfold, 'resultsFreq.txt'), 'r') as file:
        full_text = file.readlines()
        counter=0
        for line in full_text:
            if line=='\n':
                counter += 1
                continue
            
            if counter > len(amp_data) - 1:
                break
            
            freq_data[counter].append(line)
            
with open(os.path.join(fold, 'summary_Amp.txt'), 'w') as file:
    for item in amp_data:
        if len(item) > 0:
            file.writelines(item)
            file.writelines('\n')

with open(os.path.join(fold, 'summary_Freq.txt'), 'w') as file:
    for item in freq_data:
        if len(item) > 0:
            file.writelines(item)
            file.writelines('\n')
            
overall = []

for subfold in subdirs:
    #Read files with decay time and frequency data
    #to create an overall summary
    
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
        conditions = [x.split(',') for x in cell.split('|')]
        
        for state in range(len(conditions)):
            this_taus = []
            
            for fil in conditions[state]:    
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

out = open(os.path.join(fold, 'summary_Tau.txt'), 'w')

for item in range(len(overall[0])):
    for col in range(len(overall)):
        out.write(str(overall[col][item]) + '\t')
    out.write('\n')

out.close()
