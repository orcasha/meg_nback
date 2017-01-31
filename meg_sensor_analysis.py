'''
MEG preprocessing / analysis script
NOTE: Assumes data has been cleaned previously using functions in meg_artifact_
'''


import mne
import glob
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import itemfreq
import os
import sys


pre = -0.2
post = 1.6

restpre = 0.0
restpost = 1.8

baseline = (None,None)

if sys.platform == 'win32':
    rpath = 'i:/'
    cores = 7
else:
    rpath = '/media/orcasha/Analysis_Drive/'
    cores = 3

h_dir = rpath + 'studies/meg/'
raw_dir = h_dir + 'fiff_ica_clean/'
out_dir = h_dir + 'data/sensor/'

def meg_calc_epochs():
    subj_pathfiles = glob.glob(raw_dir + '*tsss.fif')
    subj_files = [os.path.split(subj_pathfiles[x])[1] for x in range(0,len(subj_pathfiles))]
    subj_files.sort()

    for n in range(0, len(subj_files)):
        fiff_file = subj_files[n]
        subj_id = fiff_file[:5]

        raw=mne.io.Raw(raw_dir + fiff_file,preload=True)
        picks_meg = mne.pick_types(raw.info, meg = True)

        raw.filter(l_freq = 1, h_freq = 150, picks = None, filter_length = '10s', l_trans_bandwidth = 0.5, h_trans_bandwidth = 0.5, method = 'fft', iir_params = None, n_jobs = cores,  verbose = None)
        #raw.filter(l_freq = 51, h_freq = 49, n_jobs = cores, picks = picks_meg) #Notch filter.
        dim=raw.estimate_rank()
    
        print('\n***The dimensionality of the data is %i***\n' %(dim,))
    
        #FIX CHANNELS#
        try:
            raw.drop_channels(['BIO001'])
        except:
            print '\n***No BIO channel present in data.***\n'
    
    
        ###FIX TRIGGERS###
        #Removes all bits > 8(255 value) from a holder variable tr_ch
        tr_ch=raw._data[raw.ch_names.index('STI101')].astype(int)&0xff
        
        #Re-inserts manipulted trigger channel
        
        raw._data[[raw.ch_names.index('STI101')]]=tr_ch
        
        #Find triggers
        ev = mne.find_events(raw, stim_channel = 'STI101', output = 'onset', consecutive = 'TRUE', shortest_event = 1)
        
        #Note: Due to the way find_events works, loop checks the value after the current x to see if the answer was correct (1). 
        cor=[]
        incor=[]
        rest=[]
        for x in range(0,len(ev)-1):
            if ev[x+1,2]==1:
                cor.append(ev[x,:])
            elif ev[x+1,2] == 9:
                incor.append(ev[x,:])
            elif ev[x,2] == 99:
                rest.append(ev[x,:])
        
        ev_rest=np.array(rest)
        
        ev_cor = np.array(cor)
        ev_cor_truth = np.logical_and(ev_cor[:,2] >=110, ev_cor[:,2] <=233) #numpy cannot do truth tables with two conditions (a and b) so it has to be wrapped in the logical_and function.
        ev_cor_list = ev_cor[ev_cor_truth]
        ev_cor_list=np.concatenate((ev_cor_list,ev_rest))
        
        ev_incor = np.array(incor)
        ev_incor_truth = np.logical_and(ev_incor[:,2] >=110, ev_incor[:,2] <=233) #numpy cannot do truth tables with two conditions (a and b) so it has to be wrapped in the logical_and function.
        ev_incor_list = ev_incor[ev_incor_truth]
        ev_incor_list=np.concatenate((ev_incor_list,ev_rest))
        
        cor_counts = itemfreq(ev_cor_list[:,2])
        incor_counts = itemfreq(ev_incor_list[:,2])
        
    
        ###FIND REJECT THRESHOLD###
        #Find individual thresholds at 5x channel type SD.
        picks_mag = mne.pick_types(raw.info, 'mag', stim = False, chpi = False)
        picks_grad = mne.pick_types(raw.info, 'grad', stim = False, chpi = False)   
    
        #MAGS
        mag_mean = np.mean(raw._data[picks_mag,:])
        mag_std = np.std(raw._data[picks_mag,:])
        mag_reject = mag_mean + (5*mag_std)
        
        #GRADS
        #grad_mean = np.mean(raw._data[picks_grad,:])
        #grad_std = np.std(raw._data[picks_grad,:])
        #grad_reject = grad_mean + (5*grad_std)
    
        #reject = {'mag': mag_reject}
        reject = {'mag': 4e-11, 'grad': 4e-10}
        #print("Data reject threshold set to %f for mags and %f for grads" %reject['mag'],reject['grad'])
    
        
        ###Epoch data##
        
        #Select only MEG channels
        picks_meg = mne.pick_types(raw.info, meg = True)
        
        #Select trials of interest
        cond_id = {'mnilhap':110, 'mhaphap':111, 'msadhap':112, 'mneuhap':113, 'fnilhap':210, 'fhaphap':211, 'fsadhap':212, 'fneuhap':213, 'mnilsad':120, 'mhapsad':121, 'msadsad':122, 'mneusad':123, 'fnilsad':220, 'fhapsad':221, 'fsadsad':222, 'fneusad':223, 'mnilneu':130, 'mhapneu':131, 'msadneu':132, 'mneuneu':133, 'fnilneu':230, 'fhapneu':231, 'fsadneu':232, 'fneuneu':233}
        
        
        rest_id = {'rest':99}
        
        epochs_cond = mne.Epochs(raw, picks = picks_meg, events = ev_cor_list, event_id = cond_id, tmin = pre, tmax = post, baseline = baseline, reject = reject)
        
        epochs_rest = mne.Epochs(raw, picks = picks_meg, events = ev_cor_list, event_id = rest_id, tmin = restpre, tmax = restpost, baseline = baseline, reject = reject)
        
        
        #Combine condition sub-categories
        
        
        mne.epochs.combine_event_ids(epochs_cond, ['mnilhap','mhaphap', 'msadhap', 'mneuhap', 'fnilhap', 'fhaphap', 'fsadhap', 'fneuhap'], {'hap': 1}, copy = False)
        
        mne.epochs.combine_event_ids(epochs_cond, ['mnilsad','mhapsad', 'msadsad', 'mneusad', 'fnilsad', 'fhapsad', 'fsadsad', 'fneusad'], {'sad': 2}, copy = False)
        
        mne.epochs.combine_event_ids(epochs_cond, ['mnilneu','mhapneu', 'msadneu', 'mneuneu', 'fnilneu', 'fhapneu', 'fsadneu', 'fneuneu'], {'neu': 3}, copy = False)
    
           
        #Concatenate cond and rest objects
        epochs_cond.event_id =  {'hap': 1, 'neu': 3, 'sad': 2, 'rest': 99}
        epochs_cond._data = np.concatenate([epochs_cond.get_data(), epochs_rest.get_data()])
        epochs_cond.events = np.concatenate([epochs_cond.events, epochs_rest.events])
        
        epochs_cond.select = np.arange(0,np.shape(epochs_cond.events)[0])
        epochs_cond.selection = np.arange(0,np.shape(epochs_cond.events)[0])
        
        #Normalise epochs to rest condition (non-bias)
        epochs_cond.equalize_event_counts(['hap','sad','neu','rest'], method = 'mintime')
       
        #Save object
        epochs_cond.save('%s%sepochs_cond-epo.fif' %(out_dir, subj_id,))


def meg_calc_evokeds():
    subj_files = glob.glob(out_dir + '*epochs*.fif')
    subj_files.sort()
    for n, fiff_file in enumerate(subj_files):
    
        subj_id = os.path.split(fiff_file)[1][:5]
        fname = out_dir + subj_id + 'cond-ave.fif'   

        epo = mne.read_epochs(fiff_file)
        ev = [epo[x].average() for x in ['hap', 'sad', 'neu', 'rest']]
        new = ev[0].copy()
        new.comment = 'cond_av'
        new.data = np.mean([ev[0].data, ev[1].data, ev[2].data], axis = 0)
        ev.append(new)
        mne.write_evokeds(fname, ev)


def meg_calc_psd():
    cond_list = ['hap', 'sad', 'neu', 'rest'] # <- Controls output
    subj_files_epo = glob.glob(out_dir + '*cond-epo.fif')
    subj_files_epo.sort()
    
    subj_files_evo = glob.glob(out_dir + '*cond-ave.fif')
    subj_files_evo.sort()
    
    for x, fiff_file in enumerate(subj_files_epo):
        subj_id = os.path.split(fiff_file)[1][:5]
        print(subj_id)
        epo = mne.read_epochs(fiff_file) 
        evo = mne.read_evokeds(subj_files_evo[x], baseline = [None, None])     
        
        print('\n***CALCULATING MULTITAPER PSD***\n')
        #Note: Frequency bandwidth (steps in which frequency bins are calcuated, NOT max frequency attainable)
        #is calculated using 1/t, where t = time (in seconds) of used data.
        #NOTE 2: Frequency bandwidth gets screwed up on save. Writing array to load into output PSDs
        
        if x == 0:
            freq_av = [mne.time_frequency.psd_multitaper(epo[n], fmax = 120.0, tmin = 0, bandwidth = 1/epo.tmax, low_bias = True, n_jobs = 3) for n in cond_list]
            np.save(out_dir + 'freq_list', freq_av[0][1])
        else: 
            freq_av = [mne.time_frequency.psd_multitaper(epo[n], fmax = 120.0, tmin = 0, bandwidth = 1/epo.tmax, low_bias = True, n_jobs = 3) for n in cond_list]    
        
        print('\n***PSD CALCULATION COMPLETE. AVERAGING CONDITION EPOCHS***\n')
        for y in range(0,len(freq_av)):
            evo[y].comment = cond_list[y] + '_psd'
            evo[y].data = np.mean(freq_av[y][0], axis = 0)

            print('Creating new frequency object for condition %i' %(y,))
            nsamp = evo[y].data.shape[1]
            evo[y].times = np.around(freq_av[y][1], decimals = 3)
            evo[y].info['times'] = np.around(freq_av[y][1], decimals = 3) #Create hide variable so times can be correct after load.
            evo[y].first = evo[y].times[0]
            evo[y].last = evo[y].first + nsamp - 1
            
        
        #Create average over conditions except rest
        new = evo[0].copy()
        new.comment = 'CondAv_psd'
        new.data = np.mean([evo[0].data, evo[1].data, evo[2].data], axis = 0) #0 = hap, 1 = sad, 2 = neu (see cond_list)
        evo[4] = new.copy()
        print('\n***SAVING NON-COMPENSATED DATA FOR %s***\n' %(subj_id,))
        mne.write_evokeds(out_dir + subj_id + 'conds_freqs-ave.fif', evo)

        print('\n***COMPENSATING FOR 1/f, SAVING DATA FOR %s***\n' %(subj_id,))
        for x in range(0,len(evo)):
            evo[x].data = evo[x].data * evo[x].times
        mne.write_evokeds(out_dir + subj_id + 'conds_freqs_comp-ave.fif', evo)
    
     


