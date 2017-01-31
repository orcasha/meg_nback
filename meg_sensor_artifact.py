from __future__ import division
import mne
import glob
import os
import sys
from matplotlib import pylab as plt

if sys.platform == 'win32':
    rpath = 'i:/'
    cores = 7
else:
    rpath = '/media/orcasha/Analysis_Drive/'
    cores = 3



fiff_dir = rpath + 'studies/meg/max_notrans/'
sensor_dir = rpath + 'studies/meg/data/sensor/'
peripheral_dir = rpath + 'studies/meg/data/peripheral/'
fiff_clean_dir = rpath + 'studies/meg/fiff_ica_clean/'



fiff_files = glob.glob(fiff_dir + 'c*meg_raw_tsss.fif')
fiff_files.sort()
ica_files = glob.glob(peripheral_dir + 'c*-ica.fif')
ica_files.sort()

def meg_ica_run(fiff_files):
    for subj in fiff_files[-4:]:
        raw = mne.io.read_raw_fif(subj, preload=True, add_eeg_ref=False)
        raw.filter(1,30, n_jobs = cores) #Filter data to make locating artifact easier.
        fname = os.path.split(subj)[0]
        reject = dict(mag=5e-12, grad=4000e-13)
        ica = mne.preprocessing.ICA(n_components = 0.95, method = 'fastica', random_state = 33).fit(raw, decim = 20, reject = reject)
        ica.save(peripheral_dir + os.path.split(subj)[1][:10] + '-ica.fif')

#EOG
def eog_ica_corr(fiff_files, ica_files):
    raw = mne.io.read_raw_fif(fiff_files[0], preload=True, add_eeg_ref=False)
    icas = [mne.preprocessing.read_ica(fname) for fname in ica_files]
    ref_ica = icas[0]
    ref_ica.plot_sources(raw, block = True)
    template = (0,ref_ica.exclude[0])
    eog_corr = mne.preprocessing.corrmap(icas, template = template, plot = True, show = True, ch_type = 'mag', threshold = 'auto')
    eog_group = eog_corr[1]
    eog_group.savefig(peripheral_dir + 'group_eog_corr.jpg')
    

eog_list = [0, 1, 1, 0, 0, 1, 1, 1, 0, 3, 0, 0, 0]    

#ECG
def ecg_raw_clean(fiff_files, ica_files):
    eog_list = [0, 1, 1, 0, 0, 1, 1, 1, 0, 3, 0, 0, 0]    
    fiff_files = fiff_files[-4:]
    eog_list = eog_list[-4:]
    ica_files = ica_files[-4:]
    
    for n, subj in enumerate(fiff_files):
        raw = mne.io.read_raw_fif(subj, preload=True, add_eeg_ref=False)
        raw_filter = raw.copy().filter(1,30, n_jobs = cores)
        ica = mne.preprocessing.read_ica(ica_files[n])
        ica.plot_sources(raw_filter, block = True)
        ica.exclude.append(eog_list[n])
        ica.apply(raw, exclude = ica.exclude)
        ica.save(peripheral_dir + os.path.split(subj)[1][:10] + '-ica.fif')
        raw.save(fiff_clean_dir + os.path.split(subj)[1][:19] + 'clean-tsss.fif')
    

