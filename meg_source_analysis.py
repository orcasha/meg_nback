import mne
import numpy as np
import glob
import os
import sys
from surfer import Brain

sensor_dir = '/media/orcasha/Analysis_Drive/studies/meg/data/sensor/'
source_dir = '/media/orcasha/Analysis_Drive/studies/meg/data/source/'
sourceP_dir = '/media/orcasha/Analysis_Drive/studies/meg/data/source_peripheral/'

if sys.platform == 'win32':
    rpath = 'i:/'
    n_cores = 7
else:
    rpath = '/media/orcasha/Analysis_Drive/'
    n_cores = 3


subj_files = glob.glob(sensor_dir + '*_epochs_cond-epo.fif')
subj_files.sort()

subj = subj_files[1]
subj_id = os.path.split(subj)[1][:4]

fwd = mne.read_forward_solution(sourceP_dir + subj_id + '-fwd.fif')


def meg_calc_noise_cov(subj):
    '''
    Calculate noise covariance using rest epochs.
    '''
    subj = subj     
    epo = mne.read_epochs(subj)
    noise_cov = mne.compute_covariance(epo['rest'], tmin = 0.0, method = 'empirical', n_jobs = n_cores)
    noise_cov.plot(epo.info)
    mne.write_cov(os.path.join(sourceP_dir, subj_id + '_noise-cov.fif'), noise_cov) 
    return(epo, noise_cov)

def meg_calc_inv_op(epo, fwd, noise_cov):
    '''
    Calculate inverse operator (aka spinning dipole model)
    Note: Uses only the info from epochs, not data.
    '''
    info = epo.info
    noise_cov = noise_cov
    inv_op = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8, fixed=False, limit_depth_chs=True, rank=None, verbose=None)
    mne.minimum_norm.write_inverse_operator(sourceP_dir + subj_id + '-inv.fif', inv_op)
    return(inv_op)

def meg_source_psd(epo, cond, inv_op):
    '''
    Calculate source model in PSD
    '''
    method = 'dSPM'
    snr = 3
    lambda2 = 1.0 / snr**2
    stc = mne.minimum_norm.compute_source_psd_epochs(epo[cond], inv_op, lambda2, method, fmax = 100)
    return(stc)
   

def meg_source_psd_average(stc, freqs):

    if len(freqs) < 2:
        print('Low and high frequencies must be included')

    stc_av = stc[0].copy()
    d_shape = stc_av.data.shape
    h = np.zeros([d_shape[0], d_shape[1], len(stc)])
    
    for x in range(0,len(stc)):
        h[:,:,x] = stc[x].data

    stc_av._data = np.mean(h, axis = 2)
    all_freqs = stc_av.times
    f_mask = [(all_freqs > freqs[0]) & (all_freqs < freqs[1])]
    

def plot_stc(stc):
    '''
    Plot average stc within freq range.
    '''

    brain = Brain(subj_id, 'split', 'inflated', size=(800, 400), views=['lat','med'])

    freqs = stc_av.times
    f_mask = [(freqs > low) & (freqs < high)]
    f_idx = np.nonzero(f_mask)[1]

    freqs_mask = freqs[f_mask]

    l_data = stc_av.lh_data
    l_data_mask = l_data[:,f_idx]
    l_data_mean = np.mean(l_data_mask, axis = 1)
    l_vertices = stc_av.lh_vertno

    r_data = stc_av.rh_data
    r_data_mask = r_data[:,f_idx]
    r_data_mean = np.mean(r_data_mask, axis = 1)
    r_vertices = stc_av.rh_vertno

    cmap = 'nipy_spectral'
    smooth = 10
    time = np.mean(freqs_mask)

    brain.add_data(l_data_mean, colormap = cmap, vertices = l_vertices, smoothing_steps = smooth, time = time, colorbar = True, hemi = 'lh') 

    brain.add_data(r_data_mean, colormap = cmap, vertices = r_vertices, smoothing_steps = smooth, time = time, colorbar = True, hemi = 'rh')   

    return(brain)   
        

[epo,cov] = meg_calc_noise_cov(subj) 
inv_op = meg_calc_inv_op(epo, fwd, cov)
stc = meg_source_psd(epo, inv_op) 
brain = plot_stc_av(stc, 0, 7)


