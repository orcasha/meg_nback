import mne
import numpy as np
import matplotlib.pylab as plt
import glob
import os
import sys
import scipy.stats

if sys.platform == 'win32':
    rpath = 'i:/'
    cores = 7
else:
    rpath = '/media/orcasha/Analysis_Drive/'
    cores = 3


h_dir = rpath + 'studies/meg/data/sensor/'

n_perms = 10000

cond_list = 'hap_psd', 'sad_psd', 'neu_psd', 'CondAv_psd'



def meg_sens_freq_stats(cond):
    connectivity, ch_names = mne.channels.read_ch_connectivity('neuromag306mag') #Set connectivity object.

    #Set up subjs and holding matrix
    subj_files = glob.glob(h_dir + '*conds_freqs_comp-ave.fif')
    subj_files.sort()
    d_shape = mne.read_evokeds(subj_files[0], condition = cond, baseline = [None, None]).data.shape
    cond_mat = np.zeros([d_shape[0], d_shape[1], len(subj_files)])
    rest_mat = np.zeros([d_shape[0], d_shape[1], len(subj_files)])

    for n, fname in enumerate(subj_files):
        print('\n***Calculating permutation statistics on %s***\n' %(cond,))
        data_hold = mne.read_evokeds(subj_files[n], condition = [cond, 'rest_psd'], baseline=[None, None])
        cond_mat[:,:,n] = data_hold[0].data
        rest_mat[:,:,n] = data_hold[1].data

    #Get mag data only, reorder axes to be obs x spectra x channel
    cond_mags = cond_mat[0:-1:3,:,:].swapaxes(0,2)
    rest_mags = rest_mat[0:-1:3,:,:].swapaxes(0,2)
    
    X_mags = cond_mags - rest_mags

    cluster_stats = mne.stats.permutation_cluster_1samp_test(X_mags, tail = 0, n_permutations = n_perms, n_jobs = cores, connectivity = connectivity)
    fname_stats = str.lower(cond) + '_perms_' + str(n_perms)
    np.save(h_dir + 'permutation/' + fname_stats, cluster_stats) #Saves as np array.     
        
    t_obs, clusters, p_values, h0 = cluster_stats
    return(t_obs, clusters, p_values, h0)



def meg_sig_clusts(t,c,p):
    sig_clusts = np.squeeze(np.where(p < 0.05))
    
    #Either kill function or get matrix with sig clusters marked as 1 (else 0)
    if len(sig_clusts) < 1:
        print('\n**No significant clusters found**\n')
    else:
        sig_clusts_matrix = np.zeros_like(t)
        for x in sig_clusts:
            sig_clusts_matrix = sig_clusts_matrix + c[x]
        
        t_sig = t * sig_clusts_matrix
        return(t_sig)


def meg_plot_sig_clusts(t_sig, t):
    """
    Plots significant clusters using output from MNE python's permutation function
    """
    t = np.load(t)
    
    ax = plt.imshow(t_sig.T, interpolation = 'nearest')
    plt.xlim([min(t), max(t)])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Channel')
    plt.savefig(h_dir + 'permutation/' + cond)
