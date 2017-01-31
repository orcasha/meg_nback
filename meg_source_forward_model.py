import mne
import mayavi.mlab
import numpy as np
from surfer import Brain
import glob
import os

sensor_dir = '/media/orcasha/Analysis_Drive/studies/meg/data/sensor/'
source_dir = '/media/orcasha/Analysis_Drive/studies/meg/data/source/'
sourceP_dir = '/media/orcasha/Analysis_Drive/studies/meg/data/source_peripheral/'
fs_dir = '/home/orcasha/freesurfer/subjects/'

subj_list = glob.glob(fs_dir + 'c1*')
subj_list.sort()

subs = ['c102', 'c104']

subj = subj_list[2]

for subj in subj_list:
    subj_id = os.path.split(subj)[1]
  
    #Use MNE C to make BEM and high res scalp image for coregistration
    os.system('mne watershed_bem -o -s ' + subj_id)
    os.system('mne make_scalp_surfaces -s ' + subj_id)
    
    #Plot BEM output, save to FSDIR/subj/bem dir
    fig = mne.viz.plot_bem(subject = subj_id, brain_surfaces = 'white', orientation = 'coronal', show = False)
    fname = os.path.join(subj, 'bem', subj_id + '_bem')
    fig.savefig(fname)

    #Load evoked file w/ head pos info, coregister scalp to sensors (see http://www.slideshare.net/mne-python/mnepythyon-coregistration-28598463 for coreg help)
    inst = sensor_dir + subj_id[:4] + '_conds_freqs-ave.fif'
    mne.gui.coregistration(subject = subj_id, tabbed = True, inst = inst)

    #Load trans file, plot and save to source_peripheral
    info = mne.io.read_info(sensor_dir + subj_id[:4] + '_conds_freqs-ave.fif')
    trans = sourceP_dir + subj_id + '-trans.fif' #mne coreg saves transform file under FS subj ID.
    fig_coreg = mne.viz.plot_trans(info, trans, subject = subj_id, dig = True, meg_sensors = True)
    mayavi.mlab.savefig(sourceP_dir + subj_id[:4] + '_coreg.png')
    mayavi.mlab.close()

    #Set up source space (aka dipole grid) w/ ~5mm spacing (see http://martinos.org/mne/stable/manual/cookbook.html#setting-up-source-space)
    src = mne.setup_source_space(subj_id, spacing = 'oct6', add_dist = False, overwrite = True)

    #Plot grid locations on brain, save to source_peripheral  

    #NEED TO DO - FIGURE OUT HOW TO GET CHILD WINDOW
    brain = Brain(subj_id, "both", "inflated", views="frontal")
    for b in brain.brains:
        surf = b._geo
        if surf.hemi == 'lh':
            vertidx = np.where(src[0]['inuse'])[0]
        elif surf.hemi == 'rh':
           vertidx = np.where(src[1]['inuse'])[0] 
        mayavi.mlab.points3d(surf.x[vertidx], surf.y[vertidx], surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)
        brain.save_image(sourceP_dir + subj_id + '_vertex.png')
    brain.close()


    #Compute BEM / forward model (aka electromagnetic current runs thusly...)
    conductivity = (0.3,)  # for single layer
    model = mne.make_bem_model(subject = subj_id, ico = 4, conductivity = conductivity)
    mne.write_bem_surfaces(sourceP_dir + subj_id[:4] + '_bem_model.fif', model)
    
    bem_sol = mne.make_bem_solution(model)
    mne.write_bem_solution(sourceP_dir + subj_id[:4] + '_bem_sol.fif', bem_sol)    

    infosrc = sensor_dir + subj_id[:4] + '_conds_freqs-ave.fif'
    fname = sourceP_dir + subj_id[:4] + '-fwd.fif'
    fwd = mne.make_forward_solution(info = infosrc, trans = trans, src = src, bem = bem_sol, fname = fname, meg = True, eeg = False, mindist=5.0, n_jobs = 3)
    
    

    

    

    
    


