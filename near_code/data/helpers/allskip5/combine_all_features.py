import numpy as np
import glob

features = ['norm_trajs','offense_dist2ball','defense_dist2bh','offense_dist2bh','offense_dist2basket','offense_paint']

for set_name in ['test','train']:
    full_arr = []
    for filename in features:
        a=np.load(set_name+'_'+filename+'.npy')
        full_arr.append(a)
    final = np.concatenate(full_arr,axis=-1)
    np.save(set_name+'_fullfeatures.npy',final)
    print(final.shape)