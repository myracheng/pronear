import numpy as np
import glob

features = ['norm_trajs','offense_dist2ball','defense_dist2bh','offense_dist2bh','offense_dist2basket','offense_paint','ball_inpaint']

for set_name in ['test','train']:
    full_arr = []
    for filename in features:
        a=np.load(set_name+'_'+filename+'.npy')
        full_arr.append(a)
    final = np.concatenate(full_arr,axis=-1)
    a=np.load(set_name+'_2fts.npy')
    print(np.shape(final))
    print(np.shape(a))
    
    final = np.concatenate([final,a],axis=-1)

    a = np.expand_dims(np.load(set_name+'_ballhandlers.npy'), axis=-1)

    final = np.concatenate([final,a],axis=-1)

    np.save(set_name+'_fullfeatures_2.npy',final)
    print(final.shape)