import numpy as np
a = np.load('acs.npy')
import matplotlib.pyplot as plt
# import seaborn as sns 
# sns.set(rc={'figure.figsize':(11.7,8.27)})
"""TSNE"""
# print(np.shape(v))
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(a)
# pal = sns.cubehelix_palette(100, start=2, rot=0, dark=0, light=.95)

plt.scatter(X_embedded[:,0], X_embedded[:,1], legend='full')

plt.savefig('test_tsne.png')
