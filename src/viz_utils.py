from sklearn import manifold

def convert_to_2d(z, perplexity=30):
    tsne = manifold.TSNE(n_components=2,
                         random_state=0,
                         init='random',
                         perplexity=perplexity)
    z2d = tsne.fit_transform(tsne)
    return z2d
