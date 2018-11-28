from sklearn import manifold

import matplotlib.pyplot as plt

def convert_to_2d(z, perplexity=30):
    tsne = manifold.TSNE(n_components=2,
                         random_state=0,
                         init='random',
                         learning_rate=15.,
                         perplexity=perplexity)
    z2d = tsne.fit_transform(z)
    return z2d

def plot_players(z, z_info, annotate_for=[], c=None, cmap='tab20b'):
    if not c:
        positions = z_info.Pos.unique()
        c_lookup = dict(zip(positions, range(len(positions))))
        c = [c_lookup[p] for p in z_info.Pos]

    plt.scatter(z[:, 0], z[:, 1], c=c, cmap=cmap, alpha=0.4)

    for annotate_info in annotate_for:
        player = annotate_info['player']
        mask = z_info.Player == player

        for xy in z[mask]:
            plt.annotate(player, xy=xy)

    plt.show()

def show_player_clusters(mixture_probs, info, top_n=10):
    k = mixture_probs.shape[1]
    for i in range(k):
        top_idxs = np.argsort(-mixture_probs[:, i])[:top_n]
        top_players = info.Player.iloc[top_idxs]
        print("Cluster {}".format(i+1))
        for player in top_players:
            print("\t{}".format(player))
