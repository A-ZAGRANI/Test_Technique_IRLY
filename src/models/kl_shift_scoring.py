# Auteur : Mohamed Amine ZAGRANI

from scipy.stats import entropy

def compute_kl_divergence(train_data, test_data, feature):
    train_dist = train_data[feature].value_counts(normalize=True)
    test_dist = test_data[feature].value_counts(normalize=True)

    all_indices = train_dist.index.union(test_dist.index)
    train_dist = train_dist.reindex(all_indices, fill_value=0)
    test_dist = test_dist.reindex(all_indices, fill_value=0)

    return entropy(train_dist, test_dist)
