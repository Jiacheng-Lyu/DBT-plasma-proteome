from sklearn.decomposition import PCA


def core_PCA(df, **kwargs):
    pca = PCA(**kwargs)
    df_pca = pca.fit_transform(df)
    return df_pca, pca.explained_variance_ratio_