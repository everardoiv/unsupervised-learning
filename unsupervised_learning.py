import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from scipy.stats import kurtosis, kurtosistest


def plot_curves(x, Y, xlabel, ylabel, curve_labels, name, save_name, flag=False, dotline=0, line_label='',
                show100=False):
    print(x)

    colors = plt.cm.rainbow(np.linspace(1, 0, len(Y)))

    plt.figure()
    plt.title(name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if (flag):
        plt.xticks(x, x)

    if (dotline!=0):
        tmp = np.zeros(len(x))
        tmp[:] = dotline
        print(tmp)
        plt.plot(x, tmp, color='black', label=line_label, lw=0.7, ls='dashed')
        tmp100 = np.zeros(len(x))
        tmp100[:] = 100
        if (show100):
            plt.plot(x, tmp100, color='black', label='train accuracy of raw data', lw=0.7, ls='dotted')

    for (y, label, c) in zip(Y, curve_labels, colors):
        plt.plot(x, y, color=c, label=label, lw=2.0)

    plt.legend(loc='best')
    plt.savefig(save_name)

    return plt


def test_one(data, X, y, min_n, max_n, title, sample_size):
    
    silhouette_score_zeros = np.zeros(max_n - min_n + 1)
    silhouette_score_ones = np.zeros(max_n - min_n + 1)

    ari_score_zeros = np.zeros(max_n - min_n + 1)
    ari_score_ones = np.zeros(max_n - min_n + 1)
    
    ami_score_zeros = np.zeros(max_n - min_n + 1)
    ami_score_ones = np.zeros(max_n - min_n + 1)

    time_zero = np.zeros(max_n - min_n + 1)
    time_ones = np.zeros(max_n - min_n + 1)

    for i in range(min_n, max_n + 1):
        baseline_index = i - min_n
        print("Testing KMeans of %d" %(i))
        start = time()
        km = KMeans(n_clusters=i, random_state=42)
        bench_k_means(km, name='k-means-' + str(i), data=data, labels=y, sample_size=sample_size)
        y_hat = km.fit_predict(X)
        time_zero[baseline_index] = time() - start
        silhouette_score_zeros[baseline_index] = metrics.silhouette_score(X, y_hat, metric='euclidean',
                                                                          sample_size=sample_size)
        ari_score_zeros[baseline_index] = metrics.adjusted_rand_score(y, y_hat)
        ami_score_zeros[baseline_index] = metrics.adjusted_mutual_info_score(y, y_hat)

        print('KM: n_cluster = %d | silhouette = %f' % (i, silhouette_score_zeros[baseline_index]))
        print('KM: n_cluster = %d | ARI = %f' % (i, ari_score_zeros[baseline_index]))
        print('KM: n_cluster = %d | AMI =%f' % (i, ami_score_zeros[baseline_index]))

        start = time()
        em = GaussianMixture(n_components=i, random_state=42, reg_covar=1.0e-4)
        em.fit(X)
        y_hat = em.predict(X)
        time_ones[baseline_index] = time() - start
        silhouette_score_ones[baseline_index] = metrics.silhouette_score(X, y_hat,
                                                                         metric='euclidean', sample_size=sample_size)
        ari_score_ones[baseline_index] = metrics.adjusted_rand_score(y, y_hat)
        ami_score_ones[baseline_index] = metrics.adjusted_mutual_info_score(y, y_hat)

        print('EM: n_cluster = %d | silhouette = %f' % (i, silhouette_score_ones[baseline_index]))
        print('EM: n_cluster = %d | ARI = %f' % (i, ari_score_ones[baseline_index]))
        print('EM: n_cluster = %d | AMI = %f' % (i, ami_score_ones[baseline_index]))

    plot_curves(np.arange(min_n, max_n + 1), (silhouette_score_zeros, silhouette_score_ones), 'k', 'Silhouette Score',
                ('k-means', 'EM'), title + ':\nsilhouette score curves for different clusters',
                'figures/test1_' + title + 'sc.png', flag=True)
    plot_curves(np.arange(min_n, max_n + 1), (ari_score_zeros, ari_score_ones), 'k', 'ARI Score', ('k-means', 'EM'),
                title + ':\nARI scores curves for different clusters', 'figures/test1_' + title + '_ari.png',
                flag=True)
    plot_curves(np.arange(min_n, max_n + 1), (ami_score_zeros, ami_score_ones), 'k', 'AMI Score', ('k-means', 'EM'),
                title + ':\nAMI score curves different clusters', 'figures/test1_' + title + '_ami.png', flag=True)

    print('times 0')
    print(time_zero)
    print('times 1')
    print(time_ones)
    arange = np.arange(max_n - min_n + 1)
    plot_curves(arange, (time_zero, time_ones), 'keep components', 'time(s)',
                ('KMeans', 'EM'),
                'Total running time for different features reduction algorithms',
                'figures/test1_' + title + '_time.png', flag=True, dotline=None, line_label='raw data')


def test_two(data, X, y, max_number_of_components, title):
    scaler = StandardScaler()
    scaler.fit(X.values)

    error_0 = np.zeros(max_number_of_components-1)
    error_1 = np.zeros(max_number_of_components-1)
    error_2 = np.zeros(max_number_of_components-1)
    error_3 = np.zeros(max_number_of_components-1)

    for i in range(1, max_number_of_components):
        print('# of Components = %d' % i)
        print(82 * '-')
        print('PCA:')
        print(82 * '-')
        pca = PCA(n_components=i, random_state=42)
        output_X_pca = pca.fit_transform(X.values)
        score = pca.score(X.values)

        eigen_values = np.linalg.eigvals(pca.get_covariance())
        explained_var = sum(pca.explained_variance_ratio_)
        R = scaler.inverse_transform(pca.inverse_transform(output_X_pca))  # Reconstruction
        R_error = sum(map(np.linalg.norm, R - X.values))
        print('Eigenvalues:')
        print('{}'.format(eigen_values))
        print('Explained variance (%): {}'.format(explained_var))
        print('Reconstruction error: {}'.format(R_error))
        print('Score: ' + str(score))

        error_0[i-1] = R_error

        # ------ICA
        print(82 * '-')
        print('ICA:')
        print(82 * '-')
        ica = FastICA(n_components=i, random_state=42)
        output_X_ica = ica.fit_transform(X.values)

        R_ica = scaler.inverse_transform(ica.inverse_transform(output_X_ica))
        print(kurtosis(R_ica))
        print(kurtosistest(R_ica))
        R_error_ica = sum(map(np.linalg.norm, R_ica - X.values))
        print('Reconstruction error: {}'.format(R_error_ica))
        error_1[i - 1] = R_error_ica

        # -------RP
        print(82 * '-')
        print('RP: ')
        print(82 * '-')
        rp = GaussianRandomProjection(n_components=i)
        output_X_rp = rp.fit_transform(X.values)

        inv = np.linalg.pinv(rp.components_)
        R_rp = scaler.inverse_transform(np.dot(output_X_rp, inv.T))  # Reconstruction
        R_error_rp = sum(map(np.linalg.norm, R_rp - X.values))
        print('Reconstruction error: {}'.format(R_error_rp))
        error_2[i - 1] = R_error_rp

        # -------SVD
        print(82 * '-')
        print('SVD: ')
        print(82 * '-')
        svd = TruncatedSVD(n_components=i)
        output_X_svd = svd.fit_transform(X.values)

        R_svd = scaler.inverse_transform(svd.inverse_transform(output_X_svd))  # Reconstruction
        R_error_svd = sum(map(np.linalg.norm, R_svd - X.values))
        print('Reconstruction error: {}'.format(R_error_svd))
        error_3[i - 1] = R_error_svd
        print()

    models = [X, X_scaled, output_X_ica, output_X_pca]
    names = ['Observations (mixed signal)',
             'Scaled X',
             'ICA recovered signals',
             'PCA recovered signals']
    colors = ['red', 'steelblue', 'orange']

    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(4, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

    plt.tight_layout()
    plt.show()

    plot_curves(np.arange(1, max_number_of_components), (error_0, error_1, error_2, error_3), 'components kept',
                'Reconstruction Error', ('PCA ' + str(error_0.mean()), 'ICA ' + str(error_1.mean()),
                                         'Random Projection ' + str(error_2.mean()), 'SVD ' + str(error_3.mean())),
                'Test 2 for ' + title + ':\nReconstruction Error for components analysis',
                'figures/test2_' + title + '.png')


def test_three(data, X, y, n_clusters, n_features, title):

    projections = (PCA, FastICA, GaussianRandomProjection, TruncatedSVD)

    arange = np.arange(1, n_features, 2)

    score0 = np.zeros(len(arange))
    score1 = np.zeros(len(arange))
    score2 = np.zeros(len(arange))
    score3 = np.zeros(len(arange))

    scores = (score0, score1, score2, score3)

    index = -1

    print('Interval: ' + str(arange))
    km = KMeans(n_clusters=n_clusters, random_state=10)
    output_y = km.fit_predict(X.values)
    score = metrics.adjusted_rand_score(y, output_y)
    print('Score: ' + str(score))

    print('-' * 20 + 'start KM:\n')
    for i in arange:
        index = index + 1
        for (pro, sco) in zip(projections, scores):
            projection = pro(n_components=i)
            new_X = projection.fit_transform(X.values)

            km = KMeans(n_clusters=n_labels)
            output_y = km.fit_predict(new_X)
            # sco[index] = metrics.adjusted_mutual_info_score(y, output_y)
            sco[index] = metrics.adjusted_rand_score(y, output_y)
            # sco[index] = metrics.v_measure_score(y, output_y)

    plot_curves(arange, (score0, score1, score2, score3), 'keep components', 'ARI score',
                ('PCA', 'ICA', 'Random Projection', 'SVD'),
                title + ':\nARI scores for different projections in k-means',
                'figures/test3_' + title + '_km.png', flag=True, dotline=score, line_label='raw data')

    index = -1
    print('-' * 20 + 'start EM:')
    for j in arange:
        index = index + 1
        for (pro, sco) in zip(projections, scores):
            projection = pro(n_components=j)
            new_X = projection.fit_transform(X.values)

            em = GaussianMixture(n_components=n_labels, reg_covar=1.0e-4)
            em.fit(new_X)
            output_y = em.predict(new_X)
            # sco[index] = metrics.adjusted_mutual_info_score(y, output_y)
            sco[index] = metrics.adjusted_rand_score(y, output_y)
            # sco[index] = metrics.v_measure_score(y, output_y)

    em = GaussianMixture(n_components=n_labels, random_state=10, reg_covar=1.0e-4)
    em.fit(X.values)
    output_y = em.predict(X.values)
    score_em = metrics.adjusted_rand_score(y, output_y)
    print('Score: ' + str(score_em))
    plot_curves(arange, (score0, score1, score2, score3), 'keep components', 'ARI score',
                ('PCA', 'ICA', 'Random Projection', 'SVD'),
                title + ':\nARI scores for features reduction algorithms in EM',
                'figures/test3_' + title + '_em.png', flag=True, dotline=score_em, line_label='raw data')


def test_four(data, train_X, train_y, test_X, test_y, start, n_features, interval, title):
    projections = (PCA, FastICA, GaussianRandomProjection, TruncatedSVD)
    names = ('PCA', 'ICA', 'RP', 'SVD')
    arange = np.arange(start, n_features, interval)

    train0 = np.zeros(len(arange))
    train1 = np.zeros(len(arange))
    train2 = np.zeros(len(arange))
    train3 = np.zeros(len(arange))
    train_scores = (train0, train1, train2, train3)

    test0 = np.zeros(len(arange))
    test1 = np.zeros(len(arange))
    test2 = np.zeros(len(arange))
    test3 = np.zeros(len(arange))
    test_scores = (test0, test1, test2, test3)

    km_test0 = np.zeros(len(arange))
    km_test1 = np.zeros(len(arange))
    km_test2 = np.zeros(len(arange))
    km_test3 = np.zeros(len(arange))
    km_test_scores = (km_test0, km_test1, km_test2, km_test3)

    km_train0 = np.zeros(len(arange))
    km_train1 = np.zeros(len(arange))
    km_train2 = np.zeros(len(arange))
    km_train3 = np.zeros(len(arange))
    km_train_scores = (km_train0, km_train1, km_train2, km_train3)

    em_train0 = np.zeros(len(arange))
    em_train1 = np.zeros(len(arange))
    em_train2 = np.zeros(len(arange))
    em_train3 = np.zeros(len(arange))
    em_train_scores = (em_train0, em_train1, em_train2, em_train3)

    em_test0 = np.zeros(len(arange))
    em_test1 = np.zeros(len(arange))
    em_test2 = np.zeros(len(arange))
    em_test3 = np.zeros(len(arange))
    em_test_scores = (em_test0, em_test1, em_test2, em_test3)

    time0 = np.zeros(len(arange))
    time1 = np.zeros(len(arange))
    time2 = np.zeros(len(arange))
    time3 = np.zeros(len(arange))

    times = (time0, time1, time2, time3)

    index = -1

    for i in arange:
        index = index + 1
        print('')
        print('keep components %d:' % i)
        for (pro, train_s, test_s, km_train, km_test, em_train, em_test, t, name) in zip(
                projections, train_scores, test_scores, km_train_scores,
                km_test_scores, em_train_scores, em_test_scores, times, names):
            print('--------%s:' % name)
            projection = pro(n_components=i)
            projection.fit(train_X)
            train_new_X = projection.transform(train_X)
            test_new_X = projection.transform(test_X)

            start_time = time()

            NN = MLPClassifier()
            NN.fit(train_new_X, train_y)
            output_y = NN.predict(train_new_X)
            train_s[index] = metrics.accuracy_score(train_y, output_y) * 100
            output_y = NN.predict(test_new_X)
            test_s[index] = metrics.accuracy_score(test_y, output_y) * 100

            end_time = time()

            t[index] = end_time - start_time

            print('test 4: train_score=%.2f%%, test_score=%.2f%%, time=%.2f'
                  % (train_s[index], test_s[index], t[index]))

            cl = KMeans(n_clusters=10)
            cl.fit(train_new_X)
            test5_train_X = make_array(cl, train_new_X, i)

            NN = MLPClassifier()
            NN.fit(test5_train_X, train_y)
            output_y = NN.predict(test5_train_X)

            km_train[index] = metrics.accuracy_score(train_y, output_y) * 100

            test5_test_X = make_array(cl, test_new_X, i)
            output_y = NN.predict(test5_test_X)
            km_test[index] = metrics.accuracy_score(test_y, output_y) * 100

            cl = GaussianMixture(n_components=10, reg_covar=1.0e-4)
            cl.fit(train_new_X)
            test5_train_X = make_array(cl, train_new_X, i)

            NN = MLPClassifier()
            NN.fit(test5_train_X, train_y)
            output_y = NN.predict(test5_train_X)

            em_train[index] = metrics.accuracy_score(train_y, output_y) * 100

            test5_test_X = make_array(cl, test_new_X, i)
            output_y = NN.predict(test5_test_X)
            em_test[index] = metrics.accuracy_score(test_y, output_y) * 100

            print('KM: train_score=%.2f%%, test_score=%.2f%%' % (km_train[index], km_test[index]))
            print('EM: train_score=%.2f%%, test_score=%.2f%%' % (em_train[index], em_test[index]))

    start_time = time()
    NN = MLPClassifier()
    NN.fit(train_X, train_y)
    output_y = NN.predict(train_X)
    acc = metrics.accuracy_score(train_y, output_y) * 100
    print('training accuracy = %.2f%%' % acc)
    output_y = NN.predict(test_X)
    acc = metrics.accuracy_score(test_y, output_y) * 100
    tot_time = time() - start_time
    print('testing accuracy = %.2f%%' % acc)
    print('total time = %fs' % tot_time)

    plot_curves(arange, (time0, time1, time2, time3), 'keep components', 'time(s)',
                ('PCA', 'ICA', 'Random Projection', 'SVD'),
                'Total running time for different features reduction algorithms',
                'figures/test4_' + title + '_time.png', flag=True, dotline=tot_time, line_label='raw data')

    plot_curves(arange, (train0, test0, km_train0, km_test0, em_train0, em_test0),
                'keep_components', 'accuracy(%)',
                ('train accuracy', 'test accuracy',
                 'train accuracy(k-means)', 'test accuracy(k-means)',
                 'train accuracy(EM)', 'test accuracy(EM)'),
                title + '\nTraining and testing accuracy for PCA',
                'figures/test4_' + title + '_PCA.png', flag=True, dotline=acc, line_label='test accuracy of raw data', show100=True)

    plot_curves(arange, (train1, test1, km_train1, km_test1, em_train1, em_test1),
                'keep_components', 'accuracy(%)',
                ('train accuracy', 'test accuracy',
                 'train accuracy(k-means)', 'test accuracy(k-means)',
                 'train accuracy(EM)', 'test accuracy(EM)'),
                title + '\nTraining and testing accuracy for ICA',
                'figures/test4_' + title + '_ICA.png', flag=True, dotline=acc, line_label='test accuracy of raw data', show100=True)

    plot_curves(arange, (train2, test2, km_train2, km_test2, em_train2, em_test2),
                'keep_components', 'accuracy(%)',
                ('train accuracy', 'test accuracy',
                 'train accuracy(k-means)', 'test accuracy(k-means)',
                 'train accuracy(EM)', 'test accuracy(EM)'),
                title + '\nTraining and testing accuracy for Random Projection',
                'figures/test4_' + title + '_RP.png', flag=True, dotline=acc, line_label='test accuracy of raw data', show100=True)

    plot_curves(arange, (train3, test3, km_train3, km_test3, em_train3, em_test3),
                'keep_components', 'accuracy(%)',
                ('train accuracy', 'test accuracy',
                 'train accuracy(k-means)', 'test accuracy(k-means)',
                 'train accuracy(EM)', 'test accuracy(EM)'),
                title + '\nTraining and testing accuracy for SVD',
                'figures/test4_' + title + '_SVD.png', flag=True, dotline=acc, line_label='test accuracy of raw data', show100=True)


def make_array(clustering, X, n_components):
    size = len(X)
    new_X = np.zeros((size, n_components + 1))
    new_X[:, 0:n_components] = X
    new_X[:, n_components] = clustering.predict(X)

    return new_X


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, random_state=42,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    plt.savefig('figures/' + title + '.png')

    return plt


def plot_validation_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, param_range=np.linspace(3, 18, dtype=int), param_name='n_clusters'):
    plt.figure()
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_scores, test_scores = validation_curve(estimator, X, y, n_jobs=n_jobs,
                                                 param_name=param_name, param_range=param_range, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot validation curve
    axes.grid()
    axes.fill_between(my_param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="b")
    axes.fill_between(my_param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="k")
    axes.plot(my_param_range, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(my_param_range, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    plt.savefig('figures/' + title + '.png')

    return plt


def plot_projection_graph(data, n_clusters, dataset, savename):

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(reduced_data)
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    print(len(xx))
    print(len(yy))
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the '+ dataset + ' (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('figures/Projection_' + savename)


def bench_k_means(estimator, name, data, labels, sample_size):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))


def split_the_data(df, column, stratify_bool):
    X = df.drop(columns=column, axis=1)
    y = df[column]

    # Stratify makes it so that the proportion of values in the sample in our test group will be the same as the
    # proportion of values provided to parameter stratify
    if stratify_bool:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=42,
                                                            stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=42)

    return X, y, X_train, X_test, y_train, y_test


def load_and_describe_data(file, exp, cv):
    # Load
    data = pd.read_csv(file)

    assert not data.isnull().values.any()

    # Describe the data
    print('%s - %s - %s' % (filename, exp, cv))
    print(data.describe())
    print()
    return data


def experiment_inputs():
    inputs = {
        'experiment': [
            'Credit Card',
            'Poker Hand'
        ],
        'data_sets_filenames': [
            'data/output/credit-card-data.csv',
            'data/output/sampled-poker-hand-data.csv'
        ],
        'num_cross_validation': [
            5,
            5
        ],
        'feature_label': [
            'default payment next month',
            'Poker Hand'
        ]
    }

    return inputs


if __name__ == '__main__':
    print("Starting Unsupervised Learning")
    inputs = experiment_inputs()

    for i, filename in enumerate(inputs['data_sets_filenames']):
        experiment = inputs['experiment'][i]
        cv = inputs['num_cross_validation'][i]
        target_feature = inputs['feature_label'][i]

        trim_experiment = experiment.replace(" ", "")
        data = load_and_describe_data(filename, experiment, cv)
        x_scaler = StandardScaler()
        X_scaled = data.drop(columns=target_feature, axis=1)
        data[list(X_scaled.columns)] = x_scaler.fit_transform(data.drop(columns=target_feature, axis=1))

        X, y, X_train, X_test, y_train, y_test = split_the_data(data, target_feature, True)

        pd.set_option('display.max_columns', 500)
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        print('# of Samples: ' + str(n_samples) + '| # of Features: ' + str(n_features) + '| # of Labels: ' + str(n_labels))

        estimator = KMeans(n_jobs=-1, random_state=42)
        print('Benchmark K-Means')
        print(82 * '_')
        print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\t\tAMI\tsilhouette')
        bench_k_means(KMeans(init='k-means++', n_clusters=n_labels, n_init=10), name="k-means++",
                      data=data, labels=y, sample_size=300)
        bench_k_means(KMeans(init='random', n_clusters=n_labels, n_init=10), name="random",
                      data=data, labels=y, sample_size=300)

        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X_train)
        correct = 0
        for j in range(len(X_train)):
            predict_me = np.array(X_train.iloc[j].astype(float))
            predict_me = predict_me.reshape(-1, len(predict_me))
            prediction = kmeans.predict(predict_me)
            if prediction[0] == y_train.iloc[j]:
                correct += 1

        print("Training Correct")
        print(correct/len(X_train))

        print(82 * '=')
        test_one(data, X, y, 2, 22, trim_experiment, 5000)
        print(82 * '=')

        # print('Validation Curve')
        # my_param_range = np.linspace(3, 18, dtype=int)
        # plt = plot_validation_curve(estimator, 'KMeans Validation Curve' + str(trim_experiment), X, y=None,
        #                             axes=None, ylim=None, cv=cv, n_jobs=4, param_range=my_param_range,
        #                             param_name='n_clusters')
        #
        # print('Learning Curve')
        # fig, axes = plt.subplots(1, 3, figsize=(21, 15))
        # plt = plot_learning_curve(estimator, 'KMeans Learning Curve' + str(trim_experiment), X, y=None,
        #                           axes=axes, ylim=None, cv=cv, n_jobs=4)

        print(82 * '=')
        test_two(data, X, y, n_features, trim_experiment)
        print(82 * '=')

        print(82 * '=')
        test_three(data, X, y, 4, n_features, trim_experiment)
        print(82 * '=')

        print(82 * '=')
        test_three(data, X, y, 7, n_features, trim_experiment + '_7 labels')
        print(82 * '=')

        print(82 * '=')
        test_three(data, X, y, n_labels, n_features, trim_experiment + '_n_labels')
        print(82 * '=')

        print(82 * '=')
        test_four(data, X_train, y_train, X_test, y_test, 1, n_features, 2, trim_experiment)
        print(82 * '=')

        # Has to be the best number of clusters
        print(82 * '=')
        plot_projection_graph(X_train, 2, trim_experiment, '2_Clusters_' + trim_experiment + '.png')
        print(82 * '=')

        print(82 * '=')
        plot_projection_graph(X_train, 4, trim_experiment, '4_Clusters_' + trim_experiment + '.png')
        print(82 * '=')

        print(82 * '=')
        plot_projection_graph(X_test, 4, trim_experiment, '4_Clusters_test' + trim_experiment + '.png')
        print(82 * '=')

        print(82 * '=')
        plot_projection_graph(X_train, 10, trim_experiment, '10_Clusters_' + trim_experiment + '.png')
        print(82 * '=')

        print(82 * '=')
        plot_projection_graph(X_train, 12, trim_experiment, '12_Clusters_' + trim_experiment + '.png')
        print(82 * '=')

        print(82 * '=')
        plot_projection_graph(X_train, 20, trim_experiment, '20_Clusters_' + trim_experiment + '.png')
        print(82 * '=')
        print()
        plt.show()
