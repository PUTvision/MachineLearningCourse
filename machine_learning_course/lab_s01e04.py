import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from sklearn import datasets
from sklearn import ensemble
from sklearn import svm
from sklearn import impute
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn import pipeline, cluster
from sklearn import decomposition, manifold
from sklearn import neighbors

# import gap_statistic

from lab_s01_utils import print_function_name, plot_iris


def todo_1():
    print_function_name()

    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)
    print(X)

    print(X.info())
    print(X.describe())

    # X, y = datasets.fetch_openml('diabetes', return_X_y=True)
    # print(X)
    # print(y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_ii = X_train.copy()

    plt.figure()
    X_train.boxplot()
    X_train.hist()

    # plt.figure()
    # sns.boxplot(x=X_train['mass'])
    # plt.show()

    imputer_mass = impute.SimpleImputer(missing_values=0.0, strategy='mean')
    # print(X[:, 5].reshape(-1, 1))
    # imputer.fit(X[:, 5].reshape(-1, 1))
    # X[:, 5] = imputer.transform(X[:, 5].reshape(-1, 1))
    imputer_skin = impute.SimpleImputer(missing_values=0.0, strategy='mean')

    X_train[['mass']] = imputer_mass.fit_transform(X_train[['mass']]) #.values.reshape(-1, 1))
    X_train[['skin']] = imputer_skin.fit_transform(X_train[['skin']])

    X_test[['mass']] = imputer_mass.transform(X_test[['mass']])
    X_test[['skin']] = imputer_skin.transform(X_test[['skin']])

    # # X_train.boxplot()
    # X_train.hist(bins=20)

    imputer_ii = impute.KNNImputer(n_neighbors=2, missing_values=0.0)

    X_train_ii[['mass']] = imputer_ii.fit_transform(X_train_ii[['mass']])
    X_train_ii[['skin']] = imputer_ii.fit_transform(X_train_ii[['skin']])

    # X_train_2.boxplot()
    # X_train_ii.hist(bins=20)
    # plt.show()

    df_mass = X_train[['mass']]
    print(df_mass.head(5))
    plt.show()


def todo_1_part_2():
    X_train_isolation = X_train.values
    X_train_isolation = X_train_isolation[:, [1, 5]]
    X_test_isolation = X_test.values
    X_test_isolation = X_test_isolation[:, [1, 5]]

    isolation_forest = ensemble.IsolationForest(contamination=0.05)
    isolation_forest.fit(X_train_isolation)
    y_predicted_outliers = isolation_forest.predict(X_test_isolation)
    print(y_predicted_outliers)

    print(X_test_isolation)
    plot_iris(X_test_isolation, y_predicted_outliers)
    plt.show()

    clf_svm = svm.SVC(random_state=42)
    clf_svm.fit(X_train, y_train)
    y_predicted_svm = clf_svm.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted_svm))

    clf_rf = ensemble.RandomForestClassifier(random_state=42)
    clf_rf.fit(X_train, y_train)
    y_predicted_rf = clf_rf.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted_rf))

    importances = clf_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

    # y[y == 'tested_positive'] = 1
    # y[y == 'tested_negative'] = 0
    # print(y)


def todo_2():
    print_function_name()

    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)

    plt.figure()
    X.boxplot()
    X.hist(bins=20)
    plt.show()

    plt.figure()
    sns.boxplot(x=X['mass'])
    plt.figure()
    sns.histplot(data=X, x='mass')
    plt.show()


def todo_3():
    print_function_name()

    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)

    plot_iris(X[['mass', 'plas']].to_numpy(), title='mass vs plas')
    plt.show()


def todo_4():
    print_function_name()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)
    print(X.describe())

    X_zscore = X.apply(scipy.stats.zscore)
    print(X_zscore.head(5))

    X_filtered = X[X_zscore < 3]
    print(X_filtered.describe())


def todo_5():
    # Load the Pima Indians Diabetes Database - https://www.openml.org/d/37
    
    print_function_name()
    
    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Extract only plas and mass features
    X_train_features = X_train[['plas', 'mass']].values
    X_test_features = X_test[['plas', 'mass']].values
    
    # Test different contamination parameters
    contamination_values = [0.05, 0.1, 0.15]
    
    for contamination in contamination_values:
        print(f"\nIsolationForest with contamination={contamination}")
        
        # Train IsolationForest
        isolation_forest = ensemble.IsolationForest(
            contamination=contamination, random_state=42
        )
        isolation_forest.fit(X_train_features)
        y_pred_isolation = isolation_forest.predict(X_test_features)
        
        # Visualize predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(
            X_test_features[y_pred_isolation == 1, 0],
            X_test_features[y_pred_isolation == 1, 1],
            c='blue', label='Inliers', alpha=0.6
        )
        plt.scatter(
            X_test_features[y_pred_isolation == -1, 0],
            X_test_features[y_pred_isolation == -1, 1],
            c='red', label='Outliers', alpha=0.6
        )
        
        # Create decision boundary using contour plot
        h = 0.5  # Step size in the mesh
        x_min, x_max = X_test_features[:, 0].min() - 1, X_test_features[:, 0].max() + 1
        y_min, y_max = X_test_features[:, 1].min() - 1, X_test_features[:, 1].max() + 1
        
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )
        
        Z = isolation_forest.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0], colors=['red'])
        plt.contourf(xx, yy, Z, alpha=0.1, levels=[0, 1], colors=['blue'])
        
        plt.xlabel('plas')
        plt.ylabel('mass')
        plt.title(f'IsolationForest (contamination={contamination})')
        plt.legend()
    
    # Test LocalOutlierFactor method
    # print("\n\nLocalOutlierFactor with n_neighbors=20")
    # lof = neighbors.LocalOutlierFactor(n_neighbors=20)
    # y_pred_lof = lof.fit_predict(X_test_features)
    #
    # plt.figure(figsize=(10, 6))
    # plt.scatter(
    #     X_test_features[y_pred_lof == 1, 0],
    #     X_test_features[y_pred_lof == 1, 1],
    #     c='green', label='Inliers', alpha=0.6
    # )
    # plt.scatter(
    #     X_test_features[y_pred_lof == -1, 0],
    #     X_test_features[y_pred_lof == -1, 1],
    #     c='orange', label='Outliers', alpha=0.6
    # )
    #
    # Create decision boundary for LOF using contour
    # h = 0.5
    # x_min, x_max = X_test_features[:, 0].min() - 1, X_test_features[:, 0].max() + 1
    # y_min, y_max = X_test_features[:, 1].min() - 1, X_test_features[:, 1].max() + 1
    #
    # xx, yy = np.meshgrid(
    #     np.arange(x_min, x_max, h),
    #     np.arange(y_min, y_max, h)
    # )
    #
    # Z_lof = lof.fit_predict(np.c_[xx.ravel(), yy.ravel()])
    # Z_lof = Z_lof.reshape(xx.shape)
    #
    # plt.contourf(xx, yy, Z_lof, alpha=0.3, levels=[-1, 0], colors=['orange'])
    # plt.contourf(xx, yy, Z_lof, alpha=0.1, levels=[0, 1], colors=['green'])
    #
    # plt.xlabel('plas')
    # plt.ylabel('mass')
    # plt.title('LocalOutlierFactor (n_neighbors=20)')
    # plt.legend()

    plt.show()

def todo_6():
    # Analyze https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html
    # For the Pima Indians Diabetes Database example, visualize the area where values are considered as inliers
    # Comparison of multiple anomaly detection methods
    
    print_function_name()
    
    from sklearn import covariance
    
    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Extract only plas and mass features
    X_train_features = X_train[['plas', 'mass']].values
    X_test_features = X_test[['plas', 'mass']].values
    
    # Define the anomaly detection methods
    methods = {
        'IsolationForest': ensemble.IsolationForest(contamination=0.1, random_state=42),
        'LocalOutlierFactor': neighbors.LocalOutlierFactor(n_neighbors=20),
        # 'OneClassSVM': svm.OneClassSVM(nu=0.1, kernel='rbf', gamma='auto'),
        'EllipticEnvelope': covariance.EllipticEnvelope(contamination=0.1, random_state=42)
    }
    
    # Create a mesh to plot decision boundaries
    h = 0.5  # Step size in the mesh
    x_min, x_max = X_test_features[:, 0].min() - 1, X_test_features[:, 0].max() + 1
    y_min, y_max = X_test_features[:, 1].min() - 1, X_test_features[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    
    # Create subplots for each method
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, (method_name, method) in enumerate(methods.items()):
        print(f"\n{method_name}")
        
        # Fit the method on training data
        if method_name == 'LocalOutlierFactor':
            method.fit(X_train_features)
            y_pred_train = method.fit_predict(X_train_features)
            y_pred_test = method.fit_predict(X_test_features)
        else:
            method.fit(X_train_features)
            y_pred_train = method.predict(X_train_features)
            y_pred_test = method.predict(X_test_features)
        
        # Predict on mesh
        if method_name == 'LocalOutlierFactor':
            Z = method.fit_predict(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = method.predict(np.c_[xx.ravel(), yy.ravel()])
        
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax = axes[idx]
        ax.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0], colors=['red'])
        ax.contourf(xx, yy, Z, alpha=0.1, levels=[0, 1], colors=['blue'])
        
        # Plot inliers and outliers
        ax.scatter(
            X_test_features[y_pred_test == 1, 0],
            X_test_features[y_pred_test == 1, 1],
            c='blue', marker='o', label='Inliers', alpha=0.7, edgecolors='k'
        )
        ax.scatter(
            X_test_features[y_pred_test == -1, 0],
            X_test_features[y_pred_test == -1, 1],
            c='red', marker='X', label='Outliers', alpha=0.7, edgecolors='k', s=100
        )
        
        ax.set_xlabel('plas')
        ax.set_ylabel('mass')
        ax.set_title(method_name)
        ax.legend()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Print statistics
        n_outliers_test = (y_pred_test == -1).sum()
        print(f"  Outliers detected in test set: {n_outliers_test} / {len(y_pred_test)}")
    
    plt.tight_layout()
    plt.show()


def todo_7():
    # Reflect on why cross-validation is used in this task
    # Test GridSearchCV and RandomizedSearchCV for hyperparameter tuning
    # of Decision Tree and SVM on a small dataset (iris)
    # Visualize the results
    # Save the best model to a file
    
    print_function_name()
    
    import joblib
    from sklearn import tree
    
    # Load iris dataset
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # ==================== GridSearchCV for SVM ====================
    print("\n" + "="*60)
    print("GridSearchCV for SVM")
    print("="*60)
    
    svm_parameters = {
        'kernel': ('linear', 'rbf', 'poly'),
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    
    svm_clf = model_selection.GridSearchCV(
        svm.SVC(),
        svm_parameters,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    svm_clf.fit(X_train, y_train)
    
    print(f"\nBest SVM parameters: {svm_clf.best_params_}")
    print(f"Best SVM CV score: {svm_clf.best_score_:.4f}")
    print(f"SVM test score: {svm_clf.score(X_test, y_test):.4f}")
    
    # Create pivot table for SVM results (kernel vs C)
    svm_results_df = pd.DataFrame(svm_clf.cv_results_)
    svm_pivot = pd.pivot_table(
        svm_results_df,
        values='mean_test_score',
        index='param_kernel',
        columns='param_C'
    )
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(svm_pivot, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'Mean CV Score'})
    plt.title('GridSearchCV Results: SVM (Kernel vs C)')
    plt.xlabel('C parameter')
    plt.ylabel('Kernel')
    
    # ==================== GridSearchCV for Decision Tree ====================
    print("\n" + "="*60)
    print("GridSearchCV for Decision Tree")
    print("="*60)
    
    dt_parameters = {
        'max_depth': [2, 3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    
    dt_clf = model_selection.GridSearchCV(
        tree.DecisionTreeClassifier(random_state=42),
        dt_parameters,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    dt_clf.fit(X_train, y_train)
    
    print(f"\nBest Decision Tree parameters: {dt_clf.best_params_}")
    print(f"Best Decision Tree CV score: {dt_clf.best_score_:.4f}")
    print(f"Decision Tree test score: {dt_clf.score(X_test, y_test):.4f}")
    
    # Create pivot table for Decision Tree results (max_depth vs min_samples_split)
    dt_results_df = pd.DataFrame(dt_clf.cv_results_)
    dt_pivot = pd.pivot_table(
        dt_results_df,
        values='mean_test_score',
        index='param_max_depth',
        columns='param_min_samples_split'
    )
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(dt_pivot, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'Mean CV Score'})
    plt.title('GridSearchCV Results: Decision Tree (Max Depth vs Min Samples Split)')
    plt.xlabel('Min Samples Split')
    plt.ylabel('Max Depth')
    
    # ==================== RandomizedSearchCV for comparison ====================
    print("\n" + "="*60)
    print("RandomizedSearchCV for SVM (for comparison)")
    print("="*60)
    
    svm_random_clf = model_selection.RandomizedSearchCV(
        svm.SVC(),
        svm_parameters,
        n_iter=10,
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    svm_random_clf.fit(X_train, y_train)
    
    print(f"\nBest SVM parameters (RandomizedSearchCV): {svm_random_clf.best_params_}")
    print(f"Best SVM CV score (RandomizedSearchCV): {svm_random_clf.best_score_:.4f}")
    print(f"SVM test score (RandomizedSearchCV): {svm_random_clf.score(X_test, y_test):.4f}")
    
    # ==================== Save best models ====================
    print("\n" + "="*60)
    print("Saving best models")
    print("="*60)
    
    # Determine which model is best overall
    if svm_clf.best_score_ >= dt_clf.best_score_:
        best_model = svm_clf
        best_model_name = "GridSearchCV_SVM_best_model.pkl"
    else:
        best_model = dt_clf
        best_model_name = "GridSearchCV_DecisionTree_best_model.pkl"
    
    joblib.dump(best_model, best_model_name)
    print(f"\nBest model saved as: {best_model_name}")
    print(f"Best model type: {type(best_model.best_estimator_).__name__}")
    print(f"Best model CV score: {best_model.best_score_:.4f}")
    
    # Save all models for reference
    joblib.dump(svm_clf, "GridSearchCV_SVM_full_results.pkl")
    joblib.dump(dt_clf, "GridSearchCV_DecisionTree_full_results.pkl")
    joblib.dump(svm_random_clf, "RandomizedSearchCV_SVM_full_results.pkl")
    
    print("\nAll models saved:")
    print("  - GridSearchCV_SVM_full_results.pkl")
    print("  - GridSearchCV_DecisionTree_full_results.pkl")
    print("  - RandomizedSearchCV_SVM_full_results.pkl")
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    models_scores = {
        'GridSearchCV SVM': svm_clf.best_score_,
        'GridSearchCV Decision Tree': dt_clf.best_score_,
        'RandomizedSearchCV SVM': svm_random_clf.best_score_
    }
    
    bars = plt.bar(models_scores.keys(), models_scores.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('Mean CV Score')
    plt.title('Comparison of Hyperparameter Search Methods')
    plt.ylim([0.9, 1.0])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def todo_10():
    # Model Ensembling: Voting and Stacking
    # Combine multiple classifiers to improve predictions
    # Understand hard voting (class) vs soft voting (probabilities)
    # Compare with individual base learners
    
    print_function_name()
    
    import joblib
    from sklearn import linear_model
    from sklearn import tree
    
    # Load Breast Cancer dataset (binary classification, more realistic medical data)
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Normalize features (important for SVM and KNN)
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define diverse base classifiers
    clf_svm = svm.SVC(kernel='rbf', C=1, probability=True, random_state=42)
    clf_dt = tree.DecisionTreeClassifier(max_depth=10, random_state=42)
    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    clf_lr = linear_model.LogisticRegression(max_iter=5000, random_state=42)
    
    print("\n" + "="*70)
    print("Training individual base classifiers on Breast Cancer Dataset")
    print("="*70)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, 2 classes")
    
    # Train base classifiers
    clf_svm.fit(X_train, y_train)
    clf_dt.fit(X_train, y_train)
    clf_knn.fit(X_train, y_train)
    clf_lr.fit(X_train, y_train)
    
    # Evaluate individual classifiers
    print(f"SVM test score: {clf_svm.score(X_test, y_test):.4f}")
    print(f"Decision Tree test score: {clf_dt.score(X_test, y_test):.4f}")
    print(f"KNN test score: {clf_knn.score(X_test, y_test):.4f}")
    print(f"Logistic Regression test score: {clf_lr.score(X_test, y_test):.4f}")
    
    # ==================== Display Probabilities for Sample Predictions ====================
    print("\n" + "="*70)
    print("Class Probabilities for Sample Test Predictions")
    print("="*70)
    
    n_samples_to_show = 5
    sample_indices = np.random.choice(len(X_test), n_samples_to_show, replace=False)
    
    # Create figure for probability visualizations
    fig, axes = plt.subplots(n_samples_to_show, 1, figsize=(10, 2*n_samples_to_show))
    if n_samples_to_show == 1:
        axes = [axes]
    
    for idx, sample_idx in enumerate(sample_indices):
        X_sample = X_test[sample_idx:sample_idx+1]
        y_true = y_test[sample_idx]
        
        # Get probabilities from classifiers that support predict_proba
        proba_svm = clf_svm.predict_proba(X_sample)[0]
        proba_dt = clf_dt.predict_proba(X_sample)[0]
        proba_knn = clf_knn.predict_proba(X_sample)[0]
        proba_lr = clf_lr.predict_proba(X_sample)[0]
        
        # Make predictions
        pred_svm = clf_svm.predict(X_sample)[0]
        pred_dt = clf_dt.predict(X_sample)[0]
        pred_knn = clf_knn.predict(X_sample)[0]
        pred_lr = clf_lr.predict(X_sample)[0]
        
        # Print textual representation
        true_label = "Malignant" if y_true == 0 else "Benign"
        print(f"\nSample {idx+1} (True label: {y_true} - {true_label})")
        print(f"  SVM prediction: {pred_svm}, probabilities: malignant={proba_svm[0]:.4f}, benign={proba_svm[1]:.4f}")
        print(f"  Decision Tree prediction: {pred_dt}, probabilities: malignant={proba_dt[0]:.4f}, benign={proba_dt[1]:.4f}")
        print(f"  KNN prediction: {pred_knn}, probabilities: malignant={proba_knn[0]:.4f}, benign={proba_knn[1]:.4f}")
        print(f"  Logistic Regression prediction: {pred_lr}, probabilities: malignant={proba_lr[0]:.4f}, benign={proba_lr[1]:.4f}")
        
        # Visualize probabilities for both classes
        classes = ['Malignant', 'Benign']
        ax = axes[idx]
        
        x_pos = np.arange(len(classes))
        width = 0.2
        
        ax.bar(x_pos - 1.5*width, proba_svm, width, label='SVM', alpha=0.8)
        ax.bar(x_pos - 0.5*width, proba_dt, width, label='Decision Tree', alpha=0.8)
        ax.bar(x_pos + 0.5*width, proba_knn, width, label='KNN', alpha=0.8)
        ax.bar(x_pos + 1.5*width, proba_lr, width, label='Logistic Regression', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Probability')
        ax.set_title(f'Sample {idx+1} - True: {true_label} (SVM:{pred_svm}, DT:{pred_dt}, KNN:{pred_knn}, LR:{pred_lr})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # ==================== VotingClassifier - Hard Voting ====================
    print("\n" + "="*70)
    print("VotingClassifier - Hard Voting (class-based)")
    print("="*70)
    
    voting_hard = ensemble.VotingClassifier(
        estimators=[
            ('svm', clf_svm),
            ('dt', clf_dt),
            ('knn', clf_knn),
            ('lr', clf_lr)
        ],
        voting='hard'
    )
    voting_hard.fit(X_train, y_train)
    
    hard_score = voting_hard.score(X_test, y_test)
    print(f"Hard Voting test score: {hard_score:.4f}")
    
    # ==================== VotingClassifier - Soft Voting ====================
    print("\n" + "="*70)
    print("VotingClassifier - Soft Voting (probability-based)")
    print("="*70)
    
    voting_soft = ensemble.VotingClassifier(
        estimators=[
            ('svm', clf_svm),
            ('dt', clf_dt),
            ('knn', clf_knn),
            ('lr', clf_lr)
        ],
        voting='soft'
    )
    voting_soft.fit(X_train, y_train)
    
    soft_score = voting_soft.score(X_test, y_test)
    print(f"Soft Voting test score: {soft_score:.4f}")
    
    # ==================== StackingClassifier ====================
    print("\n" + "="*70)
    print("StackingClassifier (meta-learner: Logistic Regression)")
    print("="*70)
    
    stacking_clf = ensemble.StackingClassifier(
        estimators=[
            ('svm', svm.SVC(kernel='rbf', C=1, probability=True, random_state=42)),
            ('dt', tree.DecisionTreeClassifier(max_depth=10, random_state=42)),
            ('knn', neighbors.KNeighborsClassifier(n_neighbors=5)),
            ('lr', linear_model.LogisticRegression(max_iter=5000, random_state=42))
        ],
        final_estimator=linear_model.LogisticRegression(max_iter=5000, random_state=42),
        cv=5,
        stack_method='predict_proba'
    )
    stacking_clf.fit(X_train, y_train)
    
    stacking_score = stacking_clf.score(X_test, y_test)
    print(f"Stacking test score: {stacking_score:.4f}")
    
    # ==================== Comparison ====================
    print("\n" + "="*70)
    print("Model Comparison")
    print("="*70)
    
    results = {
        'SVM': clf_svm.score(X_test, y_test),
        'Decision Tree': clf_dt.score(X_test, y_test),
        'KNN': clf_knn.score(X_test, y_test),
        'Logistic Regression': clf_lr.score(X_test, y_test),
        'Voting (Hard)': hard_score,
        'Voting (Soft)': soft_score,
        'Stacking': stacking_score
    }
    
    for model_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:25s}: {score:.4f}")
    
    # ==================== Visualization of Results ====================
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    scores = list(results.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    bars = plt.bar(models, scores, color=colors[:len(models)])
    plt.ylabel('Test Score (Accuracy)')
    plt.title('Ensemble Methods Comparison on Breast Cancer Dataset')
    plt.ylim([0.9, 1.0])
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # ==================== Visualize Voting Results ====================
    print("\n" + "="*70)
    print("Analyzing Voting Ensemble Decisions")
    print("="*70)
    
    # For a few test samples, show voting counts
    n_voting_samples = 5
    voting_indices = np.random.choice(len(X_test), n_voting_samples, replace=False)
    
    print(f"\nVoting decisions for {n_voting_samples} random test samples:")
    for sample_idx in voting_indices:
        X_sample = X_test[sample_idx:sample_idx+1]
        y_true = y_test[sample_idx]
        
        # Get predictions from each base classifier
        pred_svm = clf_svm.predict(X_sample)[0]
        pred_dt = clf_dt.predict(X_sample)[0]
        pred_knn = clf_knn.predict(X_sample)[0]
        pred_lr = clf_lr.predict(X_sample)[0]
        
        # Get ensemble predictions
        pred_hard = voting_hard.predict(X_sample)[0]
        pred_soft = voting_soft.predict(X_sample)[0]
        pred_stack = stacking_clf.predict(X_sample)[0]
        
        true_label = "Malignant" if y_true == 0 else "Benign"
        print(f"\nSample (True: {y_true} - {true_label})")
        print(f"  Base votes: SVM={pred_svm}, DT={pred_dt}, KNN={pred_knn}, LR={pred_lr}")
        print(f"  Hard Voting: {pred_hard}, Soft Voting: {pred_soft}, Stacking: {pred_stack}")
    
    # ==================== Save Best Model ====================
    print("\n" + "="*70)
    print("Saving Models")
    print("="*70)
    
    # Find best model
    best_model_name = max(results, key=results.get)
    best_score = results[best_model_name]
    
    if best_model_name == 'Voting (Soft)':
        best_model = voting_soft
    elif best_model_name == 'Voting (Hard)':
        best_model = voting_hard
    elif best_model_name == 'Stacking':
        best_model = stacking_clf
    elif best_model_name == 'SVM':
        best_model = clf_svm
    elif best_model_name == 'Decision Tree':
        best_model = clf_dt
    elif best_model_name == 'KNN':
        best_model = clf_knn
    else:  # Logistic Regression
        best_model = clf_lr
    
    joblib.dump(best_model, 'best_ensemble_model_breastcancer.pkl')
    print(f"\nBest model: {best_model_name} (score: {best_score:.4f})")
    print(f"Saved as: best_ensemble_model_breastcancer.pkl")
    
    # Save all ensemble models
    joblib.dump(voting_hard, 'voting_hard_classifier_breastcancer.pkl')
    joblib.dump(voting_soft, 'voting_soft_classifier_breastcancer.pkl')
    joblib.dump(stacking_clf, 'stacking_classifier_breastcancer.pkl')
    
    print("\nAll ensemble models saved:")
    print("  - voting_hard_classifier_breastcancer.pkl")
    print("  - voting_soft_classifier_breastcancer.pkl")
    print("  - stacking_classifier_breastcancer.pkl")
    print("  - best_ensemble_model_breastcancer.pkl")
    
    plt.show()


def main():
    # todo_1()  # WIP
    # todo_2()
    # todo_3()
    # todo_4()
    todo_5()
    todo_6()
    todo_7()
    todo_10()

if __name__ == '__main__':
    main()
