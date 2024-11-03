import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import xgboost as xgb
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn import metrics


def load_xgboost_train_data():
    dir = '../data/train_data/'
    train_x = pd.read_hdf(dir+'xgboost_train_x.h5', key='data', mode='r')
    train_y = pd.read_hdf(dir+'xgboost_train_y.h5', key='data', mode='r')
    test_x = pd.read_hdf(dir+'xgboost_test_x.h5', key='data', mode='r')
    test_y = pd.read_hdf(dir+'xgboost_test_y.h5', key='data', mode='r')
    return train_x, train_y, test_x, test_y


def load_model(model_path):
    if os.path.exists(model_path):
        return xgb.Booster(model_file=model_path)
    return None


# print ML model status
def print_model(y, prob, label, threshold): # threshold represent threshold
    print('-'*30)
    if threshold != 'all':
        pred = prob > threshold
        print("Accuracy  (%s): %.4g" %(label, metrics.accuracy_score(y, pred)))
        print("Precision (%s): %.4g" %(label, metrics.precision_score(y, pred)))
        print("Recall    (%s): %.4g" %(label, metrics.recall_score(y, pred)))
        print("AUC Score (%s): %.4g" %(label, metrics.roc_auc_score(y, prob)))
        print("F1-score  (%s): %.4g" %(label, metrics.f1_score(y, pred)))

    if threshold == 'all':
        print('%s set: ' %(label))
        print('Threshold  Accuracy  Precision  Recall  AUC Score  F1-score')
        for i in np.linspace(0, 1, num=21)[1:20]:
            if i !=0:
                pred = prob > i
                print('%.2f  %.4g %.4g %.4g %.4g %.4g' %(i, metrics.accuracy_score(y, pred),
                                                metrics.precision_score(y, pred),
                                                metrics.recall_score(y, pred),
                                                metrics.roc_auc_score(y, prob),
                                                metrics.f1_score(y, pred)))

# fit a XGBoost model
def model_fit(alg, features, targets, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        print("Training use CV")
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(features, label=targets, feature_names=features.columns)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        print(cvresult)
        alg.set_params(n_estimators=cvresult.shape[0])
    # Fit the algorithm on the data
    alg.fit(features, targets, eval_metric='auc')
    # Predict training set:
    dtrain_predprob = alg.predict_proba(features)[:, 1]
    # Print model report:
    print("\nModel Report")
    print_model(y=targets, prob=dtrain_predprob, label='Train', threshold=0.5)

# test XGBoost model
def model_test(alg, features, targets, X_test, Y_test):
    # Predict training set:
    # Fit the algorithm on the data
    model_fit(alg, features, targets)
    #predictions = alg.predict(X_test)
    predprob = alg.predict_proba(X_test)[:, 1]
    print_model(y=Y_test, prob=predprob, label='Test', threshold=0.5)


# train a XGBoost model with parameter setting below
# then save model to file
def train_xgb(params=None, save_model=True):
    train_x, train_y, test_x, test_y = load_xgboost_train_data()
    model_path = '../model/model_used/XGBoost_model.model'
    xgbmodel = XGBClassifier(**params)
    model_test(xgbmodel, train_x, train_y, test_x, test_y)
    if save_model:
        xgbmodel.save_model(model_path)
    return xgbmodel


if __name__=='__main__':
    model_params = {
        'learning_rate': 0.05,
        'n_estimators': 200,
        'max_depth': 5,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'scale_pos_weight': 1,
        'random_state': 22,
        'n_jobs': 10
    }
    model = train_xgb(params=model_params,save_model=True)