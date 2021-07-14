import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import xgboost as xgb

def specificity_score(y_true,y_pred):
    matrix = confusion_matrix(y_true,y_pred)
    TN = matrix[0,0]
    FP = matrix[0,1]
    return TN/(FP+TN)

def NPV(y_true,y_pred):
    matrix = confusion_matrix(y_true,y_pred)
    TN = matrix[0,0]
    FN = matrix[1,0]
    return TN/(FN+TN)

def self_auc_cv(model,model_type,X_input,y_input,cv = 3,scoring = 'auc'):
    X_train,y_train = shuffle(X_input,y_input,random_state = 67)
    params = model.get_params()
    X1 = pd.DataFrame(X_train).values[0:int(np.floor(X_train.shape[0]/3)),:]
    X2 = pd.DataFrame(X_train).values[int(np.floor(X_train.shape[0]/3)):int(np.floor(X_train.shape[0]/3)*2),:]
    X3 = pd.DataFrame(X_train).values[int(np.floor(X_train.shape[0]/3)*2):,:]
    y1 = pd.DataFrame(y_train).values[0:int(np.floor(y_train.shape[0]/3)),:]
    y2 = pd.DataFrame(y_train).values[int(np.floor(y_train.shape[0]/3)):int(np.floor(y_train.shape[0]/3))*2,:]
    y3 = pd.DataFrame(y_train).values[int(np.floor(y_train.shape[0]/3)*2):,:]
    score = []
    for i in range(1,4):
        if i == 1:
            X = np.vstack((X1,X2))
            y = np.vstack((y1,y2)).ravel()
            X_test = X3
            y_test = y3.ravel()
        if i == 2:
            X = np.vstack((X1,X3))
            y = np.vstack((y1,y3)).ravel()
            X_test = X2
            y_test = y2.ravel()
        if i == 3:
            X = np.vstack((X2,X3))
            y = np.vstack((y2,y3)).ravel()
            X_test = X1
            y_test = y1.ravel()
        if model_type == 'lgb':
            clf = lgb.LGBMClassifier()
        if model_type == 'LR':
            clf = LogisticRegression()
        if model_type == 'SVM':
            clf = SVC()
        if model_type == 'RF':
            clf = RandomForestClassifier()
        if model_type == 'MLP':
            clf = MLPClassifier()
        if model_type == 'xgb':
            clf = xgb.XGBClassifier()
        clf.set_params(**params)
        clf.fit(X,y)
        if scoring == 'auc':
            score.append(roc_auc_score(y_test,pd.DataFrame(clf.predict_proba(X_test)).values[:,1]))
        if scoring == 'specificity':
            score.append(specificity_score(y_test,clf.predict(X_test)))
    return np.array(score)

class baseline_models:
    def __init__(self,df,variable_list,label,one_hot = False,standardized_diff = False,subgroup = None,random_state = 67):
        if not subgroup:
            self.df = df
        if subgroup == 'no_to_yes':
            self.df = df.query('later_CI == 1 and current_CI == 0')
            print(subgroup,' df size',self.df.shape)
        elif subgroup == 'yes_to_no':
            self.df = df.query('later_CI == 0 and current_CI == 1')
            print(subgroup,' df size',self.df.shape)
        elif subgroup == 'yes':
            self.df = df.query('later_CI == 0 and current_CI == 0')
            print(subgroup,' df size',self.df.shape)
            
        if 'borned_prov' in variable_list:
            variable_list.remove('borned_prov')
        if 'agegroup_3years_later' in variable_list:
            variable_list.remove('agegroup_3years_later')
        
        self.random_state = random_state
        self.variable_list = variable_list
        self.label = label
        self.X = df[variable_list]
        self.y = df[label]
        if one_hot:
            print('One-hot processing ...')
            one_hot_var = variable_list.copy()
            for var in one_hot_var:
                if var[:4]=='tour':one_hot_var.remove(var)
                if var[:10]=='depression':one_hot_var.remove(var)
            self.X = pd.get_dummies(self.X,columns = one_hot_var,prefix = one_hot_var)
            print('One-hot process done!')
        if standardized_diff:
            diff_var = list(filter(lambda x:True if x[-4:] == 'diff' else False,variable_list))
            if diff_var:
                print('Standardizing diff variables ...')
                for var in diff_var:
                    self.X[var] = self.X[var].apply(lambda x:x if x<= 0 else 1)
                    self.X[var] = self.X[var].apply(lambda x:x if x>=0 else -1)
                print('Standardize diff variables done!')
            
        self.features = self.select_features(self.X,self.y)
        self.X = self.X[self.features]
        
        
        self.lgb,self.lgb_performance = self.get_lgb(self.X,self.y)
        self.LR,self.LR_performance = self.get_LR(self.X,self.y)
        self.SVM,self.SVM_performance = self.get_SVM(self.X,self.y)
        self.RF,self.RF_performance = self.get_RF(self.X,self.y)
        self.MLP,self.MLP_performance = self.get_MLP(self.X,self.y)
        self.xgb,self.xgb_performance = self.get_xgb(self.X,self.y)
        
    def select_features(self,X,y):
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=self.random_state,test_size=0.33)
        # feature selection
        print("Selecting features ... ")
        LR = LogisticRegression(solver='liblinear',class_weight = 'balanced',random_state = self.random_state)
        sfs1 = SFS(estimator=LR, 
                   k_features=(4, 9),
                   forward=True, 
                   floating=True, 
                   scoring='roc_auc',
                   cv=3)
        sfs1.fit(X_train,y_train)
        selected_features = list(sfs1.k_feature_names_)

        print("Selecting features ... Done!")
        return selected_features
        
    def get_lgb(self,X,y):
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=self.random_state,test_size=0.33)
        # first step ：tuning max_depth and num_leaves
        print("Tuning parameters for lightgbm ...")
        params_test1={'max_depth': range(3,10,1), 'num_leaves':range(5, 100, 5)}
        gbm = lgb.LGBMClassifier(objective = 'binary',
                             is_unbalance = True,
                             metric = 'auc',
                             max_depth = 6,
                             num_leaves = 40,
                             learning_rate = 0.1,
                             feature_fraction = 0.7,
                             min_data_in_leaf = 50,
                             bagging_fraction = 1,
                             bagging_freq = 2,
                             reg_alpha = 0.001,
                             reg_lambda = 8,
                             random_state = self.random_state

                            )
        gsearch1 = GridSearchCV(gbm, param_grid=params_test1, scoring='roc_auc', cv=3)
        gsearch1.fit(X_train,y_train)
        best_max_depth = gsearch1.best_params_['max_depth']
        best_num_leaves = gsearch1.best_params_['num_leaves']
        # second step：tuning min_data_in_leaf
        params_test2={'min_data_in_leaf':range(10,500,10)}
        gbm = lgb.LGBMClassifier(objective = 'binary',
                             is_unbalance = True,
                             metric = 'auc',
                             max_depth = best_max_depth,
                             num_leaves = best_num_leaves,
                             learning_rate = 0.1,
                             feature_fraction = 0.7,
                             min_data_in_leaf = 50,
                             bagging_fraction = 1,
                             bagging_freq = 2,
                             reg_alpha = 0.001,
                             reg_lambda = 8,
                             random_state = self.random_state

                            )
        gsearch2 = GridSearchCV(gbm, param_grid=params_test2, scoring='roc_auc', cv=3)
        gsearch2.fit(X_train,y_train)
        best_min_data_in_leaf = gsearch2.best_params_['min_data_in_leaf']
        # third step：tuning feature_fraction
        params_test3={'feature_fraction': [0.6,0.7,0.8,0.9,1.0]
        }
        gbm = lgb.LGBMClassifier(objective = 'binary',
                             is_unbalance = True,
                             metric = 'auc',
                             max_depth = best_max_depth,
                             num_leaves = best_num_leaves,
                             learning_rate = 0.1,
                             feature_fraction = 0.7,
                             min_data_in_leaf = best_min_data_in_leaf,
                             bagging_fraction = 1,
                             bagging_freq = 2,
                             reg_alpha = 0.001,
                             reg_lambda = 8,
                             random_state = self.random_state

                            )
        gsearch3 = GridSearchCV(gbm, param_grid=params_test3, scoring='roc_auc', cv=3)
        gsearch3.fit(X_train,y_train)
        best_feature_fraction = gsearch3.best_params_['feature_fraction']
        # fourth_step：tuning bagging_fraction and bagging_frequency
        params_test4={
         'bagging_fraction': [0.5,0.6,0.7,0.8,0.9,1],
         'bagging_freq': [2,3,4,5],
            }
        gbm = lgb.LGBMClassifier(objective = 'binary',
                             is_unbalance = True,
                             metric = 'auc',
                             max_depth = best_max_depth,
                             num_leaves = best_num_leaves,
                             learning_rate = 0.1,
                             feature_fraction = best_feature_fraction,
                             min_data_in_leaf = best_min_data_in_leaf,
                             bagging_fraction = 1,
                             bagging_freq = 2,
                             reg_alpha = 0.001,
                             reg_lambda = 8,
                             random_state = self.random_state

                            )
        gsearch4 = GridSearchCV(gbm, param_grid=params_test4, scoring='roc_auc', cv=3)
        gsearch4.fit(X_train,y_train)
        best_bagging_fraction = gsearch4.best_params_['bagging_fraction']
        best_bagging_freq = gsearch4.best_params_['bagging_freq']
        # fifth_step : tuning lambda_l1 and lambda_l2
        params_test5={'lambda_l1': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0],
                  'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]
        }
        gbm = lgb.LGBMClassifier(objective = 'binary',
                             is_unbalance = True,
                             metric = 'auc',
                             max_depth = best_max_depth,
                             num_leaves = best_num_leaves,
                             learning_rate = 0.1,
                             feature_fraction = best_feature_fraction,
                             min_data_in_leaf = best_min_data_in_leaf,
                             bagging_fraction = best_bagging_fraction,
                             bagging_freq = best_bagging_freq,
                             reg_alpha = 0.001,
                             reg_lambda = 8,
                             random_state = self.random_state

                            )
        gsearch5 = GridSearchCV(gbm, param_grid=params_test5, scoring='roc_auc', cv=3)
        gsearch5.fit(X_train,y_train)
        best_lambda_l1 = gsearch5.best_params_['lambda_l1']
        best_lambda_l2 = gsearch5.best_params_['lambda_l2']

        best_params = {'max_depth':[best_max_depth], 
                               'num_leaves':[best_num_leaves],
                               'min_data_in_leaf':[best_min_data_in_leaf],
                               'bagging_fraction':[best_bagging_fraction],
                               'bagging_freq':[best_bagging_freq], 
                               'feature_fraction':[best_feature_fraction],
                               'reg_alpha':[best_lambda_l1],
                               'reg_lambda':[best_lambda_l2]}
        print("Tuning parameters for lightgbm ... Done!")
        print('Best parameters for Lightgbm : ')
        print(best_params)

        # result
        clf = lgb.LGBMClassifier(objective = 'binary',
                                         is_unbalance = True,
                                         metric = 'auc',
                                         learning_rate = 0.1,
                                         max_depth=best_max_depth, num_leaves=best_num_leaves,min_data_in_leaf=best_min_data_in_leaf,
                                         bagging_fraction=best_bagging_fraction,bagging_freq= best_bagging_freq, feature_fraction= best_feature_fraction,
                                         reg_alpha=best_lambda_l1,reg_lambda=best_lambda_l2,random_state = self.random_state)
        cv_acc = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'accuracy').mean()
        print(self_auc_cv(clf,'lgb',X_train,y_train,cv=3))
        cv_auc = self_auc_cv(clf,'lgb',X_train,y_train,cv=3).mean()
        cv_precision = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'precision').mean()
        cv_recall = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'recall').mean()
        cv_f1 = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'f1').mean()
        cv_specificity = self_auc_cv(clf,'lgb',X_train,y_train,cv=3,scoring='specificity').mean()
        

        clf = lgb.LGBMClassifier(objective = 'binary',
                                         is_unbalance = True,
                                         metric = 'auc',
                                         learning_rate = 0.1,
                                         max_depth=best_max_depth, num_leaves=best_num_leaves,min_data_in_leaf=best_min_data_in_leaf,
                                         bagging_fraction=best_bagging_fraction,bagging_freq= best_bagging_freq, feature_fraction= best_feature_fraction,
                                         reg_alpha=best_lambda_l1,reg_lambda=best_lambda_l2,random_state = self.random_state)
       
        clf.fit(X_train,y_train)
        acc_train = clf.score(X_train,y_train)
        auc_train = roc_auc_score(y_train,pd.DataFrame(clf.predict_proba(X_train)).values[:,1])
        precision_train = precision_score(y_train,clf.predict(X_train))
        recall_train = recall_score(y_train,clf.predict(X_train))
        f1_train = f1_score(y_train,clf.predict(X_train))
        specificity_train = specificity_score(y_train,clf.predict(X_train))
        brier_loss_train = brier_score_loss(y_train,pd.DataFrame(clf.predict_proba(X_train)).values[:,1])
    
        acc_test = clf.score(X_test,y_test)
        auc_test = roc_auc_score(y_test,pd.DataFrame(clf.predict_proba(X_test)).values[:,1])
        precision_test = precision_score(y_test,clf.predict(X_test))
        recall_test = recall_score(y_test,clf.predict(X_test))
        f1_test = f1_score(y_test,clf.predict(X_test))
        specificity_test = specificity_score(y_test,clf.predict(X_test))
        brier_loss_test = brier_score_loss(y_test,pd.DataFrame(clf.predict_proba(X_test)).values[:,1])
        NPV_test = NPV(y_test,clf.predict(X_test))
        performance = pd.DataFrame({'Model':'lightgbm','CV_accuracy':cv_acc,'CV_AUC':cv_auc,'CV_precision':cv_precision,'CV_recall':cv_recall,'CV_F1score':cv_f1,'CV_specificity':cv_specificity,
                                                    'train_accuracy':acc_train,'train_AUC':auc_train,'train_precision':precision_train,'train_recall':recall_train,'train_F1score':f1_train,
                                                    'train_specificity':specificity_train,'train_brier_loss':brier_loss_train,
                                                    'test_accuracy':acc_test,'test_AUC':auc_test,'test_precision':precision_test,'test_recall':recall_test,'test_F1score':f1_test,
                                                    'test_specificity':specificity_test,'test_brier_loss':brier_loss_test,'NPV_test':NPV_test},index = [0])
        return clf,performance
    
    def get_LR(self,X,y):
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=self.random_state,test_size=0.33)
        # first step
        print("Tuning parameters for LogisticRegression ...")
        params_test={'penalty':['l1','l2'], 'C':np.logspace(-5,5,20)}
        clf = LogisticRegression(solver='liblinear',class_weight = 'balanced',random_state = self.random_state)
        gsearch1 = GridSearchCV(clf, param_grid=params_test, scoring='roc_auc', cv=3)
        gsearch1.fit(X_train,y_train)
        best_penalty = gsearch1.best_params_['penalty']
        best_C = gsearch1.best_params_['C']

        best_params = {'penalty':[best_penalty],'C':[best_C]}
        print("Tuning parameters for LogisticRegression ... Done!")
        print('Best parameters for LogisticRegression : ')
        print(best_params)
        # final CV 
        clf = LogisticRegression(solver='liblinear',class_weight = 'balanced',penalty=best_penalty,C=best_C,random_state = self.random_state)
        cv_acc = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'accuracy').mean()
        cv_auc = self_auc_cv(clf,'LR',X_train,y_train,cv=3).mean()
        cv_precision = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'precision').mean()
        cv_recall = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'recall').mean()
        cv_f1 = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'f1').mean()
        cv_specificity = self_auc_cv(clf,'LR',X_train,y_train,cv=3,scoring='specificity').mean()


        clf = LogisticRegression(solver='liblinear',class_weight = 'balanced',penalty=best_penalty,C=best_C,random_state = self.random_state)
        clf.fit(X_train,y_train)
        acc_train = clf.score(X_train,y_train)
        auc_train = roc_auc_score(y_train,pd.DataFrame(clf.predict_proba(X_train)).values[:,1])
        precision_train = precision_score(y_train,clf.predict(X_train))
        recall_train = recall_score(y_train,clf.predict(X_train))
        f1_train = f1_score(y_train,clf.predict(X_train))
        specificity_train = specificity_score(y_train,clf.predict(X_train))
        brier_loss_train = brier_score_loss(y_train,pd.DataFrame(clf.predict_proba(X_train)).values[:,1])
    
        acc_test = clf.score(X_test,y_test)
        auc_test = roc_auc_score(y_test,pd.DataFrame(clf.predict_proba(X_test)).values[:,1])
        precision_test = precision_score(y_test,clf.predict(X_test))
        recall_test = recall_score(y_test,clf.predict(X_test))
        f1_test = f1_score(y_test,clf.predict(X_test))
        specificity_test = specificity_score(y_test,clf.predict(X_test))
        brier_loss_test = brier_score_loss(y_test,pd.DataFrame(clf.predict_proba(X_test)).values[:,1])
        NPV_test = NPV(y_test,clf.predict(X_test))
        performance = pd.DataFrame({'Model':'LR','CV_accuracy':cv_acc,'CV_AUC':cv_auc,'CV_precision':cv_precision,'CV_recall':cv_recall,'CV_F1score':cv_f1,'CV_specificity':cv_specificity,
                                                    'train_accuracy':acc_train,'train_AUC':auc_train,'train_precision':precision_train,'train_recall':recall_train,'train_F1score':f1_train,
                                                    'train_specificity':specificity_train,'train_brier_loss':brier_loss_train,
                                                    'test_accuracy':acc_test,'test_AUC':auc_test,'test_precision':precision_test,'test_recall':recall_test,'test_F1score':f1_test,
                                                    'test_specificity':specificity_test,'test_brier_loss':brier_loss_test,'NPV_test':NPV_test},index = [1])
        return clf,performance
    
    def get_SVM(self,X,y):
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=self.random_state,test_size=0.33)

        # first step
        print("Tuning parameters for SVM ...")
        params_test={'C':[1e-3,1e-2,1e-1,1,10,100,1000],'gamma':[0.001,0.0005,0.0001]}
        clf = SVC(kernel = 'rbf',class_weight = 'balanced',random_state = self.random_state)
        gsearch1 = GridSearchCV(clf, param_grid=params_test, scoring='roc_auc', cv=3)
        gsearch1.fit(X_train,y_train)
        best_C = gsearch1.best_params_['C']
        best_gamma = gsearch1.best_params_['gamma']

        best_params = {'C':[best_C],'gamma':[best_gamma]}
        print("Tuning parameters for SVM ... Done!")
        print('Best parameters for SVM : ')
        print(best_params)
        # result
        clf = SVC(kernel = 'rbf',class_weight = 'balanced',C=best_C,gamma=best_gamma,random_state = self.random_state,probability = True)
        cv_acc = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'accuracy').mean()
        cv_auc = self_auc_cv(clf,'SVM',X_train,y_train,cv=3).mean()
        cv_precision = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'precision').mean()
        cv_recall = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'recall').mean()
        cv_f1 = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'f1').mean()
        cv_specificity = self_auc_cv(clf,'SVM',X_train,y_train,cv=3,scoring='specificity').mean()


        clf = SVC(kernel = 'rbf',class_weight = 'balanced',C=best_C,gamma=best_gamma,random_state = self.random_state,probability = True)
        clf.fit(X_train,y_train)
        acc_train = clf.score(X_train,y_train)
        auc_train = roc_auc_score(y_train,pd.DataFrame(clf.predict_proba(X_train)).values[:,1])
        precision_train = precision_score(y_train,clf.predict(X_train))
        recall_train = recall_score(y_train,clf.predict(X_train))
        f1_train = f1_score(y_train,clf.predict(X_train))
        specificity_train = specificity_score(y_train,clf.predict(X_train))
        brier_loss_train = brier_score_loss(y_train,pd.DataFrame(clf.predict_proba(X_train)).values[:,1])
    
        acc_test = clf.score(X_test,y_test)
        auc_test = roc_auc_score(y_test,pd.DataFrame(clf.predict_proba(X_test)).values[:,1])
        precision_test = precision_score(y_test,clf.predict(X_test))
        recall_test = recall_score(y_test,clf.predict(X_test))
        f1_test = f1_score(y_test,clf.predict(X_test))
        specificity_test = specificity_score(y_test,clf.predict(X_test))
        brier_loss_test = brier_score_loss(y_test,pd.DataFrame(clf.predict_proba(X_test)).values[:,1])
        NPV_test = NPV(y_test,clf.predict(X_test))
        performance = pd.DataFrame({'Model':'SVM','CV_accuracy':cv_acc,'CV_AUC':cv_auc,'CV_precision':cv_precision,'CV_recall':cv_recall,'CV_F1score':cv_f1,'CV_specificity':cv_specificity,
                                                    'train_accuracy':acc_train,'train_AUC':auc_train,'train_precision':precision_train,'train_recall':recall_train,'train_F1score':f1_train,
                                                    'train_specificity':specificity_train,'train_brier_loss':brier_loss_train,
                                                    'test_accuracy':acc_test,'test_AUC':auc_test,'test_precision':precision_test,'test_recall':recall_test,'test_F1score':f1_test,
                                                    'test_specificity':specificity_test,'test_brier_loss':brier_loss_test,'NPV_test':NPV_test},index = [2])
        return clf,performance
    
    def get_RF(self,X,y):
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=self.random_state,test_size=0.33)
        # first step
        print("Tuning parameters for RandomForest ... ")
        param_grid1 = {'max_depth':np.arange(1, 20, 2)} 
        clf = RandomForestClassifier(class_weight = 'balanced',random_state = self.random_state) 
        gsearch1 = GridSearchCV(clf,param_grid1,cv=3,scoring='roc_auc') 
        gsearch1.fit(X_train,y_train)
        best_max_depth = gsearch1.best_params_['max_depth']
        # second step
        param_grid2 = {'max_features':np.arange(5,23,2)}
        clf = RandomForestClassifier(max_depth = best_max_depth,class_weight = 'balanced',random_state = self.random_state) 
        gsearch2 = GridSearchCV(clf,param_grid2,cv=3,scoring='roc_auc') 
        gsearch2.fit(X_train,y_train)
        best_max_features = gsearch2.best_params_['max_features']
        # third step
        param_grid3 = {'min_samples_leaf':np.arange(1, 1+100, 2)}
        clf = RandomForestClassifier(max_depth = best_max_depth,max_features = best_max_features,class_weight = 'balanced',random_state = self.random_state) 
        gsearch3 = GridSearchCV(clf,param_grid3,cv=3,scoring='roc_auc') 
        gsearch3.fit(X_train,y_train)
        best_min_samples_leaf = gsearch3.best_params_['min_samples_leaf']
        # fourth step
        param_grid4 = {'min_samples_split':np.arange(2, 2+20, 2)}
        clf = RandomForestClassifier(max_depth = best_max_depth,max_features = best_max_features,min_samples_leaf = best_min_samples_leaf,class_weight = 'balanced',random_state = self.random_state) 
        gsearch4 = GridSearchCV(clf,param_grid4,cv=3,scoring='roc_auc') 
        gsearch4.fit(X_train,y_train)
        best_min_samples_split = gsearch4.best_params_['min_samples_split']
        # fourth step
        param_grid5 = {'criterion':['gini', 'entropy']}
        clf = RandomForestClassifier(max_depth = best_max_depth,max_features = best_max_features,min_samples_leaf = best_min_samples_leaf,min_samples_split = best_min_samples_split,class_weight = 'balanced',random_state = self.random_state)
        gsearch5 = GridSearchCV(clf,param_grid5,cv=3,scoring='roc_auc') 
        gsearch5.fit(X_train,y_train)
        best_criterion = gsearch5.best_params_['criterion']

        best_params = {'max_depth':[best_max_depth],'max_features':[best_max_features],'min_samples_leaf':[best_min_samples_leaf],
                       'min_samples_split':[best_min_samples_split],'criterion':[best_criterion]}
        print("Tuning parameters for RandomForest ... Done!")
        print('Best parameters for RandomForest : ')
        print(best_params)
        # result
        clf = RandomForestClassifier(class_weight = 'balanced',max_depth=best_max_depth,max_features=best_max_features,min_samples_leaf=best_min_samples_leaf,
                                                 min_samples_split=best_min_samples_split,criterion=best_criterion,random_state = self.random_state)
        cv_acc = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'accuracy').mean()
        cv_auc = self_auc_cv(clf,'RF',X_train,y_train,cv=3).mean()
        cv_precision = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'precision').mean()
        cv_recall = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'recall').mean()
        cv_f1 = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'f1').mean()
        cv_specificity = self_auc_cv(clf,'RF',X_train,y_train,cv=3,scoring='specificity').mean()


        clf = RandomForestClassifier(class_weight = 'balanced',max_depth=best_max_depth,max_features=best_max_features,min_samples_leaf=best_min_samples_leaf,
                                                 min_samples_split=best_min_samples_split,criterion=best_criterion,random_state = self.random_state)
        clf.fit(X_train,y_train)
        acc_train = clf.score(X_train,y_train)
        auc_train = roc_auc_score(y_train,pd.DataFrame(clf.predict_proba(X_train)).values[:,1])
        precision_train = precision_score(y_train,clf.predict(X_train))
        recall_train = recall_score(y_train,clf.predict(X_train))
        f1_train = f1_score(y_train,clf.predict(X_train))
        specificity_train = specificity_score(y_train,clf.predict(X_train))
        brier_loss_train = brier_score_loss(y_train,pd.DataFrame(clf.predict_proba(X_train)).values[:,1])
    
        acc_test = clf.score(X_test,y_test)
        auc_test = roc_auc_score(y_test,pd.DataFrame(clf.predict_proba(X_test)).values[:,1])
        precision_test = precision_score(y_test,clf.predict(X_test))
        recall_test = recall_score(y_test,clf.predict(X_test))
        f1_test = f1_score(y_test,clf.predict(X_test))
        specificity_test = specificity_score(y_test,clf.predict(X_test))
        brier_loss_test = brier_score_loss(y_test,pd.DataFrame(clf.predict_proba(X_test)).values[:,1])
        NPV_test = NPV(y_test,clf.predict(X_test))
        performance = pd.DataFrame({'Model':'RF','CV_accuracy':cv_acc,'CV_AUC':cv_auc,'CV_precision':cv_precision,'CV_recall':cv_recall,'CV_F1score':cv_f1,'CV_specificity':cv_specificity,
                                                    'train_accuracy':acc_train,'train_AUC':auc_train,'train_precision':precision_train,'train_recall':recall_train,'train_F1score':f1_train,
                                                    'train_specificity':specificity_train,'train_brier_loss':brier_loss_train,
                                                    'test_accuracy':acc_test,'test_AUC':auc_test,'test_precision':precision_test,'test_recall':recall_test,'test_F1score':f1_test,
                                                    'test_specificity':specificity_test,'test_brier_loss':brier_loss_test,'NPV_test':NPV_test},index = [3])
        return clf,performance
    
    def get_MLP(self,X,y):
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=self.random_state,test_size=0.33)
        print("Selecting features for MLP ... Done!")
        y_train = y_train.reset_index(drop=True)
        X_train = X_train.reset_index(drop=True)
        y_pos = y_train[y_train == 1]
        y_neg = y_train[y_train == 0]
        y_neg_sam = y_neg.sample(len(y_pos))
        y_train = pd.concat([y_pos,y_neg_sam],axis = 0)
        X_train = X_train.iloc[y_train.index,:]
        print("Tuning parameters for MLP ... ")
        # first step
        param_grid1 = {'hidden_layer_sizes':[16,32,64,100,128,150,200,256]} 
        clf = MLPClassifier(max_iter = 1000,random_state = self.random_state) 
        gsearch1 = GridSearchCV(clf,param_grid1,cv=3,scoring='roc_auc') 
        gsearch1.fit(X_train,y_train)
        best_hidden_layer_sizes = gsearch1.best_params_['hidden_layer_sizes']
        # second step
        param_grid2 = {'learning_rate_init':[0.01,0.005,0.001,0.0005,0.0001]}
        clf = MLPClassifier(max_iter = 1000,hidden_layer_sizes = best_hidden_layer_sizes,random_state = self.random_state)
        gsearch2 = GridSearchCV(clf,param_grid2,cv=3,scoring='roc_auc') 
        gsearch2.fit(X_train,y_train)
        best_learning_rate_init = gsearch2.best_params_['learning_rate_init']
        # third step
        param_grid3 = {'alpha':[1e-4,1e-5,1e-6]}
        clf = MLPClassifier(max_iter = 1000,hidden_layer_sizes = best_hidden_layer_sizes,learning_rate_init = best_learning_rate_init,random_state = self.random_state)
        gsearch3 = GridSearchCV(clf,param_grid3,cv=3,scoring='roc_auc') 
        gsearch3.fit(X_train,y_train)
        best_alpha = gsearch3.best_params_['alpha']

        best_params = {'hidden_layer_sizes':[best_hidden_layer_sizes],'learning_rate_init':[best_learning_rate_init],'alpha':[best_alpha]}
        print("Tuning parameters for MLP ... Done!")
        print('Best parameters for MLP : ')
        print(best_params)
        
        # final CV 
        clf = MLPClassifier(max_iter = 1000,hidden_layer_sizes = best_hidden_layer_sizes,learning_rate_init = best_learning_rate_init,alpha = best_alpha,random_state = self.random_state)
        cv_acc = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'accuracy').mean()
        cv_auc = self_auc_cv(clf,'MLP',X_train,y_train,cv=3).mean()
        cv_precision = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'precision').mean()
        cv_recall = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'recall').mean()
        cv_f1 = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'f1').mean()
        cv_specificity = self_auc_cv(clf,'MLP',X_train,y_train,cv=3,scoring='specificity').mean()

        clf = MLPClassifier(max_iter = 1000,hidden_layer_sizes = best_hidden_layer_sizes,learning_rate_init = best_learning_rate_init,alpha = best_alpha,random_state = self.random_state)
        clf.fit(X_train,y_train)
        acc_train = clf.score(X_train,y_train)
        auc_train = roc_auc_score(y_train,pd.DataFrame(clf.predict_proba(X_train)).values[:,1])
        precision_train = precision_score(y_train,clf.predict(X_train))
        recall_train = recall_score(y_train,clf.predict(X_train))
        f1_train = f1_score(y_train,clf.predict(X_train))
        specificity_train = specificity_score(y_train,clf.predict(X_train))
        brier_loss_train = brier_score_loss(y_train,pd.DataFrame(clf.predict_proba(X_train)).values[:,1])
    
        acc_test = clf.score(X_test,y_test)
        auc_test = roc_auc_score(y_test,pd.DataFrame(clf.predict_proba(X_test)).values[:,1])
        precision_test = precision_score(y_test,clf.predict(X_test))
        recall_test = recall_score(y_test,clf.predict(X_test))
        f1_test = f1_score(y_test,clf.predict(X_test))
        specificity_test = specificity_score(y_test,clf.predict(X_test))
        brier_loss_test = brier_score_loss(y_test,pd.DataFrame(clf.predict_proba(X_test)).values[:,1])
        NPV_test = NPV(y_test,clf.predict(X_test))
        performance = pd.DataFrame({'Model':'NN','CV_accuracy':cv_acc,'CV_AUC':cv_auc,'CV_precision':cv_precision,'CV_recall':cv_recall,'CV_F1score':cv_f1,'CV_specificity':cv_specificity,
                                                    'train_accuracy':acc_train,'train_AUC':auc_train,'train_precision':precision_train,'train_recall':recall_train,'train_F1score':f1_train,
                                                    'train_specificity':specificity_train,'train_brier_loss':brier_loss_train,
                                                    'test_accuracy':acc_test,'test_AUC':auc_test,'test_precision':precision_test,'test_recall':recall_test,'test_F1score':f1_test,
                                                    'test_specificity':specificity_test,'test_brier_loss':brier_loss_test,'NPV_test':NPV_test},index = [4])
        return clf,performance
    
    def get_xgb(self,X,y):
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=self.random_state,test_size=0.33)
        # first step ：tuning max_depth and num_leaves
        print("Tuning parameters for XGBoost ... ")
        params_test1={'min_child_weight': [1, 2, 3, 4, 5, 6],'max_depth': [3, 4, 5, 6, 7, 8, 9, 10]}
        gbm = xgb.XGBClassifier(
                            scale_pos_weight = len(y[y==0])/len(y[y==1]),
                            learning_rate = 0.1,
                            max_depth = 5,
                            min_child_weight = 1,
                            subsample = 0.8,
                            colsample_bytree = 0.8,
                            gamma = 0,
                            reg_alpha = 0,
                            reg_lambda = 1,
                            random_state=self.random_state
                            )
                             
        gsearch1 = GridSearchCV(gbm, param_grid=params_test1, scoring='roc_auc', cv=3)
        gsearch1.fit(X_train,y_train)
        best_max_depth = gsearch1.best_params_['max_depth']
        best_min_child_weight = gsearch1.best_params_['min_child_weight']
        # second step：tuning min_data_in_leaf
        params_test2={'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
        gbm = xgb.XGBRFClassifier(
                            scale_pos_weight = len(y[y==0])/len(y[y==1]),
                            learning_rate = 0.1,
                            max_depth = best_max_depth,
                            min_child_weight = best_min_child_weight,
                            subsample = 0.8,
                            colsample_bytree = 0.8,
                            gamma = 0,
                            reg_alpha = 0,
                            reg_lambda = 1,
                            random_state=self.random_state
                            )
        gsearch2 = GridSearchCV(gbm, param_grid=params_test2, scoring='roc_auc', cv=3)
        gsearch2.fit(X_train,y_train)
        best_gamma = gsearch2.best_params_['gamma']
        # third step：tuning feature_fraction
        params_test3={'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
        gbm = xgb.XGBClassifier(
                            scale_pos_weight = len(y[y==0])/len(y[y==1]),
                            learning_rate = 0.1,
                            max_depth = best_max_depth,
                            min_child_weight = best_min_child_weight,
                            subsample = 0.8,
                            colsample_bytree = 0.8,
                            gamma = best_gamma,
                            reg_alpha = 0,
                            reg_lambda = 1,
            random_state=self.random_state
                            )
        gsearch3 = GridSearchCV(gbm, param_grid=params_test3, scoring='roc_auc', cv=3)
        gsearch3.fit(X_train,y_train)
        best_subsample = gsearch3.best_params_['subsample']
        best_colsample_bytree = gsearch3.best_params_['colsample_bytree']
        # fourth_step：tuning bagging_fraction and bagging_frequency
        params_test4={'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
        gbm = xgb.XGBClassifier(
                            scale_pos_weight = len(y[y==0])/len(y[y==1]),
                            learning_rate = 0.1,
                            max_depth = best_max_depth,
                            min_child_weight = best_min_child_weight,
                            subsample = best_subsample,
                            colsample_bytree = best_colsample_bytree,
                            gamma = best_gamma,
                            reg_alpha = 0,
                            reg_lambda = 1,
            random_state=self.random_state
                            )
        gsearch4 = GridSearchCV(gbm, param_grid=params_test4, scoring='roc_auc', cv=3)
        gsearch4.fit(X_train,y_train)
        best_reg_alpha = gsearch4.best_params_['reg_alpha']
        best_reg_lambda = gsearch4.best_params_['reg_lambda']
        # fifth_step : tuning lambda_l1 and lambda_l2
        params_test5={'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
        gbm = xgb.XGBClassifier(
                            scale_pos_weight = len(y[y==0])/len(y[y==1]),
                            learning_rate = 0.1,
                            max_depth = best_max_depth,
                            min_child_weight = best_min_child_weight,
                            subsample = best_subsample,
                            colsample_bytree = best_colsample_bytree,
                            gamma = best_gamma,
                            reg_alpha = best_reg_alpha,
                            reg_lambda = best_reg_lambda,
            random_state=self.random_state
                            )
        gsearch5 = GridSearchCV(gbm, param_grid=params_test5, scoring='roc_auc', cv=3)
        gsearch5.fit(X_train,y_train)
        best_learning_rate = gsearch5.best_params_['learning_rate']

        best_params = {'max_depth':[best_max_depth], 
                               'min_child_weight':[best_min_child_weight],
                               'subsample':[best_subsample],
                               'colsample_bytree':[best_colsample_bytree],
                               'gamma':[best_gamma], 
                               'reg_alpha':[best_reg_alpha],
                               'reg_lambda':[best_reg_lambda],
                               'learning_rate':[best_learning_rate],
                                 }
        print("Tuning parameters for XGBoost ... Done!")
        print('Best parameters for XGBoost : ')
        print(best_params)

        # result
        clf = xgb.XGBClassifier(
                            scale_pos_weight = len(y[y==0])/len(y[y==1]),
                            learning_rate = best_learning_rate,
                            max_depth = best_max_depth,
                            min_child_weight = best_min_child_weight,
                            subsample = best_subsample,
                            colsample_bytree = best_colsample_bytree,
                            gamma = best_gamma,
                            reg_alpha = best_reg_alpha,
                            reg_lambda = best_reg_lambda,
                            random_state=self.random_state
                            )
        cv_acc = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'accuracy').mean()
        cv_auc = self_auc_cv(clf,'xgb',X_train,y_train,cv=3).mean()
        cv_precision = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'precision').mean()
        cv_recall = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'recall').mean()
        cv_f1 = cross_val_score(clf,X_train,y_train,cv=3,scoring = 'f1').mean()
        cv_specificity = self_auc_cv(clf,'xgb',X_train,y_train,cv=3,scoring='specificity').mean()
        

        clf = xgb.XGBClassifier(
                            scale_pos_weight = len(y[y==0])/len(y[y==1]),
                            learning_rate = best_learning_rate,
                            max_depth = best_max_depth,
                            min_child_weight = best_min_child_weight,
                            subsample = best_subsample,
                            colsample_bytree = best_colsample_bytree,
                            gamma = best_gamma,
                            reg_alpha = best_reg_alpha,
                            reg_lambda = best_reg_lambda,
                            random_state=self.random_state
                            )
       
        clf.fit(X_train,y_train)
        acc_train = clf.score(X_train,y_train)
        auc_train = roc_auc_score(y_train,pd.DataFrame(clf.predict_proba(X_train)).values[:,1])
        precision_train = precision_score(y_train,clf.predict(X_train))
        recall_train = recall_score(y_train,clf.predict(X_train))
        f1_train = f1_score(y_train,clf.predict(X_train))
        specificity_train = specificity_score(y_train,clf.predict(X_train))
        brier_loss_train = brier_score_loss(y_train,pd.DataFrame(clf.predict_proba(X_train)).values[:,1])
    
        acc_test = clf.score(X_test,y_test)
        auc_test = roc_auc_score(y_test,pd.DataFrame(clf.predict_proba(X_test)).values[:,1])
        precision_test = precision_score(y_test,clf.predict(X_test))
        recall_test = recall_score(y_test,clf.predict(X_test))
        f1_test = f1_score(y_test,clf.predict(X_test))
        specificity_test = specificity_score(y_test,clf.predict(X_test))
        brier_loss_test = brier_score_loss(y_test,pd.DataFrame(clf.predict_proba(X_test)).values[:,1])
        NPV_test = NPV(y_test,clf.predict(X_test))
        performance = pd.DataFrame({'Model':'XGBoost','CV_accuracy':cv_acc,'CV_AUC':cv_auc,'CV_precision':cv_precision,'CV_recall':cv_recall,'CV_F1score':cv_f1,'CV_specificity':cv_specificity,
                                                    'train_accuracy':acc_train,'train_AUC':auc_train,'train_precision':precision_train,'train_recall':recall_train,'train_F1score':f1_train,
                                                    'train_specificity':specificity_train,'train_brier_loss':brier_loss_train,
                                                    'test_accuracy':acc_test,'test_AUC':auc_test,'test_precision':precision_test,'test_recall':recall_test,'test_F1score':f1_test,
                                                    'test_specificity':specificity_test,'test_brier_loss':brier_loss_test,'NPV_test':NPV_test},index = [0])
        return clf,performance
    
    def get_performance(self):
        return pd.concat([self.lgb_performance,self.LR_performance,self.SVM_performance,self.RF_performance,self.MLP_performance,self.xgb_performance],axis=0)
    
    def get_importance(self,model):
        if model == 'lgb':clf = self.lgb
        if model == 'LR':clf = self.LR
        if model == 'SVM':clf = self.SVM
        if model == 'RF':clf = self.RF
        if model == 'MLP':clf = self.MLP
        if model == 'xgb':clf = self.xgb
        result = permutation_importance(clf,self.X,self.y,n_repeats=10,random_state=self.random_state)
        importance = pd.DataFrame({'variable':self.features,'importance':result.importances_mean}).sort_values(by='importance',ascending = False)
        plt.figure(figsize = (15,10))
        g = sns.barplot(importance.variable,importance.importance)
        g.set_xticklabels(g.get_xticklabels(),rotation = 90)
        return importance

    def get_correlation(self):
        sns.heatmap(self.X.corr(),cmap = 'YlOrRd')
        return self.X.corr()