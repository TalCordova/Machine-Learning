import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.stats import norm, binom
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from statsmodels.graphics.mosaicplot import mosaic
import scipy.stats as ss
import math
import warnings
import scipy.cluster.hierarchy as sch
from collections import Counter
import csv
from sklearn.svm import SVC
from scipy.stats import multivariate_normal
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#-------------------------------------------------------------------------full decision Trees-----------------------------------------------------------------------------

data = pd.read_csv("C:/Users/talco/Desktop/MLProject/newXY2_dummy.csv")
x_trainPrime = data.drop(['y', 'id'], 1)
y_trainPrime = data['y'].values
X_train, X_validation, y_train, y_validation = train_test_split(x_trainPrime, y_trainPrime, test_size=0.17, random_state=123)

# print(y_validation)
# model = DecisionTreeClassifier(criterion='entropy')
# model.fit(X_train, y_train)
#
# plt.figure(figsize=(7, 6))
# plot_tree(model, filled=True, class_names=True)
# plt.show()
#
# print(f"Train size: {X_train.shape[0]}")
# print(f"Test size: {X_validation.shape[0]}")
# accuracy = accuracy_score(y_true=y_train, y_pred=model.predict(X_train))
# print(f"Train Accuracy: {accuracy_score(y_true=y_train, y_pred=model.predict(X_train)):.2f}")
# #
# accuracy = accuracy_score(y_true=y_validation, y_pred=model.predict(X_validation))
# print(f"Test Accuracy: {accuracy_score(y_true=y_validation, y_pred=model.predict(X_validation)):.2f}")
# y_pred_train = model.predict(X_train)
# cm_train = confusion_matrix(y_pred_train, y_train)
#
# print(f"Train size: {X_train.shape[0]}")
# print(f"Test size: {X_validation.shape[0]}")
#
# print("Train\n-----------\n", pd.value_counts(y_train)/y_train.shape[0])
# print("\nTest\n-----------\n", pd.value_counts(y_validation)/y_validation.shape[0])
#
# #------------------------------------------------------------------------ tune max_depth--------------------------------------------------------------------------------------
# max_depth_list = np.arange(1, 30, 1)
#
# res = pd.DataFrame()
# for max_depth in max_depth_list:
#     model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
#     model.fit(X_train, y_train)
#     res = res.append({'max_depth': max_depth,
#                       'train_acc':accuracy_score(y_train, model.predict(X_train)),
#                       'test_acc':accuracy_score(y_validation, model.predict(X_validation))}, ignore_index=True)
#
# plt.figure(figsize=(13, 4))
# plt.plot(res['max_depth'], res['train_acc'], marker='o', markersize=4)
# plt.plot(res['max_depth'], res['test_acc'], marker='o', markersize=4)
# plt.legend(['Train accuracy', 'Test accuracy'])
# plt.xlabel('Max Depth', fontsize=18)
# plt.ylabel('Scores',fontsize=18)
# plt.title('Decision Tree Classifier Scores for Different Number of Maximum Depth',fontsize=25)
# plt.show()
#
# print(res.sort_values('test_acc', ascending=False))
#
# #------------------------------------------------------------------------------------criterion tuning-----------------------------------------------------------------------------------------------------
# resCriterion = pd.DataFrame()
# criterionList = ['gini', 'entropy']
#
# for criterion in criterionList:
#     print(criterion)
#     model = DecisionTreeClassifier(criterion=criterion, random_state=42)
#     model.fit(X_train, y_train)
#     train_acc = model.score(X_train, y_train)
#     test_acc = model.score(X_validation, y_validation)
#     print("Train accuracy: ", round(model.score(X_train, y_train), 3))
#     print("Test accuracy: ", round(model.score(X_validation, y_validation), 3))
#     resCriterion = resCriterion.append({'Criterion': criterion,
#                       'train_acc': model.score(X_train, y_train),
#                       'test_acc': model.score(X_validation, y_validation)}, ignore_index=True)
#
#
# plt.figure(figsize=(7, 4))
# plt.plot(resCriterion['Criterion'], resCriterion['train_acc'], label='Train')
# plt.plot(resCriterion['Criterion'], resCriterion['test_acc'], label='Test')
# plt.title("Accuracy and Criterion", fontsize=25)
# plt.xlabel("Criterion", fontsize=20)
# plt.ylabel("Accuracy",fontsize=20)
# plt.legend()
# plt.show()
# print(resCriterion.sort_values('test_acc', ascending=False))
# print(train_accs)
#
# #----------------------------------------------------------------------------------------min sample split tuning-------------------------------------------------------------------------------------------
#
# min_sample_split = np.arange(0.1, 1, 0.05)
#
# res = pd.DataFrame()
# for minSmapleSplit in min_sample_split:
#     model = DecisionTreeClassifier(criterion='entropy', min_samples_split=minSmapleSplit, random_state=42)
#     model.fit(X_train, y_train)
#     res = res.append({'min_sample_split': minSmapleSplit,
#                       'train_acc':accuracy_score(y_train, model.predict(X_train)),
#                       'test_acc':accuracy_score(y_validation, model.predict(X_validation))}, ignore_index=True)
#
# plt.figure(figsize=(13, 4))
# plt.plot(res['min_sample_split'], res['train_acc'], marker='o', markersize=4)
# plt.plot(res['min_sample_split'], res['test_acc'], marker='o', markersize=4)
# plt.legend(['Train accuracy', 'Test accuracy'])
# plt.xlabel('Min Sample Split', fontsize=18)
# plt.ylabel('Scores',fontsize=18)
# plt.title('Decision Tree Classifier Scores for Different Number of Min Sample Split',fontsize=25)
# plt.show()
#
# print(res.sort_values('test_acc', ascending=False))
#
# #----------------------------------------------------------------------------------------max features tuning--------------------------------------------------------------------------------------------
#
#
# model = DecisionTreeClassifier(criterion='entropy', random_state=42, max_features=None)
# model.fit(X_train, y_train)
# print("N_features Max Features Results")
# print(f"Train accuracy: {accuracy_score(y_train, model.predict(X_train)):.2}")
# print(f"Test accuracy: {accuracy_score(y_validation, model.predict(X_validation)):.2}")
#
# print("SQRT Criterion Results")
# model = DecisionTreeClassifier(criterion='entropy', random_state=42, max_features='sqrt')
# model.fit(X_train, y_train)
# print(f"Train accuracy: {accuracy_score(y_train, model.predict(X_train)):.2}")
# print(f"Test accuracy: {accuracy_score(y_validation, model.predict(X_validation)):.2}")
#
# print("LOG2 Criterion Results")
# model = DecisionTreeClassifier(criterion='entropy', random_state=42,  max_features='log2')
# model.fit(X_train, y_train)
# print(f"Train accuracy: {accuracy_score(y_train, model.predict(X_train)):.2}")
# print(f"Test accuracy: {accuracy_score(y_validation, model.predict(X_validation)):.2}")
#
# max_features = np.arange(1, 30, 1)
#
# res = pd.DataFrame()
# for maxFeatures in max_features:
#     model = DecisionTreeClassifier(criterion='entropy', max_features=maxFeatures, random_state=42)
#     model.fit(X_train, y_train)
#     res = res.append({'max_features': maxFeatures,
#                       'train_acc':accuracy_score(y_train, model.predict(X_train)),
#                       'test_acc':accuracy_score(y_validation, model.predict(X_validation))}, ignore_index=True)
#
# plt.figure(figsize=(13, 4))
# plt.plot(res['max_features'], res['train_acc'], marker='o', markersize=4)
# plt.plot(res['max_features'], res['test_acc'], marker='o', markersize=4)
# plt.legend(['Train accuracy', 'Test accuracy'])
# plt.xlabel('Max Feature', fontsize=18)
# plt.ylabel('Scores',fontsize=18)
# plt.title('Decision Tree Classifier Scores for Different Number of Max Number of Features',fontsize=25)
# plt.show()
#
# print(res.sort_values('test_acc', ascending=False))
# #
# #------------------------------------------------------------------------------------min samples leaf tuning----------------------------------------------------------------------------------------------------
#
# min_sample_leaf = np.arange(0.1, 0.5, 0.05)
#
# res = pd.DataFrame()
# for minSmapleLeaf in min_sample_leaf:
#     model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=minSmapleLeaf, random_state=42)
#     model.fit(X_train, y_train)
#     res = res.append({'min_sample_leaf': minSmapleLeaf,
#                       'train_acc':accuracy_score(y_train, model.predict(X_train)),
#                       'test_acc':accuracy_score(y_validation, model.predict(X_validation))}, ignore_index=True)
#
# plt.figure(figsize=(13, 4))
# plt.plot(res['min_sample_leaf'], res['train_acc'], marker='o', markersize=4)
# plt.plot(res['min_sample_leaf'], res['test_acc'], marker='o', markersize=4)
# plt.legend(['Train accuracy', 'Test accuracy'])
# plt.xlabel('Min Sample Leaf', fontsize=18)
# plt.ylabel('Scores',fontsize=18)
# plt.title('Decision Tree Classifier Scores for Different Number of Min Sample Leaf',fontsize=25)
# plt.show()
#
# print(res.sort_values('test_acc', ascending=False))
#
# #------------------------------------------------------------------------------------tune hyperparmaeters for DT---------------------------------------------------------------------------------------------
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn import tree


param_gridDT = {'max_depth': np.arange(1, 30, 1),
              'max_features':np.arange(1, 30, 1),
              'criterion': ['entropy', 'gini'],
              'min_samples_leaf': np.arange(0.1,0.5,0.05),
              'min_samples_split': np.arange(0.1,1,0.05),
             }

# comb = 1
# for list_ in param_gridDT.values():
#     comb *= len(list_)
# print(comb)
# res = pd.DataFrame()
# random_searchDT = RandomizedSearchCV(estimator=DecisionTreeClassifier(random_state=42),
#                            param_distributions=param_gridDT,
#                            refit=True,
#                            cv=5,  random_state=123, n_iter=3000, return_train_score=True)
# random_searchDT.fit(X_train, y_train)
# best_model = random_searchDT.best_estimator_
# print(best_model)
# trainPreds = best_model.predict(X_train)
# print("Train accuracy: ", accuracy_score(y_train, trainPreds))
# preds = best_model.predict(X_validation)
# print("Test accuracy: ", accuracy_score(y_validation, preds))

#
# featureNames=['age','gender','is_cp_0','is_cp_1','is_cp_2','is_cp_3','is_trestbps_0','is_trestbps_1','is_trestbps_2','is_trestbps_3','is_trestbps_4','is_chol_0','is_chol_1','is_chol_2','fbs','is_restecg_0','is_restecg_1','is_restecg_2','thalach','exang','newFeature','is_ca_0','is_ca_1','is_ca_2','is_ca_3','is_ca_4','is_thal_1','is_thal_2','is_thal_3']
# # chosenModel=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
# #                        max_depth=3, max_features=16, max_leaf_nodes=None,
# #                        min_impurity_decrease=0.0, min_impurity_split=None,
# #                        min_samples_leaf=0.1,
# #                        min_samples_split=0.15000000000000002,
# #                        min_weight_fraction_leaf=0.0, presort='deprecated',
# #                        random_state=42, splitter='best')
# # chosenModel.fit(X_train, y_train)
# # print(chosenModel.feature_importances_)
# # trainPreds = chosenModel.predict(X_train)
# # preds = chosenModel.predict(X_validation)
# # print("Train accuracy: ", round(accuracy_score(y_train, trainPreds), 3))
# # print("Test accuracy: ", round(accuracy_score(y_validation, preds), 3))
# # plt.figure(figsize=(7, 6))
# # plot_tree(chosenModel, filled=True, class_names=True, feature_names=featureNames)
# # plt.show()
#
#
# def GridSearch_table_plot(grid_clf, param_name,
#                           num_results=15,
#                           negative=True,
#                           graph=True,
#                           display_all_params=True):
#
#     '''Display grid search results
#
#     Arguments
#     ---------
#
#     grid_clf           the estimator resulting from a grid search
#                        for example: grid_clf = GridSearchCV( ...
#
#     param_name         a string with the name of the parameter being tested
#
#     num_results        an integer indicating the number of results to display
#                        Default: 15
#
#     negative           boolean: should the sign of the score be reversed?
#                        scoring = 'neg_log_loss', for instance
#                        Default: True
#
#     graph              boolean: should a graph be produced?
#                        non-numeric parameters (True/False, None) don't graph well
#                        Default: True
#
#     display_all_params boolean: should we print out all of the parameters, not just the ones searched for?
#                        Default: True
#
#     Usage
#     -----
#
#     GridSearch_table_plot(grid_clf, "min_samples_leaf")
#
#                           '''
#     from matplotlib      import pyplot as plt
#     from IPython.display import display
#     import pandas as pd
#
#     clf = grid_clf.best_estimator_
#     clf_params = grid_clf.best_params_
#     if negative:
#         clf_score = -grid_clf.best_score_
#     else:
#         clf_score = grid_clf.best_score_
#     clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
#     cv_results = grid_clf.cv_results_
#
#     print("best parameters: {}".format(clf_params))
#     print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
#     if display_all_params:
#         import pprint
#         pprint.pprint(clf.get_params())
#
#     # pick out the best results
#     # =========================
#     scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')
#
#     best_row = scores_df.iloc[0, :]
#     if negative:
#         best_mean = -best_row['mean_test_score']
#     else:
#         best_mean = best_row['mean_test_score']
#     best_stdev = best_row['std_test_score']
#     best_param = best_row['param_' + param_name]
#
#     # display the top 'num_results' results
#     # =====================================
#     display(pd.DataFrame(cv_results) \
#             .sort_values(by='rank_test_score').head(num_results))
#
#     # plot the results
#     # ================
#     scores_df = scores_df.sort_values(by='param_' + param_name)
#
#     if negative:
#         means = -scores_df['mean_test_score']
#     else:
#         means = scores_df['mean_test_score']
#     stds = scores_df['std_test_score']
#     params = scores_df['param_' + param_name]
#
#     # plot
#     if graph:
#         plt.figure(figsize=(8, 8))
#         plt.errorbar(params, means, yerr=stds)
#
#         plt.axhline(y=best_mean + best_stdev, color='red')
#         plt.axhline(y=best_mean - best_stdev, color='red')
#         plt.plot(best_param, best_mean, 'or')
#
#         plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score), fontsize=25)
#         plt.xlabel(param_name, fontsize = 20)
#         plt.ylabel('Score', fontsize=20)
#         plt.show()
#
# # GridSearch_table_plot(random_searchDT, "max_depth", negative=False)
# # GridSearch_table_plot(random_searchDT, "max_features", negative = False)
# # GridSearch_table_plot(random_searchDT, "min_samples_leaf", negative = False)
#
# #Show Results
# results_dt=pd.DataFrame(random_searchDT.cv_results_)
# results_dt=results_dt.drop(['mean_fit_time','std_fit_time','mean_score_time','std_score_time','rank_test_score'],1)
# #results_dt.to_csv("C:/Users/talco/Desktop/results.csv")
#
#
# #Check ACC
# best_model_dt = random_searchDT.best_estimator_
# print("The Best model is:")
# print(best_model_dt)
# train_acc = best_model_dt.score(X_train, y_train)
# test_acc = best_model_dt.score(X_validation, y_validation)
#
# print("Train accuracy: ", round(train_acc, 3))
# print("Test accuracy: ", round(test_acc, 3))
#
# ax1=sns.lineplot(x='param_max_depth',y='mean_test_score',hue='param_criterion',data=results_dt)
# plt.title("Random Search DT Graph-Test", fontsize=25)
# ax1.set_xlabel('Max Depth', fontsize=20)
# ax1.set_ylabel('Mean Test Score', fontsize=20)
# plt.show()
#
# ax2=sns.lineplot(x='param_max_depth',y='mean_train_score',hue='param_criterion',data=results_dt)
# plt.title("Grid Search DT Graph-Train", fontsize=25)
# ax2.set_xlabel('Max Depth', fontsize=20)
# ax2.set_ylabel('Mean Train Score ', fontsize=20)
# plt.show()
#
# ax1=sns.lineplot(x='param_min_samples_leaf',y='mean_test_score',hue='param_criterion',data=results_dt)
# plt.title("Random Search DT Graph-Test", fontsize=25)
# ax1.set_xlabel('Min Sample Leaf', fontsize=20)
# ax1.set_ylabel('Mean Test Score', fontsize=20)
# plt.show()
#
# ax2=sns.lineplot(x='param_min_samples_leaf',y='mean_train_score',hue='param_criterion',data=results_dt)
# plt.title("Grid Search DT Graph-Train", fontsize=25)
# ax2.set_xlabel('Min Sample Leaf', fontsize=20)
# ax2.set_ylabel('Mean Train Score', fontsize=20)
# plt.show()
#
# ax1=sns.lineplot(x='param_max_features',y='mean_test_score',hue='param_criterion',data=results_dt)
# plt.title("Random Search DT Graph-Test", fontsize=25)
# ax1.set_xlabel('Max Features', fontsize=20)
# ax1.set_ylabel('Mean Test Score', fontsize=20)
# plt.show()
#
# ax2=sns.lineplot(x='param_max_features',y='mean_train_score',hue='param_criterion',data=results_dt)
# plt.title("Grid Search DT Graph-Train", fontsize=25)
# ax2.set_xlabel('Max Features', fontsize=20)
# ax2.set_ylabel('Mean Train Score', fontsize=20)
# plt.show()
#
# ax1=sns.lineplot(x='param_min_samples_split',y='mean_test_score',hue='param_criterion',data=results_dt)
# plt.title("Random Search DT Graph-Test", fontsize=25)
# ax1.set_xlabel('Min Sample Split', fontsize=20)
# ax1.set_ylabel('Mean Test Score', fontsize=20)
# plt.show()
#
# ax2=sns.lineplot(x='param_min_samples_split',y='mean_train_score',hue='param_criterion',data=results_dt)
# plt.title("Grid Search DT Graph-Train", fontsize=25)
# ax2.set_xlabel('Min Sample Split', fontsize=20)
# ax2.set_ylabel('Mean Train Score', fontsize=20)
# plt.show()
#--------------------------------------------------------------------------------neural network-----------------------------------------------------------------------------------

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# #standrized Set (new)
scaler = MinMaxScaler()
standrized=scaler.fit(X_train)
x_train_s=standrized.transform(X_train)
x_train_s=pd.DataFrame(x_train_s,columns=X_train.columns)
x_test_s=standrized.transform(X_validation)
x_test_s=pd.DataFrame(x_test_s,columns=X_validation.columns)
#
#
# model = MLPClassifier(random_state=123)
# model.fit(scaler.transform(X_train), y_train)
# print("Train accuracy: ", round(model.score(x_train_s, y_train), 3))
# print("Test accuracy: ", round(model.score(x_test_s, y_validation), 3))
#--------------------------------------------------------------------hidden layer dize----------------------------------------------------------------------------
tupleArray = []
# for i in np.arange(1,21,1):
#     tupleArray.insert(i,i)
# #print(tupleArray)
#
# tupleArray1 = []
# index = 1
# for i in np.arange(1,21,1):
#     for j in np.arange(1,21,1):
#         tupleArray1.insert(index, (i,j))
#
# # print(tupleArray1)
#
# sizes=tupleArray+tupleArray1
# #print(sizes)
# train_accs = []
# test_accs = []
# res = pd.DataFrame()
# for size_ in np.arange(1,100,2):
#     print(f"size: {size_}")
#     model = MLPClassifier(random_state=123,
#                           hidden_layer_sizes=size_,
#                           max_iter=200,
#                           activation='relu',
#                           verbose=False,
#                           learning_rate_init=0.001,
#                           alpha=0.00)
#     model.fit(scaler.transform(X_train), y_train)
#     train_acc = model.score(x_train_s, y_train)
#     train_accs.append(train_acc)
#     test_acc = model.score(x_test_s, y_validation)
#     test_accs.append(test_acc)
#     res = res.append({'size': size_,
#                        'train_acc':model.score(x_train_s, y_train),
#                        'test_acc':model.score(x_test_s, y_validation)}, ignore_index=True)

# plt.figure(figsize=(7, 4))
# plt.plot(sizes, train_accs, label='Train')
# plt.plot(sizes, test_accs, label='Test')
# plt.title("Accuracy and Hidden Layer Sizes", fontsize=25)
# plt.xlabel("Hidden Layer Size", fontsize=20)
# plt.ylabel("Accuracy",fontsize=20)
# plt.legend()
# plt.show()
# print(res.sort_values('test_acc', ascending=False))
# print(train_accs)
#---------------------------------------------------------------------------------------- activation function----------------------------------------------------------------------
# resActivation = pd.DataFrame()
# activationList = ['identity','logistic','tanh','relu']
# for function in activationList:
#     print(function)
#     model = MLPClassifier(random_state=123,
#                           hidden_layer_sizes=(100),
#                           max_iter=200,
#                           activation=function,
#                           verbose=False,
#                           learning_rate_init=0.001)
#     model.fit(x_train_s, y_train)
#     train_acc = model.score(scaler.transform(X_train), y_train)
#     test_acc = model.score(scaler.transform(X_validation), y_validation)
#     print("Train accuracy: ", round(model.score(scaler.transform(X_train), y_train), 3))
#     print("Test accuracy: ", round(model.score(scaler.transform(X_validation), y_validation), 3))
#     resActivation = resActivation.append({'Activation Function': function,
#                       'train_acc': model.score(x_train_s, y_train),
#                       'test_acc': model.score(x_test_s, y_validation)}, ignore_index=True)
# plt.figure(figsize=(7, 4))
# plt.plot(resActivation['Activation Function'], resActivation['train_acc'], label='Train')
# plt.plot(resActivation['Activation Function'], resActivation['test_acc'], label='Test')
# plt.title("Accuracy and Hidden Activation Function", fontsize=25)
# plt.xlabel("Activation Function", fontsize=20)
# plt.ylabel("Accuracy",fontsize=20)
# plt.legend()
# plt.show()
# print(resActivation.sort_values('test_acc', ascending=False))
# print(train_accs)
#---------------------------------------------------------------------------------max iterations-----------------------------------------------------------------------------------------
# resIterations = pd.DataFrame()
# nIterList = np.arange(100, 1000, 100)
# for nIter in nIterList:
#     print(nIter)
#     model = MLPClassifier(random_state=123,
#                           hidden_layer_sizes=(100),
#                           max_iter=nIter,
#                           activation='relu',
#                           verbose=False,
#                           learning_rate_init=0.0001)
#     model.fit(x_train_s, y_train)
#     train_acc = model.score(x_train_s, y_train)
#     test_acc = model.score(x_test_s, y_validation)
#     print("Train accuracy: ", round(model.score(x_train_s, y_train), 3))
#     print("Test accuracy: ", round(model.score(x_test_s, y_validation), 3))
#     resIterations = resIterations.append({'nIter': nIter,
#                       'train_acc': train_acc,
#                       'test_acc': test_acc}, ignore_index=True)
#
# plt.figure(figsize=(7, 4))
# plt.plot(resIterations['nIter'], resIterations['train_acc'], label='Train')
# plt.plot(resIterations['nIter'],resIterations['test_acc'], label='Test')
# plt.title("Accuracy and Number of Iterations", fontsize=25)
# plt.xlabel("Iterations", fontsize=20)
# plt.ylabel("Accuracy", fontsize=20)
# plt.legend()
# plt.show()
# print(resIterations.sort_values('test_acc', ascending=False))
#----------------------------------------------------------hidden layers numbers-------------------------------------------------------------------------

#
# hiddenLayers = [(100), (100,100), (100,100,100)]
# count = 1
# for hiddenLayer in hiddenLayers:
#     print("number of hidden layers = ", count)
#     model = MLPClassifier(random_state=123,
#                           hidden_layer_sizes=hiddenLayer,
#                           max_iter=200,
#                           activation='relu',
#                           verbose=False,
#                           learning_rate_init=0.0001)
#     model.fit(x_train_s, y_train)
#     train_acc = model.score(x_train_s, y_train)
#     test_acc = model.score(x_test_s, y_validation)
#     print("Train accuracy: ", round(train_acc, 3))
#     print("Test accuracy: ", round(test_acc, 3))
#     count+=1
# ----------------------------------------------------------learning rate-----------------------------------------------------------------------------------
# learningRateRes = pd.DataFrame()
# learningRateList = np.arange(0.0001, 0.001, 0.0001)
# for learningRate in learningRateList:
#     print(learningRate)
#     model = MLPClassifier(random_state=123,
#                           hidden_layer_sizes=(100),
#                           max_iter=200,
#                           activation='relu',
#                           verbose=False,
#                           learning_rate_init=learningRate)
#     model.fit(x_train_s, y_train)
#     train_acc = model.score(x_train_s, y_train)
#     test_acc = model.score(x_test_s, y_validation)
#     print("Train accuracy: ", round(model.score(x_train_s, y_train), 3))
#     print("Test accuracy: ", round(model.score(x_test_s, y_validation), 3))
#     learningRateRes = learningRateRes.append({'Learning Rate': learningRate,
#                       'train_acc': train_acc,
#                       'test_acc': test_acc}, ignore_index=True)
#
# plt.figure(figsize=(7, 4))
# plt.plot(learningRateRes['Learning Rate'], learningRateRes['train_acc'], label='Train')
# plt.plot(learningRateRes['Learning Rate'],learningRateRes['test_acc'], label='Test')
# plt.title("Accuracy and Learning Rate", fontsize=25)
# plt.xlabel("Learning Rate", fontsize=20)
# plt.ylabel("Accuracy", fontsize=20)
# plt.legend()
# plt.show()
# print(learningRateRes.sort_values('test_acc', ascending=False))


#-------------------------------------------------------------------------tuning------------------------------------------------------------------------------------------------
# tupleArray = []
# for i in np.arange(1,21,1):
#     tupleArray.insert(i,i)
# #print(tupleArray)
#
# tupleArray1 = []
# index = 1
# for i in np.arange(1,21,1):
#     for j in np.arange(1,21,1):
#         tupleArray1.insert(index, (i,j))

# print(tupleArray1)

# sizes=tupleArray+tupleArray1
# print(sizes)
#
# param_gridANN1 = {'hidden_layer_sizes': np.arange(1,100,5),
#               'activation': ['identity', 'logistic','relu','tanh'],
#               'max_iter': np.arange(100,1000,100),
#               'learning_rate_init': np.arange(0.0001, 0.0009, 0.0001),}
# comb = 1
# for list_ in param_gridANN1.values():
#     comb *= len(list_)
# print(comb)
# res = pd.DataFrame()
#
# random_searchANN = RandomizedSearchCV(MLPClassifier(random_state=42),
#                                    param_distributions=param_gridANN1, cv=5,
#                                    random_state=123, n_iter=500, refit=True, return_train_score=True)
#
# random_searchANN.fit(x_train_s, y_train)
# print(random_searchANN.best_estimator_)
# trainPreds = random_searchANN.predict(x_train_s)
# preds = random_searchANN.predict(x_test_s)
# print("Train accuracy: ", round(accuracy_score(y_train, trainPreds), 3))
# print("Test accuracy: ", round(accuracy_score(y_validation, preds), 3))

# model = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#               beta_2=0.999, early_stopping=False, epsilon=1e-08,
#               hidden_layer_sizes=15, learning_rate='constant',
#               learning_rate_init=0.0007000000000000001, max_fun=15000,
#               max_iter=400, momentum=0.9, n_iter_no_change=10,
#               nesterovs_momentum=True, power_t=0.5, random_state=42,
#               shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
#               verbose=False, warm_start=False)
# model.fit(scaler.transform(X_train), y_train)
# train_acc = model.score(scaler.transform(X_train), y_train)
# test_acc = model.score(scaler.transform(X_validation), y_validation)
# print("Train accuracy: ", round(model.score(scaler.transform(X_train), y_train), 3))
# print("Test accuracy: ", round(model.score(scaler.transform(X_validation), y_validation), 3))

#Show Results
# results_ann=pd.DataFrame(random_searchANN.cv_results_)
# results_ann=results_ann.drop(['mean_fit_time','std_fit_time','mean_score_time','std_score_time','rank_test_score'],1)
#results_ann.to_csv("C:/Users/talco/Desktop/resultsANN.csv")

#Check ACC
# best_model_ann = random_searchANN.best_estimator_
# print("The Best model is:")
# print(best_model_ann)
#
# train_acc = best_model_ann.score(x_train_s, y_train)
# test_acc = best_model_ann.score(x_test_s, y_validation)
#
# print("Train accuracy: ", round(train_acc, 3))
# print("Test accuracy: ", round(test_acc, 3))

#GridSearch_table_plot(random_searchANN, "hidden_layer_sizes", negative=False)
# GridSearch_table_plot(random_searchANN, "activation", negative = False)
# GridSearch_table_plot(random_searchANN, "max_iter", negative = False)
# GridSearch_table_plot(random_searchANN, "learning_rate_init", negative = False)

# ax1=sns.lineplot(x='param_hidden_layer_sizes',y='mean_test_score',hue='param_activation',data=results_ann)
# plt.title("Random Search ANN Graph-Test", fontsize=25)
# ax1.set_xlabel('Hidden Layer Sizes', fontsize=20)
# ax1.set_ylabel('Mean Test Score', fontsize=20)
# plt.show()
#
# ax2=sns.lineplot(x='param_hidden_layer_sizes',y='mean_train_score',hue='param_activation',data=results_ann)
# plt.title("Grid Search ANN Graph-Train", fontsize=25)
# ax2.set_xlabel('Hidden Layer Sizes', fontsize=20)
# ax2.set_ylabel('Mean Train Score', fontsize=20)
# plt.show()

# ax1=sns.lineplot(x='param_max_iter',y='mean_test_score',hue='param_activation',data=results_ann)
# plt.title("Random Search ANN Graph-Test", fontsize=25)
# ax1.set_xlabel('Max Iterations', fontsize=20)
# ax1.set_ylabel('Mean Test Score', fontsize=20)
# plt.show()
#
# ax2=sns.lineplot(x='param_max_iter',y='mean_train_score',hue='param_activation',data=results_ann)
# plt.title("Grid Search ANN Graph-Train", fontsize=25)
# ax2.set_xlabel('Max Iterations', fontsize=20)
# ax2.set_ylabel('Mean Train Score', fontsize=20)
# plt.show()
#
# ax1=sns.lineplot(x='param_learning_rate_init',y='mean_test_score',hue='param_activation',data=results_ann)
# plt.title("Random Search ANN Graph-Test", fontsize=25)
# ax1.set_xlabel('Learning Rate', fontsize=20)
# ax1.set_ylabel('Mean Test Score', fontsize=20)
# plt.show()
#
# ax2=sns.lineplot(x='param_learning_rate_init',y='mean_train_score',hue='param_activation',data=results_ann)
# plt.title("Grid Search ANN Graph-Train", fontsize=25)
# ax2.set_xlabel('Learning Rate', fontsize=20)
# ax2.set_ylabel('Mean Train Score', fontsize=20)
# plt.show()

#-------------------------------------------------------kmeans----------------------------------------------------
#
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

pca = PCA(n_components=2)
pca.fit(x_train_s)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
X_pca = pca.transform(x_train_s)
X_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
X_pca['target'] = y_train
# sns.scatterplot(x='PC1', y='PC2', hue='target', data=X_pca)
# plt.show()
#
# kmeans = KMeans(n_clusters=2, max_iter=300, n_init=10, random_state=42)
# kmeans.fit(x_train_s)
# print(kmeans.cluster_centers_)
# print(kmeans.predict(x_train_s))
# X_pca['cluster'] = kmeans.predict(x_train_s)
#
# sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=X_pca, palette="husl")
# plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='+', s=150 ,color='red')
# plt.title('Clustering', fontsize=25)
# plt.xlabel('PC1', fontsize=20)
# plt.ylabel('PC2',fontsize=20)
# plt.show()
#
# iner_list = []
# dbi_list = []
# sil_list = []
#
# for n_clusters in np.arange(2,10,1):
#     kmeans = KMeans(n_clusters=n_clusters, max_iter=300, n_init=10, random_state=42)
#     kmeans.fit(x_train_s)
#     assignment = kmeans.predict(x_train_s)
#
#     iner = kmeans.inertia_
#     sil = silhouette_score(x_train_s, assignment)
#     dbi = davies_bouldin_score(x_train_s, assignment)
#
#     dbi_list.append(dbi)
#     sil_list.append(sil)
#     iner_list.append(iner)
#
# plt.plot(range(2, 10, 1), iner_list, marker='o')
# plt.title("Inertia")
# plt.xlabel("Number of clusters")
# plt.show()
#
# plt.plot(range(2, 10, 1), sil_list, marker='o')
# plt.title("Silhouette", fontsize=25)
# plt.xlabel("Number of clusters", fontsize=20)
# plt.show()
#
# plt.plot(range(2, 10, 1), dbi_list, marker='o')
# plt.title("Davies-bouldin", fontsize=25)
# plt.xlabel("Number of clusters",fontsize=20)
# plt.show()
#
#
kmeans = KMeans(n_clusters=2, max_iter=300, n_init=10, random_state=42)
kmeans.fit(x_train_s)
kmeans.fit(x_train_s)
assignment = kmeans.predict(x_train_s)
print(silhouette_score(x_train_s, assignment))
print(davies_bouldin_score(x_train_s, assignment))
print(kmeans.predict(x_train_s))
X_pca['cluster'] = kmeans.predict(x_train_s)

sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=X_pca, palette="Accent")
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='+', s=150 ,color='red')
plt.title('Clustering', fontsize=25)
plt.xlabel('PC1', fontsize=20)
plt.ylabel('PC2',fontsize=20)
plt.show()

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

# def plot_dendrogram(model, **kwargs):
#     # Create linkage matrix and then plot the dendrogram
#
#     # create the counts of samples under each node
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for i, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[i] = current_count
#
#     linkage_matrix = np.column_stack([model.children_, model.distances_,
#                                       counts]).astype(float)
#     # Plot the corresponding dendrogram
#     hierarchy.dendrogram(linkage_matrix, **kwargs)
#
#
# agglomerativeClustering =AgglomerativeClustering(distance_threshold=20, n_clusters=None)
# model = agglomerativeClustering.fit(x_train_s)
# plt.title('Hierarchical Clustering Dendrogram', fontsize=25)
# # plot the top three levels of the dendrogram
# plot_dendrogram(model, truncate_mode='level', p=3)
# plt.xlabel("Number of points in node (or index of point if no parenthesis).", fontsize = 20)
# plt.show()

# sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=X_pca, palette="Set2")
# plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='+', s=150 ,color='red')
# plt.title('Agglomerative Clustering', fontsize=25)
# plt.xlabel('PC1', fontsize=20)
# plt.ylabel('PC2',fontsize=20)
# plt.show()

# assignment=agglomerativeClustering.fit_predict(x_train_s)
# print(silhouette_score(x_train_s, assignment))
# print(davies_bouldin_score(x_train_s, assignment))


# kmeans = KMeans(n_clusters=2, max_iter=300, n_init=10, random_state=421)
# kmeans.fit(x_train_s)
# assignment = kmeans.predict(x_train_s)
# print(assignment)
# print(y_train)

# # counter = 0
# # for i in np.arange(1, len(assignment), 1):
# #     if assignment[i] == y_train.iloc[i]:
# #         counter=counter+1
# #
# # print(counter/len(y_train))
#
#
# #with Normalization And dummies
# #Train model
kmeans = KMeans(n_clusters=2, random_state=421)
k_model=kmeans.fit(x_train_s)
preds_train = k_model.predict(x_train_s)
print("Train accuracy: ", round(accuracy_score(y_train, preds_train), 3))
preds_test = k_model.predict(x_test_s)
print("Test accuracy: ", round(accuracy_score(y_validation, preds_test), 3))
#
# #Find PCA  model
# #the explaind Var for Standarized Data is VeryLow
pca = PCA(n_components=2,random_state=421)
pca=pca.fit(x_train_s)
print("The explained variance of each PCA is: ",pca.explained_variance_ratio_)
print("The explained variance (together - covariance) is: ",pca.explained_variance_ratio_.sum())
#
# #Train and Test Pca Data && Show Scatter for PCA DATA
train_pca = pca.transform(x_train_s)
train_pca = pd.DataFrame(train_pca, columns=['PC1', 'PC2'])
train_pca['y'] = y_train
#
# sns.scatterplot(x='PC1', y='PC2', hue='y', data=train_pca)
# plt.title("PCA grpha for Train Data")
# plt.show()
#
test_pca = pca.transform(x_test_s)
test_pca = pd.DataFrame(test_pca, columns=['PC1', 'PC2'])
test_pca['y'] = y_validation
print(test_pca)
# sns.scatterplot(x='PC1', y='PC2', hue='y', data=test_pca)
# plt.title("PCA grpha for Test Data")
# plt.show()
#
# #add Predicitons to Train and test PCA DataSet
train_pca['cluster'] = preds_train
test_pca['cluster'] = preds_test

#Dont think that does Scatters are good for anyThing but...

#scatterplot with Centers for TrainData
#sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=train_pca, palette='Accent')
#plt.scatter(pca.transform(k_model.cluster_centers_)[:, 0], pca.transform(k_model.cluster_centers_)[:, 1], marker='+', s=100 ,color='red')
#plt.show()

#scatterplot with Centers for TrainData
#sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=test_pca, palette='Accent')
#plt.scatter(pca.transform(k_model.cluster_centers_)[:, 0], pca.transform(k_model.cluster_centers_)[:, 1], marker='+', s=100 ,color='red')
#plt.show()

from tqdm import tqdm
x_test = np.linspace(-1.5,2, 100)
y_test = np.linspace(-1.5,2, 100)
predictions = pd.DataFrame()
for x in tqdm(x_test):
    for y in y_test:
        pred = kmeans.predict(pca.inverse_transform(np.array([x, y])).reshape(-1, 29))[0]
        predictions = predictions.append(dict(X1=x, X2=y, y=pred), ignore_index=True)

#
# #Areas Graph for Train
plt.scatter(x=predictions[predictions.y == 0]['X1'], y = predictions[predictions.y == 0]['X2'], c='ivory')
plt.scatter(x=predictions[predictions.y == 1]['X1'], y = predictions[predictions.y == 1]['X2'], c='powderblue')
sns.scatterplot(x='PC1', y='PC2', hue='y', data=train_pca)
plt.scatter(pca.transform(k_model.cluster_centers_)[:, 0], pca.transform(k_model.cluster_centers_)[:, 1], marker='+', s=200 ,color='red')
plt.title("PCA Graph for Train Data After Kmeans", fontsize = 25)
plt.show()


#Areas Graph for Test
plt.scatter(x=predictions[predictions.y == 0]['X1'], y = predictions[predictions.y == 0]['X2'], c='ivory')
plt.scatter(x=predictions[predictions.y == 1]['X1'], y = predictions[predictions.y == 1]['X2'], c='powderblue')
sns.scatterplot(x='PC1', y='PC2', hue='y', data=test_pca)
plt.scatter(pca.transform(k_model.cluster_centers_)[:, 0], pca.transform(k_model.cluster_centers_)[:, 1], marker='+', s=200 ,color='red')
plt.title("PCA Graph for Test Data After Kmeans", fontsize = 25)
plt.show()

# model = MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
#               beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
#               hidden_layer_sizes=(10, 20), learning_rate='constant',
#               learning_rate_init=0.0006000000000000001, max_fun=15000,
#               max_iter=700, momentum=0.9, n_iter_no_change=10,
#               nesterovs_momentum=True, power_t=0.5, random_state=42,
#               shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
#               verbose=False, warm_start=False)
# model.fit(x_train_s, y_train)
# print("Train accuracy: ", round(model.score(x_train_s, y_train), 3))
# print("Test accuracy: ", round(model.score(x_test_s, y_validation), 3))
# from sklearn.metrics import accuracy_score, confusion_matrix
# print(f"Accuracy: {accuracy_score(y_true=y_validation, y_pred=model.predict(x_test_s)):.3f}")
# print(confusion_matrix(y_true=y_validation, y_pred=model.predict(x_test_s)))
# print(f"Accuracy: {accuracy_score(y_true=y_train, y_pred=model.predict(x_train_s)):.3f}")
# print(confusion_matrix(y_true=y_train, y_pred=model.predict(x_train_s)))
# print(model.predict(x_test_s))
# print(y_validation)
#
# X_test = pd.read_csv("C:/Users/talco/Desktop/MLProject/X_test2_dummy.csv")
# X_test = X_test.drop(['id'],1)
# x_test_final = standrized.transform(X_test)
# x_test_final=pd.DataFrame(x_test_final,columns=X_test.columns)
# print(model.predict(x_test_final))

