#coding=utf-8
import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score,f1_score,recall_score,classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV,KFold
from imblearn.over_sampling import SMOTE
import shap
import matplotlib
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import pickle
from sklearn.preprocessing import OrdinalEncoder

# 载入数据集
df = pd.read_csv('E:/feature_speech_act/feature1.csv',encoding='GB2312', header=0)
df_display=pd.read_csv('E:/feature_speech_act/feature.csv', encoding='GB2312')
X_display=df_display.drop(["label"], axis=1)
X = df.drop(["label"], axis=1)
Y = df["label"]

# 把数据集拆分成训练集和测试集
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#运用SMOTE算法实现训练数据集的平衡
over_samples=SMOTE(random_state=123)
over_samples_X,over_samples_y = over_samples.fit_resample(X_train, y_train)
# 重抽样前的类别比例
print(y_train.value_counts()/len(y_train))
# 重抽样后的类别比例
print(pd.Series(over_samples_y).value_counts()/len(over_samples_y))

#模型测试
kfold = KFold(n_splits=10)
def gridSearch_vali(model,param_grid,cv=kfold):
    print("parameters:{}".format(param_grid))
    grid_search = GridSearchCV(estimator=model,param_grid=param_grid,cv=kfold)
    grid_search.fit(X_train,y_train)
    print(grid_search.best_params_)
    return grid_search.best_params_

xgbc_param_temp = {'reg_alpha':[0.5],'reg_lambda': [0.4],'n_estimators':[250], 'colsample_bytree':[0.6],'subsample': [0.7],'gamma':[1.2],'max_depth':[8],'learning_rate':[0.1],'min_child_weight':[8]}
# 拟合XGBoost模型 logloss','auc',
model = XGBClassifier(eval_metric=['mlogloss'],use_label_encoder=False)
model.set_params(**gridSearch_vali(model,xgbc_param_temp))
model.fit(X_train, y_train)

# 对测试集做预测
y_pred = model.predict(X_test, validate_features=False)
predictions = [round(value) for value in y_pred]

# 评估预测结果
print(classification_report(y_test, predictions,digits=4))
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#可视化
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

#多预测的解释
shap.summary_plot(shap_values, X, plot_type="bar",max_display=25,class_names=['阐述','表达','指令','承诺','其他'], class_inds=[0,1,2,3,4])
shap.summary_plot(shap_values[0], X, max_display=15)
shap.summary_plot(shap_values[1], X, max_display=15)
shap.summary_plot(shap_values[2], X, max_display=15)
shap.summary_plot(shap_values[3], X, max_display=15)
shap.summary_plot(shap_values[4], X, max_display=15)

