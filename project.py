import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from scipy.stats import gmean
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import DMatrix


df = pd.read_csv("processed.csv", header=0, index_col="ID")
#df.TARGET.describe()

y = df["TARGET"].values
X = df.ix[:, "var3":"var38"].values
X_labels = df.ix[:, "var3":"var38"].columns.values

lr = LassoLarsCV()
sfm = SelectFromModel(lr, threshold=1e-3)
X_std = StandardScaler().fit_transform(X, y)
sfm.fit(X_std,y)
lr.fit(X_std, y)

#feat_imp = pd.DataFrame(lr.coef_, index=X_labels)
#feat_imp.plot(kind="bar", title="Feature Importance", use_index=False)

chosen_feat = [ f for i,f in enumerate(X_labels) if sfm.get_support()[i] ]
#chosen_feat = pickle.load(open("feat", "rb"))
print(len(chosen_feat))
chosen_feat

# kaggle forum
df.var3 = df.var3.replace(-999999,2)
y = df["TARGET"].values
X = df.ix[:, "var3":"var38"].values
X_labels = df.ix[:, "var3":"var38"].columns.values


test = pd.read_csv("processed_test.csv", header=0, index_col="ID")
test.var3 = test.var3.replace(-999999,2)

X_test = test[chosen_feat].values

X_sel = df[chosen_feat].values

stage2_train = pd.DataFrame(index=df.index)
stage2_test = pd.DataFrame(index=test.index)

# Linear model
logmodel = Pipeline([
        ("scl", StandardScaler()),
        ("clf", LogisticRegressionCV(penalty="l2", cv=5, max_iter=500, scoring="roc_auc", 
                                     n_jobs=-1 ,random_state=1))
    ])

logmodel.fit(X_sel,y)
#logmodel = pickle.load(open("logmodel.p", "rb"))
temp = logmodel.predict_proba(X_sel)[:,1]

s = cross_val_score(logmodel, X_sel, y, scoring="roc_auc", cv=5) 
print(s)
print( "AUC score", gmean(s) )

stage2_train["LogModel"] = pd.DataFrame(temp, index=df.index)

stage2_test["LogModel"] = logmodel.predict_proba(X_test)[:,1]

xgbc = xgb.XGBClassifier(max_depth=5, n_estimators=200, learning_rate=0.03, nthread=4, 
                          subsample=0.6815, colsample_bytree=0.701, seed=1234)
xgbc.fit(X, y, eval_metric="auc", verbose=2)
#xgbc = pickle.load( open("xgbc.p", "rb") )
temp = xgbc.predict_proba(X_sel)[:, 1]

s = cross_val_score(xgbc, X_sel, y, scoring="roc_auc", cv=5) 
print(s)
print( "AUC score", gmean(s) )

stage2_train["XGBoost1"] = pd.DataFrame(temp, index=df.index)

stage2_test["XGBoost1"] = xgbc.predict_proba(X_test)[:,1]

xgbc2 = xgb.XGBClassifier(max_depth=5, n_estimators=70, learning_rate=0.1, nthread=4, gamma=0.2,
                          subsample=0.4, colsample_bytree=0.7, min_child_weight=1, seed=1)

xgbc2.fit(X_sel, y, eval_metric="auc", verbose=0)
#xgbc2 = pickle.load(open("xgbc2.p", "rb"))
temp = xgbc2.predict_proba(X_sel)[:, 1]

s = cross_val_score(xgbc2, X_sel, y, scoring="roc_auc", cv=5) 
print(s)
print( "AUC score", gmean(s) )

stage2_train["XGBoost2"] = pd.DataFrame(temp, index=df.index)

stage2_test["XGBoost2"] = xgbc2.predict_proba(X_test)[:,1]

x = pd.DataFrame(xgbc2.feature_importances_, index=chosen_feat)
x.sort_values(by=0, inplace=True, ascending=False)
#x.plot(kind="bar")

rfc = RandomForestClassifier(n_estimators=10, criterion="entropy", max_features=None, max_depth=7,
                             min_samples_leaf=9, n_jobs=4, random_state=1)
rfc.fit(X_sel, y)
#rfc = pickle.load( open("rfc.p", "rb"))
temp = rfc.predict_proba(X_sel)[:, 1]

s = cross_val_score(rfc, X_sel, y, scoring="roc_auc", cv=5) 
print(s)
print( "AUC score", gmean(s) )

stage2_train["RandomForest"] = pd.DataFrame(temp, index=df.index)
stage2_test["RandomForest"] = rfc.predict_proba(X_test)[:,1]
temp = pd.read_csv("simplexgbtrain.csv", header=0, index_col="ID")
stage2_train["XGBoost3"] = temp.PREDICTION.values
temp = pd.read_csv("simplexgbtest.csv", header=0, index_col="ID")
stage2_test["XGBoost3"] = temp.TARGET.values
temp = pd.read_csv("trainR1.csv", header=0)
stage2_train["XGBoostR1"] = temp.TARGET.values
temp = pd.read_csv("testR1.csv", header=0, index_col="ID")
stage2_test["XGBoostR1"] = temp.TARGET.values

stage2_X = stage2_train.ix[:, "LogModel": "XGBoostR1"].values
stage2_X_train, stage2_X_eval, y_train, y_eval = train_test_split(stage2_X, y, test_size=0.2)

stage2_test.to_csv("stage2_test.csv")
stage2_train.to_csv("stage2_train.csv")

x = xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.01, nthread=4, 
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, seed=1)

x.fit(stage2_X, y, early_stopping_rounds=20, eval_metric="auc", eval_set=[(stage2_X_eval, y_eval)], verbose=0)

s = cross_val_score(x, stage2_X, y, scoring="roc_auc", cv=5) 
print(s)
print( "AUC score", gmean(s) )

#pd.DataFrame(x.feature_importances_, index=stage2_train.columns).plot(kind="bar")

stage2_X_test = stage2_test.ix[:, "LogModel": "XGBoostR1"].values
y_pred = x.predict_proba(stage2_X_test)[:,1]
submission = pd.DataFrame(index=test.index)
submission["TARGET"] = y_pred
submission.to_csv("stack.csv")


