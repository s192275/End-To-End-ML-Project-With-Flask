import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

bright_palette = sns.color_palette('bright')
muted_palette = sns.color_palette('muted')
sequential_palette = sns.color_palette('light:dodgerblue_r',9)
alert_palette = sns.color_palette('Reds_r',9)
paired_palette = sns.color_palette('Paired')


target = "TARGET"

xgboost_params = {"learning_rate": [0.1],
                  "max_depth": [4,5,6],
                  "n_estimators": [50,60,70],
                  "colsample_bytree": [0.5],
                  "objective": ['multi:softmax'],
                  "num_class":[8]}

lightgbm_params = {"learning_rate": [0.01,0.1],
                   "n_estimators": [400,500],
                   "colsample_bytree": [0.7, 1]}



classifiers = [('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(verbose = -1), lightgbm_params)]



"""
    knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [10, 11, None],
             "max_features": [7, 8, "sqrt"],
             "min_samples_split": [17,18,19],
             "n_estimators": [220,230,240]}
             
# classifiers = [('KNN', KNeighborsClassifier(), knn_params),
#                ("CART", DecisionTreeClassifier(), cart_params),
#                ("RF", RandomForestClassifier(max_features='sqrt'), rf_params),
#                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
#                ('LightGBM', LGBMClassifier(verbose = -1), lightgbm_params)]
"""