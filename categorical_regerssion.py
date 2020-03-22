import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import combinations

data = pd.DataFrame({"A": np.random.permutation(['a', 'b', 'c'] * 100),
                     "B": np.random.permutation(['A', 'B', 'C'] * 100),  
                     "Y": np.random.uniform(0,1,300)
                     })

data_g = data.groupby(["A","B"])['Y'].agg(Total='sum', N='count').reset_index()
all_feature = data_g.columns[0:2]
two_feature_combinations = list(combinations(all_feature,2))

data_g2_dict = dict()
for combination in two_feature_combinations:
    data_g2 = data_g.groupby(list(combination))[['N','Total']].sum().reset_index()
    data_g2_dict[":".join(combination)] = data_g2

data_g1_dict = dict()
for feature in all_feature:
    data_g1 = data_g.groupby(list(feature))[['N','Total']].sum().reset_index()    
    data_g1_dict[feature] = data_g1

def gredient_coef(feature_category, all_feature, feature_dic, data_g1_dict, data_g2_dict):
    gredient_coef_dict = dict()
    if feature_category == "Intercept":
        for f in feature_dic:
            temp_df = data_g1_dict[f]
            for c in feature_dic[f]:
                gredient_coef_dict[f+":"+c] = temp_df[temp_df[f]==c]["N"].item()
        gredient_coef_dict["Intercept"] = temp_df["N"].sum()
        gredient_coef_dict["Total"] = temp_df["Total"].sum()
        return gredient_coef_dict
    feature, category = feature_category.split(":")
    other_feature = list(all_feature)
    other_feature.remove(feature)
    temp_df = data_g1_dict[feature]
    gredient_coef_dict[feature+":"+category] = temp_df[temp_df[feature]==category]["N"].item()
    gredient_coef_dict["Total"] = temp_df[temp_df[feature]==category]["Total"].item()
    gredient_coef_dict["Intercept"] = gredient_coef_dict[feature+":"+category]
    other_catgory = feature_dic[feature].copy()
    other_catgory.remove(category)
    for c in other_catgory:
        gredient_coef_dict[feature+":"+c] = 0
    for f in other_feature:
        f2 = [feature, f]
        temp_df = data_g2_dict[":".join(sorted(f2))]
        for c in feature_dic[f]:
            gredient_coef_dict[f+":"+c] = temp_df[(temp_df[f]==c) & (temp_df[feature]==category)]["N"].item()
    return gredient_coef_dict

def get_coef_matrix(gredient_coef_dict, feature_dic, data_g1_dict, data_g2_dict):
    column_names = sum([[ k + ":" + item for item in v] for k, v in feature_dic.items()], [])
    column_names.append("Intercept")
    column_names.append("Total")
    feature_df = pd.DataFrame(columns = column_names)
    column_names.remove("Total")
    for feature_category in column_names:
        gredient_coef_dict = gredient_coef(feature_category, all_feature, feature_dic, data_g1_dict, data_g2_dict)
        feature_df = feature_df.append(gredient_coef_dict,ignore_index=True,sort=False) 
    return feature_df

feature_dic = {name: list(data_g[name].unique()[1:]) for name in feature_names}
dummy_dic = {name: list(data_g[name].unique()[0]) for name in feature_names}

all_feature = tuple(feature_dic.keys())

def get_coef_matrix(feature_dic, data_g1_dict, data_g2_dict):
    column_names = sum([[ k + ":" + item for item in v] for k, v in feature_dic.items()], [])
    column_names.append("Intercept")
    column_names.append("Total")
    feature_df = pd.DataFrame(columns = column_names)
    column_names.remove("Total")
    for feature_category in column_names:
        gredient_coef_dict = gredient_coef(feature_category, all_feature, feature_dic, data_g1_dict, data_g2_dict)
        feature_df = feature_df.append(gredient_coef_dict,ignore_index=True,sort=False) 
    return feature_df

X = feature_df.iloc[:,:-1]
Y = feature_df.iloc[:,-1]         
beta = np.linalg.solve(X, Y)

##regression with all 
from sklearn.linear_model import LinearRegression
X_train = pd.get_dummies(data[["A", "B"]], drop_first=True)
Y_train = data[["Y"]]
reg = LinearRegression().fit(X_train, Y_train)                       

