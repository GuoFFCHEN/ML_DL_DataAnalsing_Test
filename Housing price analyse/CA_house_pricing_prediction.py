# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:27:03 2021

@author: Administrator
"""


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,StandardScaler,OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from keras.layers import Dense
from keras.models import Sequential

def load_data(path):
    data = pd.read_csv(path)
    return data
def hist_plot(dataframe):
    dataframe.hist(bins=50,figsize=(20,15))
def hist_plot2(dataframe):
    sns.set()
    fig,axes = plt.subplots(3,3,figsize=(21,21))
    fig.suptitle('Variation of house')
    sublist = dataframe.columns.values.tolist()
    for i in range(3):
        for j in range(3):
            sns.histplot(data=dataframe[sublist[i+j]],ax=axes[i,j],bins=50,kde=True)          
    plt.show()
    
# def pipeline1(dataframe):
#     label_encoder = LabelEncoder()
#     onehot = OneHotEncoder()
#     data = label_encoder.fit_transform(dataframe).reshape(-1,1)
#     data = onehot.fit_transform(data).toarray()
#     return data
def pipeline2(dataframe):
    pipeline = Pipeline([
        ('inputer',SimpleImputer(strategy='median')),
        ('std_scaler',StandardScaler())
        ])
    data = pipeline.fit_transform(dataframe)
    return data


def regression_models(X_train,X_test,y_train,y_test,model):
    model = model
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    score = mean_squared_error(y_test, predictions)
    print('rmse of this {} is {}'.format(model,np.sqrt(score)))


def build_model(X_train):
    model = Sequential()
    model.add(Dense(64,activation="relu",input_shape=(X_train.shape[1],)))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
    return model


if __name__ == '__main__':
    #get data
    df = load_data(r'C:\Users\Administrator\Desktop\coding\dataSet\housing.csv')
    
    #get feature of the data
    features = df.columns.values.tolist()
    # print(features)
    # print(df.head())
    # print(df.info())  #we found that in the total_bedrooms we got missing datas
    # print(df.describe())
    
    #plot the hist
    hist_plot(df)
    
    #seperate and split the data
    labels = df['median_house_value'].to_numpy().reshape(-1,1)
    data = df.drop('median_house_value',axis=1)
    
    
    # high = np.where(data['income_cat']>5)[0]
    # print(data['income_cat'][high])
    
    house_data = df.copy()
    sns.scatterplot(x=house_data['longitude'],y=house_data['latitude'],alpha=0.2)
    corr_matrix = house_data.corr()
    # print(corr_matrix['median_house_value'].sort_values(ascending=False))
    
    #we choose 5 most relative features for the plot
    relative_features = ['median_house_value','median_income','total_rooms','housing_median_age','households']
    fig = pd.plotting.scatter_matrix(house_data[relative_features],figsize=(12,10))
    plt.show()
    
    sns.scatterplot(x=house_data['median_income'], y=house_data['median_house_value'],alpha=0.1)
    
    #收入的中位数对房价影响很大
    df['income_cat'] = np.ceil(data['median_income'])
    df['income_cat'].where(data['income_cat']<7,7.0,inplace=True) #小于7的为True，大于7的被替换为7
    
    df['income_cat'].hist()
    print(data['income_cat'].value_counts()/len(data))
    
    
    house_data['rooms_per_household'] = house_data['total_rooms'] / house_data['households']
    house_data['bedromms_per_room'] = house_data['total_bedrooms'] / house_data['total_rooms']
    house_data['population_per_household'] =house_data['population'] / house_data['households']
    
    corr_matrix = house_data.corr()
    print(corr_matrix['median_house_value'].sort_values(ascending=False))
    
    #in feature total_bedrooms we got missing datas
    print(house_data.info())
    
    labels = house_data['median_house_value']
    data = house_data.drop('median_house_value',axis=1)
    
    data_part1 = data['ocean_proximity']
    data_part2 = data.drop('ocean_proximity',axis=1)
    
    new_features = data_part2.columns.values.tolist()
    new_features.append(['ocean_proximity'])
    
    # label_encoder = LabelEncoder()
    # onehot = OneHotEncoder()
    # data_part1_transformed = label_encoder.fit_transform(data_part1).reshape(-1,1)
    # data_part1_transformed = onehot.fit_transform(data_part1_transformed).toarray()
    
    label_encoder = LabelBinarizer()
    data_part1_transformed = label_encoder.fit_transform(data_part1)
    # data_part1_transformed = pipeline1(data_part1)
    data_part2_transformed = pipeline2(data_part2)
    print(label_encoder.classes_)
    
    data_transformed = np.hstack((data_part1_transformed,data_part2_transformed))
    X_train,X_test,y_train,y_test = train_test_split(data_transformed,labels,test_size=0.2)
    
    
    
    regression_models(X_train,X_test,y_train,y_test,LinearRegression())
    regression_models(X_train,X_test,y_train,y_test,DecisionTreeRegressor())
    regression_models(X_train,X_test,y_train,y_test,RandomForestRegressor())
    regression_models(X_train,X_test,y_train,y_test,KNeighborsRegressor())
    regression_models(X_train,X_test,y_train,y_test,svm.SVC())
    
    model = build_model(X_train)
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
    model.fit(X_train,y_train,epochs=20,batch_size=64)
    
    
    #we choose randomforest as our model and randomgridcv for the params
    param_dict = {'n_estimators':range(2,20,2),'max_features':range(2,10,2),
                  'bootstrap':[False,True]}
    
    random = RandomizedSearchCV(RandomForestRegressor(),param_dict,cv=5,n_iter=20,
                                n_jobs=-1,scoring='neg_mean_squared_error')
    random.fit(X_train,y_train)
    print(random.best_params_)
    
    
    # grid = GridSearchCV(RandomForestRegressor(),param_dict,cv=5,
    #                             n_jobs=-1,scoring='neg_mean_squared_error')
    # grid.fit(X_train,y_train)
    # print(grid.best_params_)
    
    
    cvres = random.cv_results_
    for mean_socre,params in zip(cvres['mean_test_score'],cvres['params']):
        print(np.sqrt(-mean_socre),params)
    
    feature_importances = random.best_estimator_.feature_importances_
    feature_combined = data_part2.columns.values.tolist() + list(label_encoder.classes_)
    
    estimator_combined = dict(zip(feature_combined,feature_importances))
    
    final_model = random.best_estimator_
    regression_models(X_train,X_test,y_train,y_test,final_model)
































