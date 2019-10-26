import streamlit as st
import pandas as pd
import time
#from matplotlib import pyplot
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import roc_auc_score,roc_curve,scorer,precision_recall_curve,f1_score
from sklearn.metrics import precision_score,recall_score
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

# defining models and environment runtime

@st.cache(persist=True,suppress_st_warning=True)
def train_model():
    st.success('Model trained!')

@st.cache(persist=True,suppress_st_warning=True)
def read_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    # show data
    st.success('Dataset in memory!')
    return df


@st.cache(persist=True,suppress_st_warning=True)
def process_data(data):
    data = data.replace(' ', np.nan)
    data["TotalCharges"] = data["TotalCharges"].astype(float)
    # just yes and no
    categories_no = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']
    for el in categories_no : 
        data[el]  = data[el].replace({'No internet service' : 'No'})
    data["SeniorCitizen"] = data["SeniorCitizen"].replace({1:"Yes",0:"No"})
    # separating categorical and numerical features
    threshold_numerical = 40
    cat_features = data.nunique()[data.nunique() < threshold_numerical].keys().tolist()
    cat_features = cat_features[:-1]
    bin_features   = data.nunique()[data.nunique() == 2].keys().tolist()
    num_features   = [x for x in data.columns if x not in cat_features + ['Churn'] + ['customerID']]
    multi_features = [i for i in cat_features if i not in bin_features]

    # Feature scaling and encoding
    # label encoder for binary features
    label_enc = LabelEncoder()
    for feat in bin_features :
        data[feat] = label_enc.fit_transform(data[feat])
        
    # dummies for cat features
    data = pd.get_dummies(data = data,columns = multi_features )

    # scaling numerical features
    std = StandardScaler()
    data_scaled = std.fit_transform(data[num_features])
    data_scaled = pd.DataFrame(data_scaled,columns=num_features)

    # merging scaled dataset
    data = data.drop(columns = num_features,axis = 1)
    data = data.merge(data_scaled,left_index=True,right_index=True,how = "left")

    # dropping nas
    data = data.dropna()

    process_data = data
    return process_data

@st.cache(persist=True,suppress_st_warning=True)
def data_split(data):
    #splitting train and test data 
    train,test = train_test_split(data,test_size = .25 ,random_state = 256)
    
    ##seperating dependent and independent variables
    Id_col     = ['customerID']
    target_col = ["Churn"]
    cols    = [i for i in data.columns if i not in Id_col + target_col]
    train_X = train[cols]
    train_Y = train[target_col]
    test_X  = test[cols]
    test_Y  = test[target_col]

    return train_X, train_Y, test_X, test_Y


@st.cache(persist=True,suppress_st_warning=True)
def oversampling(data):
    Id_col     = ['customerID']
    target_col = ["Churn"]
    cols    = [i for i in data.columns if i not in Id_col+target_col]

    over_X = data[cols]
    over_Y = data[target_col]

    #   Split train and test data
    over_train_X,over_test_X,over_train_Y,over_test_Y = train_test_split(over_X,over_Y,
                                                                         test_size = .1 ,
                                                                         random_state = 256)

    #oversampling minority class using smote
    over_sampler = SMOTE(random_state=256)
    over_sampled_X, over_sampled_Y = over_sampler.fit_sample(over_train_X,over_train_Y)
    over_sampled_X = pd.DataFrame(data = over_sampled_X,columns=cols)
    over_sampled_Y = pd.DataFrame(data = over_sampled_Y,columns=target_col)

    return over_sampled_X, over_sampled_Y


def churn_prediction(algorithm,training_x,testing_x,
                             training_y,testing_y,cols,cf,bar) :
    #model
    algorithm.fit(training_x,training_y)
    bar.progress(25)
    predictions   = algorithm.predict(testing_x)
    bar.progress(50)
    probabilities = algorithm.predict_proba(testing_x)
    bar.progress(75)
    #coeffs
    if   cf == "coefficients" :
        coefficients  = pd.DataFrame(algorithm.coef_.ravel())
    elif cf == "features" :
        coefficients  = pd.DataFrame(algorithm.feature_importances_)
        
    column_df = pd.DataFrame(cols)

    if cf in ['coefficients','features']:
        coef_sumry = (pd.merge(coefficients,column_df,left_index= True,
                                right_index= True, how = "left"))
        coef_sumry.columns = ["coefficients","features"]
        coef_sumry = coef_sumry.sort_values(by = "coefficients",ascending = False)
    
    accuracy = accuracy_score(testing_y,predictions)
    bar.progress(100)
    
    precision, recall, f_score, support = precision_recall_fscore_support(testing_y,
                                                                          predictions,
                                                                          labels=[0,1],
                                                                          )
    # a pandas way:
    results_pd = pd.DataFrame({"class": ['not churn','churn'],
                               "precision": precision,
                               "recall": recall,
                               "f_score": f_score,
                               "support": support
                               })
    '''
    ### Classification report
    '''

    st.write(pd.DataFrame.from_dict(results_pd))
    
    #confusion matrix
    conf_matrix = confusion_matrix(testing_y,predictions)
    #roc_auc_score
    model_roc_auc = roc_auc_score(testing_y,predictions) 
    '''
    Often, stakeholders are interested in a single metric that can quantify model performance. The Area Under the Curve or AUC is one metric you can use in these cases, and another is the F1 score, which is calculated as below:

    2 * (precision * recall) / (precision + recall)

    The advantage of the F1 score is it incorporates both precision and recall into a single metric, and a high F1 score is a sign of a well-performing model, even in situations where you might have imbalanced classes.
    '''
    st.write("Area under curve : ",model_roc_auc)
    st.write("F1 score : ",f_score)
    st.write("Precision : ",precision)
    st.write('Recall :', recall)
    st.write('Confusion matrix : ',conf_matrix)
    
    return predictions,probabilities,accuracy,model_roc_auc


'# Customer churn prediction demo'

'''
Churn rate is a business term describing the rate at which customers leave or cease paying for a product or service. 
It's a critical figure in many businesses, as it's often the case that acquiring new customers is a lot more costly than retaining existing ones (in some cases, 5 to 20 times more expensive).

Understanding what keeps customers engaged, therefore, is incredibly valuable, as it is a logical foundation from which to develop retention strategies and roll out operational practices aimed to keep customers from walking out the door. Consequently, there's growing interest among companies to develop better churn-detection techniques, leading many to look to data mining and machine learning for new and creative approaches.

'''

# read dataset
'''
The data set we are using is a famous telecom customer data set.
The goal is to predict behavior to retain customers. 
Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

The data set includes information about:

- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

Data Source: https://www.kaggle.com/blastchar/telco-customer-churn
'''

df = read_data()
st.write(df)
Id_col     = ['customerID']
target_col = ["Churn"]
cols    = [i for i in df.columns if i not in Id_col+target_col]

'''
## Processing and Balancing data

Most of the datasets are skewed and unbalanced. In the case of a churn prediction model, there is an imbalance between classes Churned and Not Churned. Usually the latter being a majority class.
Upsampling and downsampling can help.
It is also important to process data to remove outliers, scale and normalize the dataset in a format digestible for the ML algorithms.

'''

with st.spinner('Processing...'):
    processed_data = process_data(df)
    # Splitting data
    train_X, train_Y, test_X, test_Y = data_split(processed_data)
    # Oversampling
    over_sampled_X, over_sampled_Y = oversampling(processed_data)

# show processed data
st.write(processed_data)
processed = True
st.success('Data processed')

'''
## Select and run the ML model
You can choose between different models such as logistic regression, knn, random forests, LGBM, XGboost and AutoML
'''

option = st.selectbox(
    'Which model do you want to run?',['logistic regression', 'KNN', 'random forests', 'LGBM', 'XGboost', 'AutoML'])


'''
### Choose model parameters
'''
'Select model parameters for ', option

if option == 'logistic regression':
    max_iter = st.slider('max_iter',1,1000,100)
    penalty = st.selectbox('penalty',['l1','l2'])
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=max_iter, multi_class='ovr', n_jobs=1,
          penalty=penalty, random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
    cf = 'coefficients'
elif option == 'KNN':
    leaf_size = st.slider('leaf_size',1,100,30)
    n_neighbors = st.slider('number_neighbors',1,20,5)
    model = KNeighborsClassifier(algorithm='auto', leaf_size=leaf_size, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=n_neighbors, p=2,
           weights='uniform')
    cf = 'NA'
elif option == 'random forests':
    pass
elif option == 'LGBM':
    from lightgbm import LGBMClassifier
    learning_rate = st.slider('learning_rate',0.1,1.0,0.5,0.1)
    max_depth = st.slider('max_depth',1,10,7)
    n_estimators = st.slider('n_estimators',1,500,100,5)
    num_leaves = st.slider('num_leaves',1,1000,500)

    model = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                        learning_rate=learning_rate, max_depth=max_depth, min_child_samples=20,
                        min_child_weight=0.001, min_split_gain=0.0, n_estimators=n_estimators,
                        n_jobs=-1, num_leaves=num_leaves, objective='binary', random_state=None,
                        reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                        subsample_for_bin=200000, subsample_freq=0)
    cf = 'features'
elif option == 'XGboost':
    from xgboost import XGBClassifier
    learning_rate = st.slider('learning_rate',0.1,1.0,0.9,0.1)
    max_depth = st.slider('max_depth',1,10,7)
    n_estimators = st.slider('n_estimators',1,500,100,5)
    model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bytree=1, gamma=0, learning_rate=learning_rate, max_delta_step=0,
                    max_depth = max_depth, min_child_weight=1, missing=None, n_estimators=n_estimators,
                    n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                    silent=True, subsample=1)
    cf = 'features'
elif option == 'AutoML':
    from supervised.automl import AutoML
    '''
    All the time parameters are in seconds.
    '''
    total_time_limit = st.slider('total_time_limit',60,600,120)
    top_models_to_improve = st.slider('top_models_to_improve',1,10,4)
    learner_time_limit = st.slider('learner_time_limit',10,120,60)
    hill_climbing_steps = st.slider('hill_climbing_steps',1,10,3)
    model = AutoML(total_time_limit=total_time_limit,top_models_to_improve=top_models_to_improve,
                    learner_time_limit=learner_time_limit,algorithms=["Xgboost", "RF", "LightGBM"],
                    start_random_models=10, hill_climbing_steps=hill_climbing_steps)
    

'''
### Run the model
Now that everything is properly set-up you can run the model.
'''
# running models
if st.button('Run model'):
    if processed:
        with st.spinner(text='Training models...'):
            my_bar = st.progress(0)
            my_bar.progress(1)

            if option == 'AutoML':
                my_bar.progress(20)
                model.fit(over_sampled_X,over_sampled_Y)
                my_bar.progress(70)
                predictions = model.predict(test_X)
                probabilities = predictions['label'].values
            else:
                predictions,probabilities,accuracy,model_roc_auc = churn_prediction(model,over_sampled_X,test_X,
                             over_sampled_Y,test_Y,cols,cf,my_bar)
                probabilities = probabilities[:,1]
            my_bar.progress(100)
        st.success('Done!')

        
        '''
        ## Receiver Operating Curve

        The area under a ROC curve is also a good measure of model performances.

        '''
        fpr, tpr, _ = roc_curve(test_Y, probabilities)
        # # plot the roc curve for the model
        # roc_fig = pyplot.figure()
        # pyplot.plot(fpr, tpr, marker='.', label=option, color='red')
        # # axis labels
        # pyplot.xlabel('False Positive Rate')
        # pyplot.ylabel('True Positive Rate')
        # # show the legend
        # pyplot.legend()
        # # show the plot
        # st.pyplot(roc_fig)


        
        fig = go.Figure(data=go.Scatter(x=fpr, y=tpr,
                                        line = dict(color='royalblue', width=4, dash='dash')))
        fig.update_layout(title='ROC Curve',
                   xaxis_title='False Positive Rate',
                   yaxis_title='True Positive Rate')
        st.plotly_chart(fig)

        '''
        ## Precision Recall Curve

        Accuracy in general is not a good metric for unbalanced datasets, because the model will predict with high accuracy the the customer is not churning, but that doesn't help us in predicting if a customer will churn.

        Two important mentrics to consider are precision and recall.
        Precision is: TP / (TP + FP)
        Recall is:    TP / (TP + FN)

        - High precision means few false positives: not so many non churners falsely classified as churners.
        - High recall means few false negatives: not so many churners falsely classified as non churners. So correctly classifies churners.


        The choice depends on the business objective and costs:
        - If keeping existing customers which are potential churners is more expensive than the value of acquiring customers -> your model should have high precision
        - If acquiring customers is more expensive than an offer to keep existing customers -> you want your model to have high recall

        '''
        recall,precision,thresholds = precision_recall_curve(test_Y,probabilities)
        # # plot the roc curve for the model
        # pr_fig = pyplot.figure()
        # pyplot.plot(recall, precision, marker='.', label=option, color='red')
        # # axis labels
        # pyplot.xlabel('Recall')
        # pyplot.ylabel('Precision')
        # # show the legend
        # pyplot.legend()
        # # show the plot
        # st.pyplot(pr_fig)

        fig = go.Figure(data=go.Scatter(x=recall, y=precision,
                                        line = dict(color='royalblue', width=4, dash='dash')))
        fig.update_layout(title='Precision Recall Curve',
                   xaxis_title='Recall',
                   yaxis_title='Precision')
        st.plotly_chart(fig)
        st.balloons()
    elif not processed:
        st.warning('Please process the data first!')