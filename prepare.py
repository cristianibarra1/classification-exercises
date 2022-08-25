
import numpy as np
import pandas as pd
import os
import acquire
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def prep_iris(iris):
    iris = iris.drop(columns=['species_id','measurement_id'])
    iris = iris.rename(columns={'species_name':'species'})
    dummy_iris = pd.get_dummies(iris.species, drop_first=True)
    iris = pd.concat([iris, dummy_iris], axis=1)
    return iris


def prep_titanic(titanic):
    titanic = titanic.drop(columns=['embarked','class', 'age','deck'])
    dummy_df = pd.get_dummies(data=titanic[['sex','embark_town']], drop_first=True)
    titanic = pd.concat([titanic, dummy_df], axis=1)
    
    return titanic


def prep_telco(telco):
    '''
    This function accepts the telco df and preps it.
    '''
    # fix monthly charges from string to floats
    telco.total_charges = telco.total_charges.str.replace(' ','0')
    telco.total_charges = telco.total_charges.astype('float')
    
    # drop foreign key columns
    telco = telco.drop(columns=['internet_service_type_id', 'payment_type_id', 'contract_type_id'])
    
    # split out category columns
    dum = pd.get_dummies(telco.internet_service_type)
    telco = pd.concat([telco, dum[['DSL', 'Fiber optic']]], axis=1)
    dum = pd.get_dummies(telco.payment_type)
    telco = pd.concat([telco, dum], axis=1)
    telco = telco.drop(columns=(['payment_type', 'internet_service_type']))
    
    # relabel final columns with 0 or 1 values
    telco.contract_type = telco.contract_type.str.replace('Month-to-month', '12').str.replace('One year', '1').str.replace('Two year', '2').astype('int')
    telco.gender = telco.gender.str.replace('Female', '0').str.replace('Male', '1').astype('int')
    telco.partner = telco.partner.str.replace('No', '0').str.replace('Yes', '1').astype('int')
    telco.dependents = telco.dependents.str.replace('No', '0').str.replace('Yes', '1').astype('int')
    telco.phone_service = telco.phone_service.str.replace('No', '0').str.replace('Yes', '1').astype('int')
    telco.paperless_billing = telco.paperless_billing.str.replace('No', '0').str.replace('Yes', '1').astype('int')
    telco.churn = telco.churn.str.replace('No', '0').str.replace('Yes', '1').astype('int')
    telco.multiple_lines = telco.multiple_lines.str.replace('No phone service', '0').str.replace('Yes', '2').str.replace('No', '1').astype('int')
    telco.online_security = telco.online_security.str.replace('No internet service', '0').str.replace('Yes', '1').str.replace('No', '0').astype('int')
    telco.online_backup = telco.online_backup.str.replace('No internet service', '0').str.replace('Yes', '1').str.replace('No', '0').astype('int')
    telco.device_protection = telco.device_protection.str.replace('No internet service', '0').str.replace('Yes', '1').str.replace('No', '0').astype('int')
    telco.tech_support = telco.tech_support.str.replace('No internet service', '0').str.replace('Yes', '1').str.replace('No', '0').astype('int')
    telco.streaming_tv = telco.streaming_tv.str.replace('No internet service', '0').str.replace('Yes', '1').str.replace('No', '0').astype('int')
    telco.streaming_movies = telco.streaming_movies.str.replace('No internet service', '0').str.replace('Yes', '1').str.replace('No', '0').astype('int')
    
    return telco


def my_train_test_split(df, target):
    '''
    Takes in a dataframe and target (as a string). Returns train, validate, and test subset 
    dataframes with the .2/.8 and .25/.75 splits to create a final .2/.2/.6 split between datasets
    '''
    train, test = train_test_split(df, test_size = .2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train, test_size = .25, random_state=123, stratify=train[target])
    
    return train, validate, test


def float64_model(df):
    num_cols = df.columns[[df[col].dtype == 'float64' for col in df.columns]]
    for col in num_cols:
        plt.style.use('dark_background')
        plt.hist(df[col])
        plt.title(col)
        plt.show()
        return plt.show()

def model(df):
    num_cols = df.columns[[df[col].dtype == 'uint8' for col in df.columns]]
    for col in num_cols:
        plt.style.use('dark_background')
        plt.hist(df[col])
        plt.title(col)
        plt.show()
        return plt.show()