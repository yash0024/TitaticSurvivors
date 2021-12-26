import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from sklearn.linear_model import LogisticRegression

def featureEngineering(df):
    """ Apply Feature Engineering on the dataframe

    :param df: A pandas dataframe
    :param process: a string which takes on the values "train" or "test" 
    depending on the dataframe to which featureEngineering is applies to
    :return: the modified dataframe.
    """
    # Creating a SizeOfFamily column
    df['SizeOfFamily'] = df['SibSp'] + df['Parch'] + 1
    
    df['Fare'] = df['Fare'].fillna(0)
    df['Embarked'] = df['Embarked'].fillna('S')
    
    # A one hot representation of the embarked column
    embarked_one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_one_hot], axis=1)
    
    # Filling missing values in the Cabin column with 'Unknown'
    df['Cabin'] = df['Cabin'].fillna('Unknown')
     
    # Only the first letter of the cabin number seems to be important 
    df['Cabin'] = df['Cabin'].apply(lambda x: x[0])

    
    # replace unknown values with mode
    df['Cabin'] = np.where((df.Pclass == 1) & (df.Cabin =='U'), 'C',
                  np.where((df.Pclass == 2) & (df.Cabin == 'U'), 'D',
                  np.where((df.Pclass == 3) & (df.Cabin == 'U'), 'G',
                  np.where(df.Cabin == 'T', 'C', df.Cabin))))
    
    def get_title(x):
        return x.split(',')[1].split('.')[0].strip()

    # the names seem unimportant but the tile of each name has a high
    # correlation with survival on the training set
    df['Title'] = df['Name'].apply(get_title)
    
    df['Title'] = df['Title'].replace(['Lady', 'the Countess', 'Capt', 'Col','Don', 
                                                 'Dr', 'Major', 'Rev', 'Sir',
                                                 'Jonkheer', 'Dona'], 
                                      'Uncommon')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # A one hot representation of the Title column
    title_one_hot = pd.get_dummies(df['Title'], prefix='Title')
    df = pd.concat([df, title_one_hot], axis=1)
    
    age_mean = df.groupby('Title')['Age'].mean()

    # filling in the missing ages with the mean, grouped by title
    def fill_age(x):
        for index, age in zip(age_mean.index, age_mean.values):
            if x['Title'] == index:
                return age
    
    df['Age'] = df.apply(lambda x: fill_age(x) if np.isnan(x['Age']) else x['Age'], axis=1)
    
    # grouping age by category
    bins = [0, 12, 24, 45, 60, df.Age.max()]
    labels = ['Child', 'Young Adult', 'Adult','Older Adult','Senior']
    df["Age"] = pd.cut(df["Age"], bins, labels = labels)
    
    # A one hot representation of the new Cabin column
    cabin_one_hot = pd.get_dummies(df['Cabin'], prefix='Cabin')
    df = pd.concat([df, cabin_one_hot], axis=1)
    
    # A one hot representation of the Sex column
    sex_one_hot = pd.get_dummies(df['Sex'], prefix='Sex')
    df = pd.concat([df, sex_one_hot], axis=1)
    
    # A one hot representation of the Age column
    age_one_hot = pd.get_dummies(df['Age'], prefix='Age')
    df = pd.concat([df, age_one_hot], axis=1)
    
    df['Mother'] = np.where((df.Title=='Mrs') & (df.Parch > 0), 1, 0)
    df['Free'] = np.where(df['Fare']== 0, 1, 0)
    df = df.drop(['SibSp','Parch','Sex'], axis=1)

    # Dropping all columns containing categorical values
    df = df.drop(['PassengerId', 'Embarked', 'Name', 'Ticket', 'Cabin', 'Title', 'Age'], axis=1)
    
    # Normalizing
    df = (df-df.min())/(df.max()-df.min())
    
    # returning the dataframe
    return df
    
def readTrainData():
    """ Read the training data
    
    :return the data as a pandas dataframe
    """
    return pd.read_csv('train.csv')

def readTestData():
    """ Read the test data
    
    :return the data as a Pandas dataframe
    """
    
    return pd.read_csv('test.csv')
    

def RunAndSaveModelPredictionsLogisticRegression():
    """ Run the program and save the predicted values in a file named 
    submission.csv. Use Logistic Regression
    """
    
    trainData = readTrainData()
    testData = readTestData()
    passengerIds = testData['PassengerId']
    y_train = trainData['Survived'].values
    
    df1 = featureEngineering(trainData)
    df1 = df1.drop(['Survived'], axis = 1)
    df2 = featureEngineering(testData)
    
    X_train, X_test = df1.iloc[:,1:].values, df2.iloc[:,1:].values
    
    model = LogisticRegression()
    
    # training model
    model.fit(X_train, y_train)
    
    # finding the accuracy score on the training set
    print('Accuracy on the training set', model.score(X_train, y_train))
    
    predictions = model.predict(X_test)
    
    # the predictions (on the test set) are in the file submission.csv
    output = pd.DataFrame({'PassengerId': passengerIds, 'Survived': predictions})
    output.to_csv('submission.csv', index = False)
    

def RunAndSaveModelPredictionsXGBClassifier():
    """ Run the program and save the predicted values in a file named 
    submission.csv. Use the XGMoost Algorithm
    """
    
    trainData = readTrainData()
    testData = readTestData()
    passengerIds = testData['PassengerId']
    y_train = trainData['Survived'].values
    
    df1 = featureEngineering(trainData)
    df1 = df1.drop(['Survived'], axis = 1)
    df2 = featureEngineering(testData)
    
    X_train, X_test = df1.iloc[:,1:].values, df2.iloc[:,1:].values
    
    from xgboost import XGBClassifier
    model = XGBClassifier(learning_rate=0.0006,n_estimators=3000,
                                max_depth=4, min_child_weight=0,
                                gamma=0, subsample=0.7,
                                colsample_bytree=0.7,
                                scale_pos_weight=1, seed=27,
                                reg_alpha=0.00006)
    
    # training model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # finding the accuracy score on the training set
    print('Accuracy on the training set', model.score(X_train, y_train))
    
    # the predictions (on the test set) are in the file submission.csv
    output = pd.DataFrame({'PassengerId': passengerIds, 'Survived': predictions})
    output.to_csv('submission.csv', index = False)
    
RunAndSaveModelPredictionsLogisticRegression()
# RunAndSaveModelPredictionsXGBClassifier()