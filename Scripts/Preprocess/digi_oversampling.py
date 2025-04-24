# Razieh Askarizade @ 2025 
'''
refrence: https://github.com/zhpinkman/sentiment-analysis-digikala
'''

import pandas as pd
from os import path
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def oversampling(root_path, origin_data_path):
    '''
    input: root_path: str, origin_data_path: str
    output: df_train: pd.DataFrame, df_test: pd.DataFrame

    Read original data and convert it to binary classification.
    There are for class 1, 2, 3 which 3 is negative and 1 and 2 are positive; So we convert it to binary classification by replacing 2 with 1 and 3 with 0.
    Then we upsample the minority class to make the dataset balanced; Finally, we split the data into training and testing sets and save them to CSV files.

    Notice: Additionally, we add 1 to the labels to make them 1 and 2 instead of 0 and 1, becouse the model we use in the next steps reduces 1 form labels and labels for training model should not be negative.
    '''
    df = pd.read_csv(path.join(root_path, origin_data_path))

    df.drop(columns=['Score'], axis=1, inplace=True)

    df['Suggestion'] = df['Suggestion'].replace(2, 1)
    df['Suggestion'] = df['Suggestion'].replace(3, 0)

    df_minority = df[df['Suggestion']==0]
    df_majority = df[df['Suggestion']==1]

    df_minority_upsampled = resample(df_minority,
                                    replace=True,
                                    n_samples=df_majority.shape[0],    # to match majority class
                                    random_state=103)

    # Combine majority class with upsampled minority class
    df = pd.concat([df_majority, df_minority_upsampled])

    X = df['Text']
    y = df['Suggestion']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True,
                                                        stratify=y)

    # Create dataframes for training and testing sets
    df_train = pd.DataFrame({'Text': X_train, 'Suggestion': y_train + 1})
    df_test = pd.DataFrame({'Text': X_test, 'Suggestion': y_test + 1})

    return df_train, df_test