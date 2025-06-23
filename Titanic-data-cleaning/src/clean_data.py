import pandas as pd
from sklearn.preprocessing import StandardScaler # type: ignore

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    
    Q1 = df['Age'].quantile(0.25)
    Q3 = df['Age'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['Age'] < (Q1 - 1.5 * IQR)) | (df['Age'] > (Q3 + 1.5 * IQR)))]
    
    return df

if __name__ == "__main__":
    df = load_data('../data/titanic.csv')
    cleaned_df = clean_data(df)
    cleaned_df.to_csv('../data/titanic_cleaned.csv', index=False)
    print("Data cleaned and saved successfully.")
