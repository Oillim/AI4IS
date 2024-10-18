import joblib
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os

pipeline = joblib.load('spam_pipeline.pkl')

if __name__ == '__main__':
    print('Select function:')
    print('1. Check email immediately.')
    print('2. Check email csv file.')
    print('3. Exit.')
    choice = input('Enter choice: ')
    if choice == '1':
        subject = input('Enter email subject: ')
        message = input('Enter email message: ')
        text = pd.DataFrame(columns=['Text'], data=[subject + " " + message])
        if pipeline.predict(text)[0] == 1:
            print('\nThis is Spam email.')
        else:
            print('\nThis is not Spam email.')
    elif choice == '2':
        #Load data
        file_path = input('Enter file path: ')
        if not file_path.endswith('.csv') and not os.path.exists(file_path):
            print('Invalid file.')
            exit()

        #Preprocess data
        df = pd.read_csv(file_path)
        df = df[['Message ID','Subject', 'Message', 'Spam/Ham']]
        df = df[~(df['Subject'].isnull() & df['Message'].isnull())]
        df.fillna('', inplace=True)
        df['Text'] = df['Subject'] + " " + df['Message']
        df.drop(['Subject', 'Message'], axis=1, inplace=True)

        y = df['Spam/Ham'].map({'spam': 1, 'ham': 0})

        #Predict
        y_pred = pd.Series(pipeline.predict(df.Text))
        # for i in range(len(pred)):
        #     if pred[i] == 1:
        #         print(f'{df.iloc[i, 0]} is Spam email.')
        #     else:
        #         print(f'{df.iloc[i, 0]} is not Spam email.')
        
        #Save to output.csv
        df['Spam/Ham Predict'] = y_pred.map({1: 'spam', 0: 'ham'})
        df = df[['Message ID', 'Spam/Ham', 'Spam/Ham Predict']]
        print('Accuracy:', accuracy_score(y, y_pred))
        print(classification_report(y, y_pred, digits=4))
        df.to_csv('output.csv', index=False)
        print('Output saved to output.csv')
    elif choice == '3':
        print('Exiting...')
    else:
        print('Invalid choice.')
