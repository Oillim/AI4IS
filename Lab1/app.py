import joblib
import pandas as pd

model = joblib.load('spam_classifier.pkl')
cv = joblib.load('spam_vectorizer.pkl')
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
        text_cv = cv.transform(text.Text)
        if model.predict(text_cv)[0] == 1:
            print('\nThis is Spam email.')
        else:
            print('\nThis is not Spam email.')
    elif choice == '2':
        file_path = input('Enter file path: ')
        df = pd.read_csv(file_path)
        df.drop(['Spam/Ham', 'split'], axis=1, inplace=True)
        df = df[~(df['Subject'].isnull() & df['Message'].isnull())]
        df.fillna('', inplace=True)
        df['Text'] = df['Subject'] + " " + df['Message']
        text_cv = cv.transform(df['Text'])
        pred = model.predict(text_cv)
        for i in range(len(pred)):
            if pred[i] == 1:
                print(f'{df.iloc[i, 0]} is Spam email.')
            else:
                print(f'{df.iloc[i, 0]} is not Spam email.')
    elif choice == '3':
        print('Exiting...')
    else:
        print('Invalid choice.')
