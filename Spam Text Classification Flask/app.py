from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import nltk as nlp
from nltk.corpus import stopwords
from statistics import mean 

import re

from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



app = Flask(__name__)

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df= pd.read_csv("Text Messages Dataset.csv", encoding="latin-1")
    
    new_Category = list()
    for i in df["Category"]:
            if i == "spam":
                new_Category.append(1)
            else:
                new_Category.append(0)
    
        
        
    #cleaning message text
    message_texts = []
    for message_text in df["Message"]:
            
            # removing all the punctuation marks from the list
            message_text = re.sub("[^a-zA-Z]"," ",message_text)
            
            # converting all the words in lower case
            message_text = message_text.lower() 
            
            message_texts.append(message_text)
 
    X,y=message_texts, new_Category
  
    
    
        
    porter = PorterStemmer()
    new_message_texts = []
        
    for message_text in message_texts:
            
            # Tokenize each message text
            message_text = message_text.split(" ")
            
            #Stemming each word of each message txt
            message_text = [ porter.stem(word) for word in message_text]
            
            #joining back each word
            message_text = " ".join(message_text)
            
            #appending each message txt to the new list
            new_message_texts.append(message_text)
        
    count_vectorizer = CountVectorizer(stop_words = "english")
    sparse_matrix = count_vectorizer.fit_transform(message_texts).toarray()
        
    #limiting Max feature to 50% of the total features
    max_features = len(sparse_matrix[0])//2 
    count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")
    sparse_matrix = count_vectorizer.fit_transform(message_texts).toarray()
        

    X_train, X_test, y_train, y_test = train_test_split(sparse_matrix, y, test_size=0.15, random_state=11)
    lr = LogisticRegression(max_iter = 10)
    lr.fit(X_train,y_train)
    print("Accuracy of LogisticRegressor model: {0:.2f}%".format(lr.score(X_test,y_test)*100))

	# Extract Feature With CountVectorizer
	
	#Alternative Usage of Saved Model
    
    joblib.dump(lr, 'NB_spam_model.pkl')
    NB_spam_model = open('NB_spam_model.pkl','rb')
    lr = joblib.load(NB_spam_model)

    if request.method == 'POST':
        message = request.form['message']
        #remove puctuations
        data = re.sub("[^a-zA-Z]"," ",message)
        # converting all the words in lower case
        message = message.lower() 
     
        #Tokenize the message
        data=message.split(" ")
        #print(data)
        #data=porter.stem(data)
        data = [ porter.stem(word) for word in data]
        
            
        #joining back each word
        data = " ".join(data)
        
            
       # removing extra whitespaces 
        data = re.sub(' +', ' ', data)
        
        sm = count_vectorizer.transform([data]).toarray()
        my_prediction = lr.predict(sm)
        print(my_prediction)
        probability = lr.predict_proba(sm) #predict probability

        print(my_prediction, " prob:(0,1) or (ham, spam) ", probability)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=False)
    