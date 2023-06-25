from flask import Flask,render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def ValuePredictor(texts):
    m=pickle.load(open('model.pkl','rb'))
    tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='latin-1', ngram_range=(1, 2))
    text_features = tfidf.fit_transform(texts)
    predictions = m.predict(text_features)

    return predictions

app=Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    return render_template('a.html')
@app.route('/result', methods = ['GET','POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        texts = list(to_predict_list.values())
        results = ValuePredictor(texts)

        if results ==0:
            pred='business'
        if results == 1:
            pred='entertainment'
        if results==2:
            pred='politics'
        if results==3:
            pred='sport'
        if results==4:
            pred='tech'
    return render_template('index.html',prediction = pred)

if __name__ == "__main__":
    app.run(host="localhost", port=8000,debug=True)