from flask import Flask,render_template,request,redirect
from textsimilarity import TextSimilarityEngine
app = Flask(__name__)


@app.route('/')
def homePage():
   return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def getSimilarity():
    cosine_doc1_doc2 = None
    try:
        doc1 = request.form['text1'].strip()
        doc2 = request.form['text2'].strip()
        docs = [doc1,doc2]
        textSimilarityObj = TextSimilarityEngine(docs)
        textSimilarityObj.generateFeatureVector()
        textSimilarityObj.generateDocumentTermFrequencies()
        textSimilarityObj.generateInverseDataFrequencyDictionary()
        textSimilarityObj.generateTFIDF()
        doc1_tfidf = list(textSimilarityObj._tfidf[0].values())
        doc2_tfidf = list(textSimilarityObj._tfidf[1].values())
        cosine_doc1_doc2 = str(TextSimilarityEngine.getSimilarity(doc1_tfidf,doc2_tfidf))        
    except Exception as e: 
        return render_template('index.html',errorMsg=e, text1 = doc1, text2=doc2)
    
    return render_template('index.html',score=cosine_doc1_doc2, text1 = doc1, text2=doc2)
if __name__ == '__main__':
    app.run()