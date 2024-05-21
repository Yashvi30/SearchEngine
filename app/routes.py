from flask import render_template, request
from app import app
from app.scorer import Scorer

# Define your documents
docs = [
    'About us. We deliver Artificial Intelligence & Machine Learning solutions to solve business challenges.',
    'Contact information. Email [martin davtyan at filament dot ai] if you have any questions',
    'Filament Chat. A framework for building and maintaining a scalable chatbot capability',
]

# Initialize the Scorer
scorer = Scorer(docs)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    results = scorer.score(query)
    relevant_docs = [docs[idx] for idx in results.argsort()[::-1]]
    return render_template('index.html', query=query, results=relevant_docs)
