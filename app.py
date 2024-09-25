from flask import Flask, request, jsonify, render_template
import time
from model import analyze_statement, generate_counterspeech

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Renders the frontend HTML

@app.route('/analyze_statement', methods=['POST'])
def analyze_statement_route():
    data = request.json
    statement = data['statement']

    # Call the dummy analyze_statement function
    analysis = analyze_statement(statement, return_dummy_values=False)

    return jsonify(analysis)

@app.route('/generate_counterspeech', methods=['POST'])
def generate_counterspeech_route():
    data = request.json
    statement = data['statement']

    # Call the dummy generate_counterspeech function
    counterspeech, fact = generate_counterspeech(statement, return_dummy_values=False)

    return jsonify({'counterspeech': counterspeech, 'fact': fact})

if __name__ == '__main__':
    app.run(debug=True)
