# app.py

# from flask import Flask, request, jsonify
# import pickle

# app = Flask(__name__)

# # Load models
# with open('dv.bin', 'rb') as f:
#     dv = pickle.load(f)

# with open('model1.bin', 'rb') as f:
#     model = pickle.load(f)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get client data from POST request
#     client = request.get_json()
    
#     # Vectorize the client data
#     X_client = dv.transform([client])
    
#     # Predict the probability
#     probability = model.predict_proba(X_client)[0, 1]
    
#     # Return the probability as JSON
#     return jsonify({'probability': probability})

# # Run the Flask app
# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

# app.py

from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load models
with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)

with open('model1.bin', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X_client = dv.transform([client])
    probability = model.predict_proba(X_client)[0, 1]
    return jsonify({'probability': probability})

if __name__ == "__main__":
    app.run(debug=True, port=5000)

