import pickle

# Load the models
with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)

with open('model1.bin', 'rb') as f:
    model = pickle.load(f)

# Client data
client = {"job": "retired", "duration": 445, "poutcome": "success"}

# Vectorize the client data
X_client = dv.transform([client])

# Predict the probability
probability = model.predict_proba(X_client)[0, 1]
print(probability)
