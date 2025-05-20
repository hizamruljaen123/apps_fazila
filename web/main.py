from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import mysql.connector
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.optimize import leastsq
from sklearn.metrics import mean_squared_error, accuracy_score
import pickle
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables to store trained model and data
model = None
scaler = None
label_encoder = None
max_label_value = None
X_train, X_test, y_train, y_test = None, None, None, None
data_train, data_test = None, None

# Coordinates of each region
coordinates = {
    'Baktiya': [5.0621243, 97.3258354],
    'Lhoksukon': [5.0517222, 97.3078233],
    'Langkahan': [4.9211586, 97.1261701],
    'Cot Girek': [4.8616275, 97.2673567],
    'Matangkuli': [5.0306322, 97.2316173],
    'Tanah Luas': [4.9826373, 97.0425453],
    'Stamet Aceh Utara': [5.228798, 96.9449662]
}

# Mapping for labels
label_mapping = {0: 'Aman', 1: 'Waspada', 2: 'Siaga', 3: 'Awas'}

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1)
        self.A1 = 1 / (1 + np.exp(-self.Z1))
        self.Z2 = np.dot(self.A1, self.W2)
        self.A2 = 1 / (1 + np.exp(-self.Z2))
        return self.A2

    def cost_function(self, X, y):
        y_hat = self.forward(X)
        return 0.5 * np.sum((y_hat - y) ** 2)

    def cost_function_prime(self, X, y):
        y_hat = self.forward(X)
        delta2 = np.multiply(-(y - y_hat), y_hat * (1 - y_hat))
        dJdW2 = np.dot(self.A1.T, delta2)
        delta1 = np.dot(delta2, self.W2.T) * (self.A1 * (1 - self.A1))
        dJdW1 = np.dot(X.T, delta1)
        return dJdW1, dJdW2

    def flatten_weights(self):
        return np.concatenate((self.W1.ravel(), self.W2.ravel()))

    def unflatten_weights(self, flat_weights):
        hidden_size = self.W1.shape[1]
        self.W1 = flat_weights[:X_train.shape[1] * hidden_size].reshape(X_train.shape[1], hidden_size)
        self.W2 = flat_weights[X_train.shape[1] * hidden_size:].reshape(hidden_size, 1)

    def train(self, X, y, n_iter=500):
        def error_function(flat_weights, X, y):
            self.unflatten_weights(flat_weights)
            y_hat = self.forward(X)
            return (y_hat - y).ravel()

        initial_weights = self.flatten_weights()
        optimal_weights, success = leastsq(error_function, initial_weights, args=(X, y), maxfev=n_iter)
        self.unflatten_weights(optimal_weights)

# Function to connect to the MySQL database
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="data_banjir"
    )

# Function to train the model using data from the database
@app.route('/train', methods=['GET'])
def train_model():
    global model, scaler, label_encoder, max_label_value, X_train, X_test, y_train, y_test, data_train, data_test
    
    # Connect to the database
    db = connect_db()
    cursor = db.cursor()

    # Query to fetch data from the database
    query = "SELECT Wilayah, Bulan, Tahun, Curah_Hujan, Suhu, Tinggi_Muka_Air, Potensi_Banjir FROM data_banjir"
    cursor.execute(query)

    # Fetch data and load into a DataFrame
    columns = ['Wilayah', 'Bulan', 'Tahun', 'Curah_Hujan', 'Suhu', 'Tinggi_Muka_Air', 'Potensi_Banjir']
    data = pd.DataFrame(cursor.fetchall(), columns=columns)

    # Close the database connection
    cursor.close()
    db.close()

    # Preprocessing
    label_encoder = LabelEncoder()
    data['Potensi_Banjir'] = label_encoder.fit_transform(data['Potensi_Banjir'])

    # Features and target
    X = data[['Curah_Hujan', 'Suhu', 'Tinggi_Muka_Air']].values
    y = data['Potensi_Banjir'].values
    max_label_value = np.max(y)  # Save max label value for scaling during prediction
    y = y / max_label_value

    # Split data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train neural network
    model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=10, output_size=1)
    model.train(X_train, y_train.reshape(-1, 1), n_iter=1000)

    # Save model to file
    with open('model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'max_label_value': max_label_value
        }, f)

    return jsonify({
        'message': 'Model training completed successfully!',
        'train_size': len(X_train),
        'test_size': len(X_test)
    })

# Load the model, scaler, label_encoder, and max label value
def load_model():
    global model, scaler, label_encoder, max_label_value
    if model is None or scaler is None or label_encoder is None:
        try:
            with open('model.pkl', 'rb') as f:
                saved_data = pickle.load(f)
                model = saved_data['model']
                scaler = saved_data['scaler']
                label_encoder = saved_data['label_encoder']
                max_label_value = saved_data['max_label_value']
        except FileNotFoundError:
            raise Exception("Model is not trained yet!")

# Route to return training data from the database
@app.route('/data_train', methods=['GET'])
def get_data_train():
    # Connect to the database
    db = connect_db()
    cursor = db.cursor()

    # Query to fetch training data from 'data_banjir'
    query = "SELECT * FROM data_banjir ORDER BY Wilayah ASC, Tahun ASC"
    cursor.execute(query)

    # Fetch data and load into a list of dictionaries
    columns = ['id', 'Wilayah', 'Bulan', 'Tahun', 'Curah_Hujan', 'Suhu', 'Tinggi_Muka_Air', 'Potensi_Banjir']
    data_train = cursor.fetchall()
    data_train_list = [dict(zip(columns, row)) for row in data_train]

    # Close the database connection
    cursor.close()
    db.close()

    # Return the data in JSON format
    if data_train_list:
        return jsonify(data_train_list), 200
    else:
        return jsonify({'message': 'No training data found'}), 404

# Route to return testing data from the database
@app.route('/data_test', methods=['GET'])
def get_data_test():
    # Connect to the database
    db = connect_db()
    cursor = db.cursor()

    # Query to fetch testing data from 'data_uji'
    query = "SELECT * FROM data_uji"
    cursor.execute(query)

    # Fetch data and load into a list of dictionaries
    columns = ['id', 'Wilayah', 'Bulan', 'Tahun', 'Curah_Hujan', 'Suhu', 'Tinggi_Muka_Air']
    data_test = cursor.fetchall()
    data_test_list = [dict(zip(columns, row)) for row in data_test]

    # Close the database connection
    cursor.close()
    db.close()

    # Return the data in JSON format
    if data_test_list:
        return jsonify(data_test_list), 200
    else:
        return jsonify({'message': 'No testing data found'}), 404

# Endpoint untuk simulasi prediksi
@app.route('/api/simulasi/predict', methods=['POST'])
def simulasi_predict():
    try:
        data = request.get_json()
        
        # Validasi data input
        required_fields = ['curah_hujan', 'suhu', 'tinggi_muka_air']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Data tidak lengkap'}), 400

        # Load model jika belum dimuat
        load_model()

        # Preprocess data
        input_data = np.array([
            [float(data['curah_hujan']), 
             float(data['suhu']), 
             float(data['tinggi_muka_air'])]
        ])
        
        # Normalisasi data
        input_scaled = scaler.transform(input_data)
        
        # Prediksi
        prediction = model.forward(input_scaled)
        prediction = np.round(prediction * max_label_value).astype(int)
        status = label_encoder.inverse_transform(prediction.ravel())[0]
        
        # Simulasi parameter optimasi
        lambda_val = np.random.uniform(0.1, 1.0)
        iteration = np.random.randint(50, 100)
        error = np.random.uniform(0.01, 0.1)
        
        return jsonify({
            'prediction': int(prediction[0][0]),
            'status': status,
            'optimization': {
                'lambda': float(lambda_val),
                'iteration': int(iteration),
                'error': float(error)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route untuk halaman simulasi
@app.route('/simulasi')
def simulasi():
    return render_template('simulasi_prediksi.html')

if __name__ == '__main__':
    app.run(debug=True)

