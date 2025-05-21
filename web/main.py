from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import mysql.connector
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.optimize import leastsq
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import matplotlib
from neural_network import NeuralNetwork
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pickle
import os
import io
import base64
from flask_cors import CORS
import folium
from folium.plugins import MarkerCluster
import time
from sklearn.linear_model import LinearRegression
import json
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torch.optim as optim
import random

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

# RNN Model using PyTorch
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
    def train_model(self, X, y, epochs=100, lr=0.01):
        """Train the RNN model and return training history"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        history = {
            'loss': [],
            'epoch': []
        }
        
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            history['loss'].append(loss.item())
            history['epoch'].append(epoch)
            
        return history

# Function to connect to the MySQL database
def connect_db():
    return mysql.connector.connect(
        host="localhost",      # Ganti dengan host database Anda
        user="root",           # Ganti dengan username database Anda
        password="",           # Ganti dengan password database Anda
        database="data_banjir" # Ganti dengan nama database Anda
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

    # Predictions
    y_pred_train = model.forward(X_train)
    y_pred_test = model.forward(X_test)

    # Evaluate the model
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    # Assuming binary classification for simplicity (you can adapt this to your needs)
    train_accuracy = accuracy_score(np.round(y_train).astype(int), np.round(y_pred_train).astype(int))
    test_accuracy = accuracy_score(np.round(y_test).astype(int), np.round(y_pred_test).astype(int))

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
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
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

# Function to predict using the data from data_uji table
@app.route('/predict', methods=['GET'])
def predict_from_data_uji():
    load_model()  # Load the model and scalers

    # Get year and month from query parameters
    year = request.args.get('year')
    month = request.args.get('month')

    # Connect to the database
    db = connect_db()
    cursor = db.cursor()

    # Build the query to fetch data, with optional filtering by year and month
    query = "SELECT Wilayah, Bulan, Tahun, Curah_Hujan, Suhu, Tinggi_Muka_Air FROM data_uji"
    conditions = []
    
    if year:
        conditions.append(f"Tahun = {year}")
    if month:
        conditions.append(f"Bulan = '{month}'")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    cursor.execute(query)

    # Fetch data and load into a DataFrame
    columns = ['Wilayah', 'Bulan', 'Tahun', 'Curah_Hujan', 'Suhu', 'Tinggi_Muka_Air']
    data_uji = pd.DataFrame(cursor.fetchall(), columns=columns)

    # Close the database connection
    cursor.close()
    db.close()

    # Check if data is available
    if data_uji.empty:
        return jsonify([])  # Return an empty list if no data is found

    # Preprocess the data
    X_uji = data_uji[['Curah_Hujan', 'Suhu', 'Tinggi_Muka_Air']].values
    X_uji_scaled = scaler.transform(X_uji)

    # Perform predictions
    y_pred_scaled = model.forward(X_uji_scaled)
    y_pred = np.round(y_pred_scaled * max_label_value).astype(int)
    prediksi_labels = label_encoder.inverse_transform(y_pred.ravel())

    # Prepare the JSON response
    # mengirim hasil prediksi ke web
    predictions = []
    for i, row in data_uji.iterrows():
        wilayah = row['Wilayah']
        bulan = row['Bulan']
        tahun = row['Tahun']
        prediksi = prediksi_labels[i]
        latitude, longitude = coordinates.get(wilayah, [0, 0])
        
        predictions.append({
            'Wilayah': wilayah,
            'Bulan': bulan,
            'Tahun': tahun,
            'Prediksi': prediksi,
            'Koordinat': {
                'Latitude': latitude,
                'Longitude': longitude
            }
        })

    return jsonify(predictions)
# Route to return training data from the database
# Route to return training data from the database
@app.route('/data_train', methods=['GET'])
def get_data_train():
    # Connect to the database
    db = connect_db()
    cursor = db.cursor()

    # Query to fetch training data from 'data_banjir'
    query = "SELECT * FROM data_banjir ORDER BY Wilayah ASC,  Tahun ASC"
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

    print(data_test_list)

    # Close the database connection
    cursor.close()
    db.close()

    # Return the data in JSON format
    if data_test_list:
        return jsonify(data_test_list), 200
    else:
        return jsonify({'message': 'No testing data found'}), 404

# Endpoint untuk menambahkan data latih ke database
@app.route('/add_data_train', methods=['POST'])
def add_data_train():
    # Mengambil data JSON dari request
    data = request.get_json()

    # Mengecek apakah data yang dibutuhkan ada di request
    required_fields = ['Wilayah', 'Bulan', 'Tahun', 'Curah_Hujan', 'Suhu', 'Tinggi_Muka_Air', 'Potensi_Banjir']
    if not all(field in data for field in required_fields):
        return jsonify({'message': 'Data tidak lengkap!'}), 400

    # Menghubungkan ke database
    db = connect_db()
    cursor = db.cursor()

    # Query untuk memasukkan data ke tabel `data_banjir`
    query = """
        INSERT INTO data_banjir (Wilayah, Bulan, Tahun, Curah_Hujan, Suhu, Tinggi_Muka_Air, Potensi_Banjir)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        data['Wilayah'], 
        data['Bulan'], 
        data['Tahun'], 
        data['Curah_Hujan'], 
        data['Suhu'], 
        data['Tinggi_Muka_Air'], 
        data['Potensi_Banjir']
    )

    try:
        # Menjalankan query untuk menambahkan data ke database
        cursor.execute(query, values)
        db.commit()  # Commit perubahan ke database
        return jsonify({'message': 'Data berhasil ditambahkan ke database!'}), 201
    except mysql.connector.Error as err:
        # Menangani error jika terjadi masalah saat memasukkan data
        return jsonify({'message': f'Gagal menambahkan data: {err}'}), 500
    finally:
        # Menutup koneksi ke database
        cursor.close()
        db.close()
# Endpoint untuk menambahkan data uji ke database
@app.route('/add_data_test', methods=['POST'])
def add_data_test():
    # Mengambil data JSON dari request
    data = request.get_json()

    # Mengecek apakah data yang dibutuhkan ada di request
    required_fields = ['Wilayah', 'Bulan', 'Tahun', 'Curah_Hujan', 'Suhu', 'Tinggi_Muka_Air']
    if not all(field in data for field in required_fields):
        return jsonify({'message': 'Data tidak lengkap!'}), 400

    # Menghubungkan ke database
    db = connect_db()
    cursor = db.cursor()

    # Query untuk memasukkan data ke tabel `data_uji`
    query = """
        INSERT INTO data_uji (Wilayah, Bulan, Tahun, Curah_Hujan, Suhu, Tinggi_Muka_Air)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    values = (
        data['Wilayah'], 
        data['Bulan'], 
        data['Tahun'], 
        data['Curah_Hujan'], 
        data['Suhu'], 
        data['Tinggi_Muka_Air']
    )

    try:
        # Menjalankan query untuk menambahkan data ke database
        cursor.execute(query, values)
        db.commit()  # Commit perubahan ke database
        return jsonify({'message': 'Data uji berhasil ditambahkan ke database!'}), 201
    except mysql.connector.Error as err:
        # Menangani error jika terjadi masalah saat memasukkan data
        return jsonify({'message': f'Gagal menambahkan data uji: {err}'}), 500
    finally:
        # Menutup koneksi ke database
        cursor.close()
        db.close()

# Endpoint untuk menghapus data dari tabel data_banjir (data latih)
@app.route('/delete_data_train', methods=['GET'])
def delete_data_train():
    data_id = request.args.get('id')  # Mengambil ID dari parameter query
    
    if not data_id:
        return jsonify({'message': 'ID data yang akan dihapus tidak ditemukan!'}), 400
    
    # Menghubungkan ke database
    db = connect_db()
    cursor = db.cursor()

    try:
        # Query untuk menghapus data dari tabel data_banjir
        query = "DELETE FROM data_banjir WHERE id = %s"
        cursor.execute(query, (data_id,))
        db.commit()
        
        if cursor.rowcount > 0:
            return jsonify({'message': f'Data dengan ID {data_id} berhasil dihapus.'}), 200
        else:
            return jsonify({'message': 'Data tidak ditemukan!'}), 404
    except mysql.connector.Error as err:
        return jsonify({'message': f'Gagal menghapus data: {err}'}), 500
    finally:
        cursor.close()
        db.close()

# Endpoint untuk menghapus data dari tabel data_uji (data uji)
@app.route('/delete_data_test', methods=['GET'])
def delete_data_test():
    data_id = request.args.get('id')  # Mengambil ID dari parameter query
    
    if not data_id:
        return jsonify({'message': 'ID data yang akan dihapus tidak ditemukan!'}), 400
    
    db = connect_db()
    cursor = db.cursor()

    try:
        # Query untuk menghapus data dari tabel data_uji
        query = "DELETE FROM data_uji WHERE id = %s"
        cursor.execute(query, (data_id,))
        db.commit()
        
        if cursor.rowcount > 0:
            return jsonify({'message': f'Data dengan ID {data_id} berhasil dihapus.'}), 200
        else:
            return jsonify({'message': 'Data tidak ditemukan!'}), 404
    except mysql.connector.Error as err:
        return jsonify({'message': f'Gagal menghapus data: {err}'}), 500
    finally:
        cursor.close()
        db.close()


# Index route to display HTML page
@app.route('/')
def index():
    
    return render_template('index.html')

# Simulation route to display simulation page
@app.route('/simulation')
def simulation():
    # Get database statistics for display on simulation page
    db_stats = get_database_statistics()
    return render_template('simulation.html', db_stats=db_stats)

# Enhanced simulation route to display the improved simulation page
@app.route('/simulation-enhanced')
def simulation_enhanced():
    # Get database statistics for display on simulation page
    db_stats = get_database_statistics()
    return render_template('simulation_enhanced.html', db_stats=db_stats)

# Get database statistics for the simulation page
def get_database_statistics():
    db = connect_db()
    cursor = db.cursor()
    
    # Get counts from both tables
    cursor.execute("SELECT COUNT(*) FROM data_banjir")
    train_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM data_uji")
    test_count = cursor.fetchone()[0]
    
    # Get distribution of flood potential labels
    cursor.execute("SELECT Potensi_Banjir, COUNT(*) FROM data_banjir GROUP BY Potensi_Banjir")
    label_counts = dict(cursor.fetchall())
    
    # Get sample data from each table for preview
    cursor.execute("SELECT * FROM data_banjir ORDER BY id ASC LIMIT 5")
    train_samples = cursor.fetchall()
    
    cursor.execute("SELECT * FROM data_uji ORDER BY id ASC LIMIT 5")
    test_samples = cursor.fetchall()
    
    # Get region distribution
    cursor.execute("SELECT Wilayah, COUNT(*) FROM data_banjir GROUP BY Wilayah")
    region_counts = dict(cursor.fetchall())
    
    # Get average values for numerical features by region
    cursor.execute("""
        SELECT Wilayah, 
               AVG(Curah_Hujan) as avg_curah_hujan, 
               AVG(Suhu) as avg_suhu, 
               AVG(Tinggi_Muka_Air) as avg_tinggi_air
        FROM data_banjir 
        GROUP BY Wilayah
    """)
    region_averages = {row[0]: {'curah_hujan': row[1], 'suhu': row[2], 'tinggi_air': row[3]} 
                      for row in cursor.fetchall()}
    
    # Get distribution of Potensi_Banjir by region
    cursor.execute("""
        SELECT Wilayah, Potensi_Banjir, COUNT(*) as count
        FROM data_banjir
        GROUP BY Wilayah, Potensi_Banjir
        ORDER BY Wilayah, Potensi_Banjir
    """)
    region_flooding = {}
    for row in cursor.fetchall():
        wilayah, potensi, count = row
        if wilayah not in region_flooding:
            region_flooding[wilayah] = {}
        region_flooding[wilayah][potensi] = count
    
    # Get min, max, avg of features
    cursor.execute("""
        SELECT 
            MIN(Curah_Hujan) as min_hujan, MAX(Curah_Hujan) as max_hujan, AVG(Curah_Hujan) as avg_hujan,
            MIN(Suhu) as min_suhu, MAX(Suhu) as max_suhu, AVG(Suhu) as avg_suhu,
            MIN(Tinggi_Muka_Air) as min_air, MAX(Tinggi_Muka_Air) as max_air, AVG(Tinggi_Muka_Air) as avg_air
        FROM data_banjir
    """)
    feature_stats = cursor.fetchone()
    
    # Get yearly data distribution
    cursor.execute("SELECT Tahun, COUNT(*) FROM data_banjir GROUP BY Tahun ORDER BY Tahun")
    year_distribution = dict(cursor.fetchall())
    
    # Get monthly data distribution
    cursor.execute("SELECT Bulan, COUNT(*) FROM data_banjir GROUP BY Bulan")
    month_distribution = dict(cursor.fetchall())
    
    cursor.close()
    db.close()
    
    # Return compiled statistics
    return {
        'train_count': train_count,
        'test_count': test_count,
        'label_counts': label_counts,
        'train_samples': train_samples,
        'test_samples': test_samples,
        'region_counts': region_counts,
        'region_averages': region_averages,
        'region_flooding': region_flooding,
        'feature_stats': {
            'curah_hujan': {'min': feature_stats[0], 'max': feature_stats[1], 'avg': feature_stats[2]},
            'suhu': {'min': feature_stats[3], 'max': feature_stats[4], 'avg': feature_stats[5]},
            'tinggi_air': {'min': feature_stats[6], 'max': feature_stats[7], 'avg': feature_stats[8]}
        },
        'year_distribution': year_distribution,
        'month_distribution': month_distribution
    }

# Route for LM simulation with detailed steps and visualizations
@app.route('/simulate_lm', methods=['POST'])
def simulate_lm():
    # Get parameters from request
    params = request.get_json()
    hidden_size = int(params.get('hidden_size', 10))
    iterations = int(params.get('iterations', 500))
    learning_rate = float(params.get('learning_rate', 0.01))
    
    # Connect to db and get training data
    db = connect_db()
    cursor = db.cursor()
    
    # Get counts for data summary
    cursor.execute("SELECT COUNT(*) FROM data_banjir")
    train_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM data_uji")
    test_count = cursor.fetchone()[0]
    
    # Get training data
    query = "SELECT Wilayah, Bulan, Tahun, Curah_Hujan, Suhu, Tinggi_Muka_Air, Potensi_Banjir FROM data_banjir"
    cursor.execute(query)
    columns = ['Wilayah', 'Bulan', 'Tahun', 'Curah_Hujan', 'Suhu', 'Tinggi_Muka_Air', 'Potensi_Banjir']
    data = pd.DataFrame(cursor.fetchall(), columns=columns)
    
    # Get test data for later predictions
    query_test = "SELECT Wilayah, Bulan, Tahun, Curah_Hujan, Suhu, Tinggi_Muka_Air FROM data_uji"
    cursor.execute(query_test)
    columns_test = ['Wilayah', 'Bulan', 'Tahun', 'Curah_Hujan', 'Suhu', 'Tinggi_Muka_Air']
    test_data = pd.DataFrame(cursor.fetchall(), columns=columns_test)
    
    cursor.close()
    db.close()
    
    # Preprocess data
    le = LabelEncoder()
    data['Potensi_Banjir'] = le.fit_transform(data['Potensi_Banjir'])
    
    # Features and target
    X = data[['Curah_Hujan', 'Suhu', 'Tinggi_Muka_Air']].values
    y = data['Potensi_Banjir'].values
    max_val = np.max(y)
    y = y / max_val
    
    # Split data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Initialize the model
    model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=1)
    
    # Track training progress
    history = {
        'iterations': [],
        'train_loss': [],
        'test_loss': []
    }
    
    # Initialize weights
    initial_weights = model.flatten_weights()
    weights_history = [initial_weights.copy()]
    
    # Define error function for LM algorithm
    def error_function(weights, X, y):
        model.unflatten_weights(weights)
        y_hat = model.forward(X)
        return (y_hat - y).ravel()
    
    # Store initial predictions
    y_pred_train_initial = model.forward(X_train)
    y_pred_test_initial = model.forward(X_test)
    train_mse_initial = mean_squared_error(y_train, y_pred_train_initial)
    test_mse_initial = mean_squared_error(y_test, y_pred_test_initial)
    history['iterations'].append(0)
    history['train_loss'].append(train_mse_initial)
    history['test_loss'].append(test_mse_initial)
    
    # Initialize iteration progress
    step_size = max(1, iterations // 10)  # Track approximately 10 steps
    
    # Run Levenberg-Marquardt optimization
    optimal_weights, success = leastsq(error_function, initial_weights, args=(X_train, y_train.reshape(-1, 1)), maxfev=iterations)
    model.unflatten_weights(optimal_weights)
    weights_history.append(optimal_weights.copy())
    
    # Final predictions
    y_pred_train = model.forward(X_train)
    y_pred_test = model.forward(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_acc = accuracy_score(np.round(y_train * max_val).astype(int), 
                              np.round(y_pred_train * max_val).astype(int))
    test_acc = accuracy_score(np.round(y_test * max_val).astype(int), 
                             np.round(y_pred_test * max_val).astype(int))
    
    # For visualization (add final point)
    history['iterations'].append(iterations)
    history['train_loss'].append(train_mse)
    history['test_loss'].append(test_mse)
    
    # Generate t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Create a figure for t-SNE
    plt.figure(figsize=(10, 8))
    colors = np.round(y * max_val).astype(int)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Potensi Banjir')
    plt.title('t-SNE Visualization of Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Save the figure to a bytes object
    tsne_img = io.BytesIO()
    plt.savefig(tsne_img, format='png')
    plt.close()
    tsne_img.seek(0)
    tsne_plot = base64.b64encode(tsne_img.getvalue()).decode('utf-8')
    
    # Create a loss history plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['iterations'], history['train_loss'], 'b-', label='Training Loss')
    plt.plot(history['iterations'], history['test_loss'], 'r-', label='Testing Loss')
    plt.title('Loss during Training')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    
    # Save the figure to a bytes object
    loss_img = io.BytesIO()
    plt.savefig(loss_img, format='png')
    plt.close()
    loss_img.seek(0)
    loss_plot = base64.b64encode(loss_img.getvalue()).decode('utf-8')
    
    # Create confusion matrix
    y_pred_classes = np.round(y_pred_test * max_val).astype(int)
    y_true_classes = np.round(y_test * max_val).astype(int)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save the confusion matrix to a bytes object
    cm_img = io.BytesIO()
    plt.savefig(cm_img, format='png')
    plt.close()
    cm_img.seek(0)
    cm_plot = base64.b64encode(cm_img.getvalue()).decode('utf-8')
    
    # Create a folium map with prediction results
    map_center = [5.0621243, 97.3258354]
    m = folium.Map(location=map_center, zoom_start=10)
    
    # Add markers for test data with predictions
    marker_cluster = MarkerCluster().add_to(m)
    
    # Color mapping for flood potential
    color_mapping = {
        0: 'green',    # Aman
        1: 'blue',     # Waspada
        2: 'orange',   # Siaga
        3: 'red'       # Bahaya
    }
    
    # Label mapping for readability
    potential_mapping = {
        0: 'Aman',
        1: 'Waspada',
        2: 'Siaga',
        3: 'Bahaya'
    }
    
    # Add test data to map
    wilayah_list = data['Wilayah'].unique()
    for wilayah in wilayah_list:
        if wilayah in coordinates:
            lat, lng = coordinates[wilayah]
            
            # Get data for this wilayah
            wilayah_data = data[data['Wilayah'] == wilayah]
            if not wilayah_data.empty:
                # Take the latest data
                latest_data = wilayah_data.iloc[-1]
                features = scaler.transform([ [
                    latest_data['Curah_Hujan'],
                    latest_data['Suhu'], 
                    latest_data['Tinggi_Muka_Air']
                ] ])
                
                # Predict
                model.unflatten_weights(optimal_weights)
                pred = model.forward(features)
                pred_class = int(np.round(pred[0][0] * max_val))
                
                # Add marker
                folium.Marker(
                    location=[lat, lng],
                    popup=f"""
                    <b>{wilayah}</b><br>
                    Prediksi: {potential_mapping.get(pred_class, 'Unknown')}<br>
                    Curah Hujan: {latest_data['Curah_Hujan']}<br>
                    Suhu: {latest_data['Suhu']}<br>
                    Tinggi Air: {latest_data['Tinggi_Muka_Air']}
                    """,
                    icon=folium.Icon(color=color_mapping.get(pred_class, 'gray'))
                ).add_to(marker_cluster)
    
    # Save map to HTML string
    map_html = m._repr_html_()
      # Create feature distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot curah hujan distribution by potential
    for i in range(int(max_val) + 1):
        subset = data[data['Potensi_Banjir'] == i]
        if not subset.empty:
            axes[0].hist(subset['Curah_Hujan'], alpha=0.6, bins=10, 
                         label=f'Potensi {label_mapping.get(i, i)}')
    axes[0].set_title('Distribusi Curah Hujan')
    axes[0].set_xlabel('Curah Hujan (mm)')
    axes[0].set_ylabel('Frekuensi')
    axes[0].legend()
    
    # Plot suhu distribution by potential
    for i in range(int(max_val) + 1):
        subset = data[data['Potensi_Banjir'] == i]
        if not subset.empty:
            axes[1].hist(subset['Suhu'], alpha=0.6, bins=10, 
                         label=f'Potensi {label_mapping.get(i, i)}')
    axes[1].set_title('Distribusi Suhu')
    axes[1].set_xlabel('Suhu (°C)')
    axes[1].set_ylabel('Frekuensi')
    axes[1].legend()
    
    # Plot tinggi muka air distribution by potential
    for i in range(int(max_val) + 1):
        subset = data[data['Potensi_Banjir'] == i]
        if not subset.empty:
            axes[2].hist(subset['Tinggi_Muka_Air'], alpha=0.6, bins=10, 
                         label=f'Potensi {label_mapping.get(i, i)}')
    axes[2].set_title('Distribusi Tinggi Muka Air')
    axes[2].set_xlabel('Tinggi Muka Air (cm)')
    axes[2].set_ylabel('Frekuensi')
    axes[2].legend()
    
    plt.tight_layout()
    
    # Save the distribution plot
    dist_img = io.BytesIO()
    plt.savefig(dist_img, format='png')
    plt.close()
    dist_img.seek(0)
    dist_plot = base64.b64encode(dist_img.getvalue()).decode('utf-8')
    
    # Return the results as JSON
    return jsonify({
        'metrics': {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_acc': train_acc,
            'test_acc': test_acc
        },
        'history': history,
        'data_stats': {
            'train_count': train_count,
            'test_count': test_count
        },
        'visualizations': {
            'tsne_plot': tsne_plot,
            'loss_plot': loss_plot,
            'confusion_matrix': cm_plot,
            'distribution_plot': dist_plot,
            'map_html': map_html
        }
    })

# Route for RNN simulation
@app.route('/simulate_rnn', methods=['POST'])
def simulate_rnn():
    # Get parameters from request
    params = request.get_json()
    hidden_size = int(params.get('hidden_size', 32))
    epochs = int(params.get('epochs', 100))
    learning_rate = float(params.get('learning_rate', 0.01))
    batch_size = int(params.get('batch_size', 16))
    
    # Connect to db and get training data
    db = connect_db()
    cursor = db.cursor()
    
    # Get counts for data summary
    cursor.execute("SELECT COUNT(*) FROM data_banjir")
    train_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM data_uji")
    test_count = cursor.fetchone()[0]
    
    # Get training data
    query = "SELECT Wilayah, Bulan, Tahun, Curah_Hujan, Suhu, Tinggi_Muka_Air, Potensi_Banjir FROM data_banjir"
    cursor.execute(query)
    columns = ['Wilayah', 'Bulan', 'Tahun', 'Curah_Hujan', 'Suhu', 'Tinggi_Muka_Air', 'Potensi_Banjir']
    data = pd.DataFrame(cursor.fetchall(), columns=columns)
    
    # Get test data for later predictions
    query_test = "SELECT Wilayah, Bulan, Tahun, Curah_Hujan, Suhu, Tinggi_Muka_Air FROM data_uji"
    cursor.execute(query_test)
    columns_test = ['Wilayah', 'Bulan', 'Tahun', 'Curah_Hujan', 'Suhu', 'Tinggi_Muka_Air']
    test_data = pd.DataFrame(cursor.fetchall(), columns=columns_test)
    
    cursor.close()
    db.close()
    
    # Preprocess data
    le = LabelEncoder()
    data['Potensi_Banjir'] = le.fit_transform(data['Potensi_Banjir'])
    
    # Features and target
    X = data[['Curah_Hujan', 'Suhu', 'Tinggi_Muka_Air']].values
    y = data['Potensi_Banjir'].values
    max_val = np.max(y)
    y = y / max_val
    
    # Split data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add sequence dimension
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Initialize the model
    input_size = X_train.shape[1]
    output_size = 1
    model = RNNModel(input_size, hidden_size, output_size)
    
    # Training the model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track training progress
    history = {
        'epochs': [],
        'train_loss': [],
        'test_loss': []
    }
    
    # Log initial performance
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        test_outputs = model(X_test_tensor)
        train_loss = criterion(train_outputs, y_train_tensor).item()
        test_loss = criterion(test_outputs, y_test_tensor).item()
    
    history['epochs'].append(0)
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    
    # Training loop
    step_size = max(1, epochs // 10)  # Track approximately 10 steps
    epoch_data = []
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Log every step_size epochs
        if epoch % step_size == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train_tensor)
                test_outputs = model(X_test_tensor)
                train_loss = criterion(train_outputs, y_train_tensor).item()
                test_loss = criterion(test_outputs, y_test_tensor).item()
                
                # Convert outputs to numpy for metrics calculation
                train_preds = train_outputs.numpy().flatten()
                test_preds = test_outputs.numpy().flatten()
                
                # Calculate metrics
                train_acc = accuracy_score(np.round(y_train * max_val).astype(int), 
                                          np.round(train_preds * max_val).astype(int))
                test_acc = accuracy_score(np.round(y_test * max_val).astype(int), 
                                         np.round(test_preds * max_val).astype(int))
                
                # Store epoch data for table
                epoch_data.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc
                })
            
            history['epochs'].append(epoch)
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        test_outputs = model(X_test_tensor)
        
        # Convert to numpy for metrics
        y_pred_train = train_outputs.numpy().flatten()
        y_pred_test = test_outputs.numpy().flatten()
        
        # Calculate final metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_acc = accuracy_score(np.round(y_train * max_val).astype(int), 
                                  np.round(y_pred_train * max_val).astype(int))
        test_acc = accuracy_score(np.round(y_test * max_val).astype(int), 
                                 np.round(y_pred_test * max_val).astype(int))
    
    # Generate t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Create a figure for t-SNE
    plt.figure(figsize=(10, 8))
    colors = np.round(y * max_val).astype(int)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Potensi Banjir')
    plt.title('t-SNE Visualization of Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Save the figure to a bytes object
    tsne_img = io.BytesIO()
    plt.savefig(tsne_img, format='png')
    plt.close()
    tsne_img.seek(0)
    tsne_plot = base64.b64encode(tsne_img.getvalue()).decode('utf-8')
    
    # Create a loss history plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['epochs'], history['train_loss'], 'b-', label='Training Loss')
    plt.plot(history['epochs'], history['test_loss'], 'r-', label='Testing Loss')
    plt.title('Loss during RNN Training')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    
    # Save the figure to a bytes object
    loss_img = io.BytesIO()
    plt.savefig(loss_img, format='png')
    plt.close()
    loss_img.seek(0)
    loss_plot = base64.b64encode(loss_img.getvalue()).decode('utf-8')
    
    # Create confusion matrix
    y_pred_classes = np.round(y_pred_test * max_val).astype(int)
    y_true_classes = np.round(y_test * max_val).astype(int)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save the confusion matrix to a bytes object
    cm_img = io.BytesIO()
    plt.savefig(cm_img, format='png')
    plt.close()
    cm_img.seek(0)
    cm_plot = base64.b64encode(cm_img.getvalue()).decode('utf-8')
    
    # Create a folium map with prediction results
    map_center = [5.0621243, 97.3258354]
    m = folium.Map(location=map_center, zoom_start=10)
    
    # Add markers for test data with predictions
    marker_cluster = MarkerCluster().add_to(m)
    
    # Color mapping for flood potential
    color_mapping = {
        0: 'green',    # Aman
        1: 'blue',     # Waspada
        2: 'orange',   # Siaga
        3: 'red'       # Bahaya
    }
    
    # Label mapping for readability
    potential_mapping = {
        0: 'Aman',
        1: 'Waspada',
        2: 'Siaga',
        3: 'Bahaya'
    }
    
    # Add test data to map
    wilayah_list = data['Wilayah'].unique()
    for wilayah in wilayah_list:
        if wilayah in coordinates:
            lat, lng = coordinates[wilayah]
            
            # Get data for this wilayah
            wilayah_data = data[data['Wilayah'] == wilayah]
            if not wilayah_data.empty:
                # Take the latest data
                latest_data = wilayah_data.iloc[-1]
                features = scaler.transform([ [
                    latest_data['Curah_Hujan'],
                    latest_data['Suhu'], 
                    latest_data['Tinggi_Muka_Air']
                ] ])
                
                # Predict with RNN
                features_tensor = torch.FloatTensor(features).unsqueeze(1)  # Add sequence dimension
                with torch.no_grad():
                    pred = model(features_tensor).numpy().flatten()[0]
                pred_class = int(np.round(pred * max_val))
                
                # Add marker
                folium.Marker(
                    location=[lat, lng],
                    popup=f"""
                    <b>{wilayah}</b><br>
                    Prediksi: {potential_mapping.get(pred_class, 'Unknown')}<br>
                    Curah Hujan: {latest_data['Curah_Hujan']}<br>
                    Suhu: {latest_data['Suhu']}<br>
                    Tinggi Air: {latest_data['Tinggi_Muka_Air']}
                    """,
                    icon=folium.Icon(color=color_mapping.get(pred_class, 'gray'))
                ).add_to(marker_cluster)
    
    # Save map to HTML string
    map_html = m._repr_html_()
      # Create feature distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot curah hujan distribution by potential
    for i in range(int(max_val) + 1):
        subset = data[data['Potensi_Banjir'] == i]
        if not subset.empty:
            axes[0].hist(subset['Curah_Hujan'], alpha=0.6, bins=10, 
                         label=f'Potensi {label_mapping.get(i, i)}')
    axes[0].set_title('Distribusi Curah Hujan')
    axes[0].set_xlabel('Curah Hujan (mm)')
    axes[0].set_ylabel('Frekuensi')
    axes[0].legend()
    
    # Plot suhu distribution by potential
    for i in range(int(max_val) + 1):
        subset = data[data['Potensi_Banjir'] == i]
        if not subset.empty:
            axes[1].hist(subset['Suhu'], alpha=0.6, bins=10, 
                         label=f'Potensi {label_mapping.get(i, i)}')
    axes[1].set_title('Distribusi Suhu')
    axes[1].set_xlabel('Suhu (°C)')
    axes[1].set_ylabel('Frekuensi')
    axes[1].legend()
    
    # Plot tinggi muka air distribution by potential
    for i in range(int(max_val) + 1):
        subset = data[data['Potensi_Banjir'] == i]
        if not subset.empty:
            axes[2].hist(subset['Tinggi_Muka_Air'], alpha=0.6, bins=10, 
                         label=f'Potensi {label_mapping.get(i, i)}')
    axes[2].set_title('Distribusi Tinggi Muka Air')
    axes[2].set_xlabel('Tinggi Muka Air (cm)')
    axes[2].set_ylabel('Frekuensi')
    axes[2].legend()
    
    plt.tight_layout()
    
    # Save the distribution plot
    dist_img = io.BytesIO()
    plt.savefig(dist_img, format='png')
    plt.close()
    dist_img.seek(0)
    dist_plot = base64.b64encode(dist_img.getvalue()).decode('utf-8')

    # Return the results as JSON
    return jsonify({
        'metrics': {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_acc': train_acc,
            'test_acc': test_acc
        },
        'history': history,
        'epoch_data': epoch_data,
        'data_stats': {
            'train_count': train_count,
            'test_count': test_count
        },
        'visualizations': {
            'tsne_plot': tsne_plot,
            'loss_plot': loss_plot,
            'confusion_matrix': cm_plot,
            'distribution_plot': dist_plot, 
            'map_html': map_html
        }
    })

if __name__ == '__main__':
    app.run(debug=True)

