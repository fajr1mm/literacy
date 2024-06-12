from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from database import fetch_selected_data_raw_sql, save_model_version, merge_data_training, update_data_training, get_latest_version_table_info
from train_model import train_model, TrainItem
from modelpredict import InputItem, predictmodel
from preprocessing_function import clean_text
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import random
import io
import chardet
from sqlalchemy.exc import IntegrityError
import pandas as pd
from head import app, db


with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

## PREDICT
@app.route('/predict', methods=['POST'])
def predict():
    try:
       # Ambil data dari permintaan POST
        data = request.get_json(force=True)
        input_batch = data.get('batch', [])
        save_dir = data.get('save_dir', None)

        # print(input_batch)

       # Ubah struktur data sesuai dengan skema InputItem
        input_items = [
            InputItem(id=item['id'], soal=item['Soal'], jawaban=item['Jawaban'])
            for item in input_batch
        ]

        input_predict = input_items

        response = predictmodel(save_dir, input_predict)
        predictions = response
        print(predictions)

        return predictions

    except Exception as e:
        return jsonify({'error': str(e)})
#------------------------------------------------------------------------------------------------------------#
## TRAIN 
# @app.route('/train', methods=['POST'])
# def train():
#     try:
#         data = request.get_json(force=True)
#         columns = data.get('columns', [])
#         parameters = data.get('parameters', {})
#         version = data.get('version', None)

#         if version:
#             # Fetch data based on the selected version
#             print("cek")
#             selected_data = fetch_selected_data(version)
#             train_item_data = [{col: getattr(row, col) for col in columns} for row in selected_data]
#             train_item = TrainItem(columns=columns, parameters={}, data=train_item_data)
#             result = train_model(train_item, columns=columns, parameters=parameters)
#         else:
#             latest_version_table_info = get_latest_version_table_info()
#             latest_version_table_name = latest_version_table_info['table_name']

#             # Fetch data from the latest version table
#             selected_data = fetch_selected_data(latest_version_table_name)
#             train_item_data = [{col: getattr(row, col) for col in columns} for row in selected_data]
#             train_item = TrainItem(columns=columns, parameters={}, data=train_item_data)
#             result = train_model(train_item, columns=columns, parameters=parameters)


#         save_model_version(result['model_path'])

#         return result

#     except Exception as e:
#         return jsonify({"error": str(e)}), 505
    
@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json(force=True)
        columns = data.get('columns', [])
        parameters = data.get('parameters', {})
        version = data.get('version', None)

        if version:
            # Fetch data based on the selected version
            selected_data = fetch_selected_data_raw_sql(version, columns)
            # print(selected_data)
            # train_item_data = [{col: getattr(row, col) for col in columns} for row in selected_data]
            train_item = TrainItem(columns=columns, parameters={}, data=selected_data)
            result = train_model(selected_data, columns=columns, parameters=parameters)

        else:
            latest_version_table_info = get_latest_version_table_info()
            latest_version_table_name = latest_version_table_info['table_name']

            # Fetch data from the latest version table
            selected_data = fetch_selected_data_raw_sql(latest_version_table_name)
            train_item_data = [{col: getattr(row, col) for col in columns} for row in selected_data]
            train_item = TrainItem(columns=columns, parameters={}, data=train_item_data)
            result = train_model(train_item, columns=columns, parameters=parameters)

        save_model_version(result['model_path'])

        return result

    except Exception as e:
        return jsonify({"error": str(e)}), 505
#------------------------------------------------------------------------------------------------------------#
## UPDATE DATA TRAINING DATABASE and CSV asdasdassadasfa
@app.route('/update-data-training-csv', methods=['POST'])
def update_data_training_csv():
    try:        
        if 'file' not in request.files:
            return jsonify({"error": "No CSV file provided"}), 400
        
        file = request.files['file']
        file_bytes = file.read()

        result = chardet.detect(file_bytes)
        encoding = result['encoding']

        df = pd.read_csv(io.StringIO(file_bytes.decode(encoding)))
        print(df)

        if df.empty:
            return jsonify({"error": "CSV file is empty"}), 400

        # Pilihan merge atau update dari GUI
        action = request.form.get('action', '')

        # Tanggal saat ini untuk versi tabel baru
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%Y%m%d")
        current_time = current_datetime.strftime("%H%M")
        random_digits = str(random.randint(10, 99))
        table_name = f"dt_v{current_date}{current_time}{random_digits}"

        # Cek aksi yang dipilih
        if action == 'merge':
            return merge_data_training(df, table_name)
        elif action == 'update':
            return update_data_training(df, table_name, 0)
        else:
            return jsonify({"error": "Invalid action"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 501
#------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    app.run(debug=True)
