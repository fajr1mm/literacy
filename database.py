
from datetime import datetime
from io import StringIO
import pandas as pd
from sqlalchemy.exc import IntegrityError, NoSuchTableError
from flask import jsonify
from models import DataTraining, ModelVersi
from io import StringIO
import pandas as pd
from sqlalchemy import Column, Integer, VARCHAR, Table, inspect, text, create_engine
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.sql import text
from head import db

#database new
def get_latest_version_table_info():
    # get all table 
    all_tables = inspect(db.engine).get_table_names()

    # get table with relevant name
    relevant_tables = [table for table in all_tables if table.startswith("dt_v")]

    # get latest table version
    if relevant_tables:
        latest_version_table_name = max(relevant_tables)
        return get_table_info(latest_version_table_name)
    else:
        latest_version_table_name = None
        return latest_version_table_name

def get_table_info(table_name):
    if table_name:
        # Use inspector to get columns
        inspector = inspect(db.engine)
        columns = inspector.get_columns(table_name)

        # Exclude 'id' column from the list of columns
        column_names = [column["name"] for column in columns if column["name"] != "id"]

        return {"table_name": table_name, "columns": column_names}

    return None


def create_dynamic_columns(df, table_name):
    try:
        columns = df.columns

        # create new table with 'id' column
        dynamic_columns = [
            Column('id', Integer, primary_key=True, autoincrement=True),
            * [Column(col.strip(), VARCHAR(255)) if ' ' not in col else Column(col.strip(), VARCHAR(255), key=f"`{col.strip()}`") for col in columns]
        ]
        dynamic_table = Table(table_name, db.metadata, *dynamic_columns, extend_existing=True)

        # Create the table in the database
        with db.engine.begin() as connection:
            dynamic_table.create(bind=connection, checkfirst=True)

    except Exception as e:
        raise e


def merge_data_training(df, table_name):
    try:
        latest_version_table_info = get_latest_version_table_info()

        if not latest_version_table_info:
            return jsonify({"error": "Merging data training failed, data training is empty. You should perform an update first."}), 400

        # Check if the columns match
        if set(df.columns) != set(latest_version_table_info['columns']):
            return update_data_training(df, table_name, 1)

        # Create a new table with dynamic columns
        create_dynamic_columns(df, table_name)

        # Merge the data and insert into the new table
        merged_df = pd.concat([pd.read_sql_table(latest_version_table_info['table_name'], db.engine), df], ignore_index=True)
        merged_df.to_sql(table_name, db.engine, if_exists='replace', index=False)

        return jsonify({"message": "Data training merged successfully"}), 200

    except IntegrityError as e:
        db.session.rollback()
        return jsonify({"error": "Merging data training failed. Integrity error."}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 502

def update_data_training(df, table_name, code):
    try:
        # create table
        create_dynamic_columns(df, table_name)

        # Assign struktur tabel ke objek DataFrame
        df.to_sql(table_name, db.engine, if_exists='replace', index=False)

        if code == 0:
            return jsonify({"message": "Data training updated successfully"}), 200
        elif code == 1:
            return jsonify({"error": "Merging data training failed, data training is not synchronized. Your perform an update."}), 400

    except NoSuchTableError:
        return jsonify({"error": f"Table {table_name} does not exist"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 503
    
def fetch_selected_data(version):
    if version:
        # Fetch data based on the selected version
        selected_data = DataTraining.query.from_statement(
            text(f"SELECT * FROM {version}")
        ).all()
    else:
        # If version is not provided, fetch all data
        selected_data = DataTraining.query.all()

    return selected_data

def fetch_selected_data_raw_sql(version, col):

    engine = create_engine('mysql://root:@localhost/engine_literacy', echo=True)
    # Fetch data based on the selected version using raw SQL
    query_col = ','.join(col)
    query = text(f"SELECT {query_col} FROM {version}")
    with engine.connect() as connection:
        result = connection.execute(query)
        # Fetch all rows as tuples
        rows = result.fetchall()
        # Fetch column names
        columns = result.keys()
        # Convert each row to a dictionary
        selected_data = [dict(zip(columns, row)) for row in rows]

    return selected_data

    
def save_model_version(model_path):
    # Simpan versi model ke database
    new_model_version = ModelVersi(versi_model=f"ML_DT_{datetime.now().strftime('%d%m%Y')}", model_path=model_path)
    db.session.add(new_model_version)
    db.session.commit()

    return new_model_version.versi_model
