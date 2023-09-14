from sqlalchemy import create_engine
from sqlalchemy import create_engine, Column, Integer, JSON, String
from sqlalchemy.sql.sqltypes import TIMESTAMP
from sqlalchemy.sql.expression import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import json
import pandas as pd
from datetime import datetime

from ..config import constants as Constantor

db_engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/{Constantor.DATABASE_NAME}')

# Create a base class for models
Base = declarative_base()

# Define model table attributes
class Model(Base):
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True)
    model_id = Column(String)
    model_algorithm = Column(String)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))
    parameters = Column(JSON)


def store_model(class_object:object):
    # Create tables in the database (if they don't exist)
    Base.metadata.create_all(db_engine)

    # Create a session
    Session = sessionmaker(bind=db_engine)
    session = Session()

    model_parameters = {}
    # Iterate through the input dictionary
    for key, value in class_object.__dict__.items():
        # Check if the value is not a DataFrame
        if not isinstance(value, pd.DataFrame) and not isinstance(value, list) and key is not 'logger':
            # Add the key-value pair to the result dictionary
            model_parameters[key] = value
    
    model_instance = Model(
        model_id=class_object.model_id, 
        model_algorithm = class_object.model_algorithm,
        parameters=model_parameters)
    session.add(model_instance)
    session.commit()
    session.close()

    print("Successfully store model in postgres!")
