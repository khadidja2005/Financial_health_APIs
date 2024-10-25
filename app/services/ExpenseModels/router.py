from fastapi import  HTTPException, APIRouter
from pydantic import BaseModel
from typing import List
import torch
import joblib
import os
import pandas as pd
import numpy as np
from datetime import datetime
from app.services.ExpenseModels.Model import ExpensePredictor

routerEX = APIRouter()

def load_model():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, '..', '..', 'saved_models')
        
        # Load scaler using joblib
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        scaler = joblib.load(scaler_path)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ExpensePredictor(
            input_size=5,
            hidden_size=64,
            num_layers=2,
            output_length=10
        ).to(device)
        
        # Load model weights with weights_only=True
        model_path = os.path.join(model_dir, "expense_predictor.pth")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, scaler, device
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

# Load model at startup
model, scaler, device = load_model()

class Expense(BaseModel):
    date: str
    amount: float

class ExpenseList(BaseModel):
    expenses: List[Expense]

class PredictionResponse(BaseModel):
    predicted_expenses: List[float]

def prepare_sequence_data(expenses: List[Expense], scaler):
    # Define feature names
    feature_names = ['hour', 'day', 'month', 'day_of_week', 'amount']
    
    # Convert expenses to DataFrame with feature names
    df = pd.DataFrame([{
        'date': exp.date,
        'amount': exp.amount
    } for exp in expenses])
    
    # Process the data
    df['date'] = pd.to_datetime(df['date'])
    
    # Create features DataFrame with correct column names
    features_df = pd.DataFrame({
        'hour': df['date'].dt.hour,
        'day': df['date'].dt.day,
        'month': df['date'].dt.month,
        'day_of_week': df['date'].dt.dayofweek,
        'amount': df['amount']
    })
    
    # Scale the data maintaining feature names
    scaled_data = scaler.transform(features_df)
    scaled_df = pd.DataFrame(scaled_data, columns=feature_names)
    
    return scaled_df.values

@routerEX.post("/predict/", response_model=PredictionResponse)
async def predict_expenses(expense_data: ExpenseList):
    try:
        if len(expense_data.expenses) < 20:
            raise HTTPException(
                status_code=400,
                detail=f"Please provide at least 20 historical expense records. Currently provided: {len(expense_data.expenses)}"
            )
        
        # Prepare and scale the input data
        sequence_data = prepare_sequence_data(expense_data.expenses, scaler)
        
        # Convert to tensor and make prediction
        with torch.no_grad():
            try:
                input_seq = torch.FloatTensor(sequence_data).unsqueeze(0).to(device)
                predictions = model(input_seq)
                predictions = predictions.cpu().numpy()[0]
                
                # Create DataFrame for inverse transform with correct feature names
                feature_names = ['hour', 'day', 'month', 'day_of_week', 'amount']
                pred_df = pd.DataFrame(
                    np.zeros((len(predictions), len(feature_names))),
                    columns=feature_names
                )
                pred_df['amount'] = predictions
                
                # Inverse transform
                inversed_predictions = scaler.inverse_transform(pred_df)
                final_predictions = inversed_predictions[:, -1].tolist()
                
                return PredictionResponse(predicted_expenses=final_predictions)
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction error: {str(e)}"
                )
                
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )