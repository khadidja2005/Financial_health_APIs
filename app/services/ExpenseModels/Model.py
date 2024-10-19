import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

class ExpenseDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length -10
        
    def __getitem__(self, idx):
      x = self.data[idx:idx + self.sequence_length]        # Input sequence
      y = self.data[idx + self.sequence_length:idx + self.sequence_length + 10]  # Target sequence
      return x, y

def load_and_prepare_data(csv_path):
    """
    Load and prepare data from CSV file with Date and Amount columns
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    sequence_length = df.shape[0]
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date to ensure temporal order
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # Extract time-based features
    df['hour'] = df['Date'].dt.hour
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek
    
    # Scale the features
    scaler = MinMaxScaler()
    features = ['hour', 'day', 'month', 'day_of_week', 'Amount']
    scaled_data = scaler.fit_transform(df[features])
    
    return scaled_data, scaler, df

def save_model(model, scaler, save_dir='saved_models'):
    """Save the trained model and scaler"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'expense_predictor.pth')
    torch.save(model.state_dict(), model_path)
    
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    print(f"Model and scaler saved in {save_dir}")

class ExpensePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_length):
        super(ExpensePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_length = output_length
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_length * input_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = out[:, -1, :] 
        out = self.fc(out)
        out = out.view(batch_size, self.output_length, self.input_size)
        return out

def train_model(model, train_loader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def predict_future_expenses(model, last_sequence, scaler, device):
    model.eval()
    with torch.no_grad():
        input_seq = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        predictions = model(input_seq)
        
        predictions = predictions.cpu().numpy()[0]
        inversed_predictions = scaler.inverse_transform(predictions)
        amount_predictions = inversed_predictions[:, -1]  # Amount is the last column
        
        return amount_predictions

def main(csv_path):
    # Hyperparameters
    sequence_length = 20
    hidden_size = 64
    num_layers = 2
    batch_size = 32
    num_epochs = 500
    learning_rate = 0.001
    prediction_length = 10
    
    print(f"Loading data from {csv_path}...")
    scaled_data, scaler, df = load_and_prepare_data(csv_path)
    
    input_size = scaled_data.shape[1]
    
    # Create dataset and dataloader
    dataset = ExpenseDataset(scaled_data, sequence_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ExpensePredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_length=prediction_length
    ).to(device)
    
    # Train model
    print("\nStarting training...")
    train_model(model, train_loader, num_epochs, learning_rate, device)
    
    # Save the trained model
    save_model(model, scaler)
    
    # Make predictions
    last_sequence = scaled_data[-sequence_length:]
    predictions = predict_future_expenses(model, last_sequence, scaler, device)
    #print(predictions)
    # Print predictions with dates
    #last_date = df['Date'].iloc[-1]
    #future_dates = pd.date_range(start=last_date, periods=prediction_length + 1)[1:]
    
    # print("\nPredicted expenses for the next 10 periods:")
    # for date, amount in zip(future_dates, predictions):
    #     print(f"{date.strftime('%Y-%m-%d %H:%M')}: ${amount:.2f}")

if __name__ == "__main__":
    csv_path = "expense_data_2.csv" 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path
    abs_file_path = os.path.abspath(os.path.join(base_dir, csv_path))
    main(abs_file_path)