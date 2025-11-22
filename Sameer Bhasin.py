import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import io
import re

# --- 1. File Paths and Data Loading ---

# 🛑 FILE 1: Original Loan Data (Used for all risk calculations and visualizations)
LOAN_DATA_FILE = 'C:/Users/sinha/Downloads/MA Solved/loan_default.xlsx'

# AUC Value (extracted from the 'Test and Score' results)
auc_value = 0.655 

# Metrics from the 'Test and Score' table
model_metrics = {
    'AUC': 0.655,
    'CA': 0.670,
    'F1': 0.569,
    'Precision': 0.536,
    'Recall': 0.607
}

# 🛑 IMAGE URL: Placeholder for the Confusion Matrix Image
# *** IMPORTANT: REPLACE 'YOUR_DIRECT_IMAGE_URL_HERE' with the link you copied from Imgur. ***
CONFUSION_MATRIX_IMAGE_URL = 'https://i.postimg.cc/nc23w0G9/Whats-App-Image-2025-11-22-at-15-31-16-35e51d99.jpg' 

try:
    # Load data from the main CSV file
    df = pd.read_excel(LOAN_DATA_FILE)
    df['Default_Numeric'] = df['Default'].apply(lambda x: 1 if x == 'Yes' else 0)

except Exception as e:
    print(f"Error loading files: {e}. Please ensure the data files are in the same directory.")
    exit()


# --- 2. Calculation Functions ---

def calculate_default_rate(dataframe):
    """Calculates the overall default rate."""
    return dataframe['Default_Numeric'].mean()

def calculate_risk_by_feature(dataframe, feature):
    """Calculates default rate per category of a given feature."""
    risk_df = dataframe.groupby(feature)['Default_Numeric'].mean().reset_index()
    risk_df.columns = [feature, 'Default_Rate']
    return risk_df.sort_values(by='Default_Rate', ascending=False)


# Run the calculations
overall_default_rate = calculate_default_rate(df)
risk_by_employment = calculate_risk_by_feature(df, 'Employment_Type')
risk_by_credit_history = calculate_risk_by_feature(df, 'Credit_History')


# --- 3. Dashboard Components ---

# 3.1. Top Row: Default Rate Gauge
default_gauge = dcc.Graph(
    figure={
        'data': [{
            'type': 'indicator', 'mode': 'gauge+number', 'value': overall_default_rate,
            'title': {'text': "Overall Loan Default Rate"},
            'gauge': {'axis': {'range': [0, 1]}, 'bar': {'color': "#1E90FF"},
                      'steps': [{'range': [0, 0.2], 'color': "lightgreen"},
                                {'range': [0.2, 0.4], 'color': "yellow"},
                                {'range': [0.4, 1], 'color': "salmon"}],
                      'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': overall_default_rate}},
            'number': {'valueformat': '.1%'}
        }],
        'layout': {'height': 300, 'margin': {'t': 10, 'b': 10, 'l': 10, 'r': 10}}
    }
)

# 3.2. Middle Row: Risk by Employment (Bar Chart)
employment_bar = dcc.Graph(
    figure=px.bar(
        risk_by_employment, x='Employment_Type', y='Default_Rate',
        title='Risk by Employment Type (The Driver)', color='Default_Rate',
        color_continuous_scale=px.colors.sequential.YlOrRd,
        text=risk_by_employment['Default_Rate'].apply(lambda x: f'{x:.1%}')
    ).update_layout(yaxis_tickformat=".0%", uniformtext_minsize=8, uniformtext_mode='hide')
)

# 3.3. Middle Row: Risk by Credit History (Bar Chart)
credit_bar = dcc.Graph(
    figure=px.bar(
        risk_by_credit_history, x='Credit_History', y='Default_Rate',
        title='Risk by Credit History (The Driver)', color='Default_Rate',
        color_continuous_scale=px.colors.sequential.YlOrRd,
        text=risk_by_credit_history['Default_Rate'].apply(lambda x: f'{x:.1%}')
    ).update_layout(yaxis_tickformat=".0%", uniformtext_minsize=8, uniformtext_mode='hide')
)

# 3.4. Bottom Row: Confusion Matrix (Uses the online URL)
confusion_matrix_section = html.Div([
    html.H3("Model Performance: Confusion Matrix", style={'textAlign': 'center'}),
    # This line uses the online URL
    html.Img(src=CONFUSION_MATRIX_IMAGE_URL, 
             style={'width': '95%', 'height': 'auto', 'display': 'block', 'margin': '10px auto', 'border': '1px solid #ccc'}),
    html.P([
        html.Strong("Purpose:"),
        " The Confusion Matrix helps us evaluate the model's classifications. Rows show the actual class, and columns show the predicted class. The main diagonal represents correct predictions.",
        html.Br(),
        "The critical area is **False Negatives** (Actual Default, Predicted No-Default), which represent defaults the model failed to flag and pose a financial risk."
    ], style={'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'backgroundColor': '#F0F8FF'})
])

# 3.5. Bottom Row: AI Insight (Detailed Interpretation)
ai_insight = html.Div([
    html.H4("Model Performance Metrics & Interpretation", style={'textAlign': 'center'}),
    html.Ul([
        html.Li([html.Strong("AUC (Area Under Curve): "), f"{model_metrics['AUC']:.3f}. This is a **moderate score**, indicating the model is better than random (0.5) but not highly accurate."]),
        html.Li([html.Strong("CA (Accuracy): "), f"{model_metrics['CA']:.1%} of all cases were classified correctly."]),
        html.Li([html.Strong("Precision: "), f"{model_metrics['Precision']:.1%}. When the model predicts default, it is only correct about half the time. This suggests a high rate of False Positives (false alarms)."]),
        html.Li([html.Strong("Recall: "), f"{model_metrics['Recall']:.1%}. The model successfully identifies **60.7% of actual defaulters**. This is the critical measure for risk managers—it shows the model's ability to catch true risks."]),
    ], style={'padding-left': '20px'}),
    html.P([
        html.Strong("Key Drivers:"),
        " Analysis confirms that **Credit\_History** and **Employment\_Type** are the primary factors driving loan risk, which should guide your initial screening process."
    ], style={'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'backgroundColor': '#F0F8FF', 'marginTop': '10px'})
])

# --- 4. Dashboard Layout ---

app = Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#f9f9f9', 'padding': '20px'}, children=[
    html.H1("Loan Default Risk Dashboard", style={'textAlign': 'center', 'color': '#333'}),
    html.Hr(),

    # Top Row: Default Rate
    html.Div([html.Div(default_gauge, style={'width': '50%', 'margin': '0 auto'})], style={'marginBottom': '20px'}),
    html.Hr(),

    # Middle Row: The Drivers
    html.Div([
        html.H2("The Drivers: Feature-Specific Risk", style={'textAlign': 'center', 'color': '#555'}),
        html.Div([
            html.Div(employment_bar, style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div(credit_bar, style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
        ], style={'display': 'flex', 'flexDirection': 'row'})
    ], style={'marginBottom': '20px'}),
    html.Hr(),

    # Bottom Row: Model Evaluation and Insight
    html.Div([
        html.H2("Model Evaluation & Insights", style={'textAlign': 'center', 'color': '#555'}),
        html.Div([
            # Confusion Matrix Section
            html.Div(confusion_matrix_section, style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
            
            # AI Insight/Interpretation
            html.Div(ai_insight, style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
            
        ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'flex-start'})
    ])
])

# --- 5. Run App ---
if __name__ == '__main__':
    print("\n\nDashboard code prepared.")
    print("FINAL STEP: Ensure 'YOUR_DIRECT_IMAGE_URL_HERE' has been replaced with your actual Imgur link.")
    # This is the corrected command for running the server:
    app.run(debug=True)