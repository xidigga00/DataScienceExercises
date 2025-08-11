import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(0)
dates = [datetime.today() - timedelta(days=i) for i in range(9, -1, -1)]
temps = [30 + np.sin(i / 2) * 3 + np.random.rand() * 2 for i in range(10)]

# Put into a DataFrame
df = pd.DataFrame({'Date': dates, 'Temperature': temps})

x = list(range(10))  # 0 to 9
y = df['Temperature']

coeffs = np.polyfit(x, y, 1) 
print("Slope:", coeffs[0], "Intercept:", coeffs[1])

# Predict tomorrow (x=10)
tomorrow_temp = coeffs[0] * 10 + coeffs[1]
print("Forecast for Tomorrow:", round(tomorrow_temp, 2))

# Predict values using the line
y_pred = coeffs[0] * np.arange(10) + coeffs[1]

# Calculate R-squared manually
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"R² (Goodness of Fit): {r2:.2f}")


import dash
from dash import dcc, html
import plotly.graph_objs as go

app = dash.Dash(__name__)

# Line for past 10 days
past_trace = go.Scatter(
    x=df['Date'],
    y=df['Temperature'],
    mode='lines+markers',
    name='Last 10 Days',
    line=dict(color='blue')
)

# Point for tomorrow's forecast
forecast_trace = go.Scatter(
    x=[datetime.today() + timedelta(days=1)],
    y=[tomorrow_temp],
    mode='markers',
    name='Tomorrow Forecast',
    marker=dict(size=12, color='red', symbol='star')
)

# Combine both into one figure
fig = go.Figure(data=[past_trace, forecast_trace])

# Add labels
fig.update_layout(
    title='Temperature Trend with Forecast',
    xaxis_title='Date',
    yaxis_title='Temperature (°C)',
)

app.layout = html.Div([
    html.H1("Weather Forecast App"),
    dcc.Graph(
        id='weather-graph',
        figure=fig
    ),
    html.Div(f"Forecast for Tomorrow: {tomorrow_temp:.2f} °C", style={'fontSize': 18, 'marginTop': 20})
])

if __name__ == '__main__':
    app.run(debug=True)
