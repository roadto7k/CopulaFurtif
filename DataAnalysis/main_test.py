import dash
from dash import dcc, html
import plotly.graph_objs as go

app = dash.Dash(__name__)

def empty_fig(msg="Test"):
    return go.Figure(layout=dict(
        template='plotly_dark',
        title=msg,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[{
            "text": msg,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 22, "color": "red"},
            "x": 0.5, "y": 0.5
        }]
    ))

app.layout = html.Div([
    html.H1("Test bug graph growing"),
    html.Div([
        dcc.Graph(id='adf-plot', figure=empty_fig(), style={'height': '400px'}),
        dcc.Graph(id='kss-plot', figure=empty_fig(), style={'height': '400px'}),
        dcc.Graph(id='rolling-plot', figure=empty_fig(), style={'height': '400px'}),
    ], style={'display':'flex'}),
])

app.run(debug=True)
