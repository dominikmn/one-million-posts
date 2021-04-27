import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import requests
from typing import Dict

import plotly.express as px
import plotly.graph_objects as go

COLOR_PINK = "#ec008e"
COLOR_GREY = "#bdbdbd"
URL_BACKEND = 'http://127.0.0.1:8000/predict'


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
app.title = "One Million Posts"

app.layout = html.Div(
    children = [
        html.Div(
            children=[
                html.H1(children="One Million Posts", className="header-title"),
                html.H6(
                    children="Assisting newspaper moderators with machine learning",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.P(
                    children="Enter your post:",
                ),
                html.Div(
                    dcc.Textarea(
                        id="post-input",
                        value="",
                        style={"width": "100%", "height": 100},
                    ),
                ),
                html.Button(id='submit-state', n_clicks=0, children='Analyse'),
                html.H6(id="post-prediction", style={'whiteSpace': 'pre-line'}),
                html.Div(
                    children = dcc.Graph(
                        id="prediction-chart", config={"displayModeBar": False},
                    ),
                    id="display-prediction-chart",
                    style={"visibility": "hidden"},
                    className="card",
                ),
            ],
        className="wrapper",
        ),
        html.Div(
            children=[
                html.Footer("https://github.com/dominikmn/one-million-posts/"),
            ],
            className="footer",
        )
    ]
)


@app.callback(
    [
        Output(component_id="post-prediction", component_property="children"),
        Output(component_id="post-prediction", component_property="style"),
        Output(component_id="display-prediction-chart", component_property="style"),
        Output(component_id="prediction-chart", component_property="figure"),
    ],
    [Input(component_id="submit-state", component_property="n_clicks")],
    [State(component_id="post-input", component_property="value")]
)
def update_prediction(n_clicks, input_value):
    if input_value:
        new_measurement = {"text": input_value}
        response = requests.post(URL_BACKEND, json=new_measurement)
        if response.ok:
            result = response.json()
            mapping_needsmoderation = {0: "Everything's fine", 1: "Needs moderation"}
            needs_moderation = result['needsmoderation']
            style_prediction = {
                0: {"color": "black"},
                1: {"color": COLOR_PINK},
            }
            style_prediction_chart = {
                0: {"visibility": "hidden"},
                1: {"visibility": "visible"},
            }
            predictions = result.copy()
            predictions.pop("needsmoderation")
            df = get_df_from_predictions(predictions)
            return (
                [f"{mapping_needsmoderation[needs_moderation]}"],
                style_prediction[needs_moderation],
                style_prediction_chart[needs_moderation],
                update_prediction_chart(df),
            )
    predictions = {'sentimentnegative': 0.0, 'inappropriate': 0.0, 'discriminating': 0.0}
    df = get_df_from_predictions(predictions)
    return (
        [f"No text provided"],
        {"color": "black"},
        {"visibility": "hidden"},
        update_prediction_chart(df),
    )

def get_df_from_predictions(predictions: Dict):
    data = [(category, prediction) for category, prediction in predictions.items()]
    return pd.DataFrame.from_records(data, columns=["category", "prediction"])


def update_prediction_chart(long_df):
    colors = [COLOR_GREY]*3
    colors[long_df.prediction.idxmax()] = COLOR_PINK
    fig = px.bar(long_df, x="category", y="prediction", template="none")
    fig = go.Figure(data=[go.Bar(
        x=long_df["prediction"],
        y=long_df["category"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{x}<extra></extra>",
    )])
    fig.layout = {
        "template": "none",
        "height": 250,
        "margin": {
            "t": 25,
            "r": 5,
            "b": 60,
        },
        "xaxis": {
            "title": "Probability",
            "tickformat": "%",
        },
        "yaxis": {
            "automargin": True,
            "tickvals": ["discriminating", "inappropriate", "sentimentnegative"],
            "ticktext": ["Discriminating ", "Inappropriate ", "Negative Sentiment "],
        },
    }

    return fig

if __name__ == "__main__":
    app.run_server(debug=False)
