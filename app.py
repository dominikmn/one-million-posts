import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

from utils import loading

#data = loading.load_extended_posts()
#data_plot = data.groupby("id_article")["id_post"].count().reset_index()

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#server = app.server
app.title = "One Million Posts"

app.layout = html.Div(
    children = [
        html.Div(
            children=[
                html.H1(children="One Million Posts", className="header-title"),
                html.P(
                    children="Annotate newspaper posts regarding negative-sentiment, off-topic, discrimination, and inappropriateness.",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                #html.Div(
                    #children=dcc.Graph(
                        #id="post-per-article-chart",
                        #config={"displayModeBar": False},
                        #figure={
                            #"data": [
                                #{
                                    #"x": data_plot["id_article"],
                                    #"y": data_plot["id_post"],
                                    #"type": "lines",
                                    #"hovertemplate": "%{y:.2f}" "<extra></extra>",
                                #},
                            #],
                            #"layout": {
                                #"title": {
                                    #"text": "Number of posts for each Article",
                                    #"x": 0.05,
                                    #"xanchor": "left",
                                #},
                                #"xaxis": {
                                    #"text": "Article ID",
                                    #"fixedrange": True,
                                #},
                                #"yaxis": {
                                    #"text": "Number of posts",
                                    #"fixedrange": True,
                                #},
                                #"colorway": ["#17B897"],
                            #},
                        #},
                    #),
                    #className="card",
                #),
                html.Div(
                    children=[
                        html.H6(
                            children="Enter the post that shall be analysed:",
                        ),
                        html.Div(
                            dcc.Textarea(
                                id="post-input",
                                value="",
                                style={"width": "100%", "height": 100},
                            ),
                        ),
                        html.Br(),
                        html.Div(id="post-output", style={'whiteSpace': 'pre-line'}),
                    ],
                ),
            ],
        className="wrapper",
        ),
    ]
)

@app.callback(
    [Output(component_id="post-output", component_property="children")],
    [Input(component_id="post-input", component_property="value")],
)
def update_output_div(input_value):
    return ["Output: {}".format(input_value)]

if __name__ == "__main__":
    app.run_server(debug=True)
