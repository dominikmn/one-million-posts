import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

from utils import loading

data = loading.load_extended_posts()
data_plot = data.groupby("id_article")["id_post"].count().reset_index()

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
app.title = "One Million Posts"

app.layout = html.Div(
    children = [
        html.H1(children="One Million Posts",),
        html.P(children="Annotate newspaper posts regarding negative-sentiment, off-topic, discrimination, and inappropriateness.",),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": data_plot["id_article"],
                        "y": data_plot["id_post"],
                        "type": "dots",
                    },
                ],
                "layout": {
                    "title": "Number of posts for each Article",
                },
            }
        )
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
