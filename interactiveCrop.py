import json
from textwrap import dedent as d

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

app = dash.Dash(__name__)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

trace0 = go.Scatter(
    x=[0,10],
    y=[0,10],
    showlegend=False,
    mode = 'markers',
)

data = [trace0]
layout = {}
layout['images'] = []

## giotto @ https://www.thoughtco.com/thmb/WC0kY9VWk957sCRcgsWk8jcu96o=/768x0/filters:no_upscale():max_bytes(150000):strip_icc()/giotto-marriage-ofthe-virgin-56c8d8445f9b5879cc4523f9.jpg


layout['images'].append(dict(
    source= "http://localhost:8000/0016.png",
    xref= "x",
    yref= "y",
    x= 0,
    y= 0,
    sizex= 10,
    sizey= 10,
    xanchor= "left",
    yanchor= "bottom"  ##This really is how the two have to be defined..... seems pretty silly
  ))


app.layout = html.Div([
    dcc.Graph(
        id='basic-interactions',
        figure=go.Figure(data,layout=layout),
        style={'height': 1000}
    ),

    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown(d("""
                **Hover Data**

                Mouse over values in the graph.
            """)),
            html.Pre(id='hover-data', style=styles['pre'])
        ], className='three columns'),

        html.Div([
            dcc.Markdown(d("""
                **Click Data**

                Click on points in the graph.
            """)),
            html.Pre(id='click-data', style=styles['pre']),
        ], className='three columns'),

        html.Div([
            dcc.Markdown(d("""
                **Selection Data**

                Choose the lasso or rectangle tool in the graph's menu
                bar and then select points in the graph.
            """)),
            html.Pre(id='selected-data', style=styles['pre']),
        ], className='three columns'),

        html.Div([
            dcc.Markdown(d("""
                **Zoom and Relayout Data**

                Click and drag on the graph to zoom or click on the zoom
                buttons in the graph's menu bar.
                Clicking on legend items will also fire
                this event.
            """)),
            html.Pre(id='relayout-data', style=styles['pre']),
        ], className='three columns')
    ])
])


@app.callback(
    Output('hover-data', 'children'),
    [Input('basic-interactions', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    Output('click-data', 'children'),
    [Input('basic-interactions', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)


@app.callback(
    Output('selected-data', 'children'),
    [Input('basic-interactions', 'selectedData')])
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)


@app.callback(
    Output('relayout-data', 'children'),
    [Input('basic-interactions', 'relayoutData')])
def display_selected_data(relayoutData):
    return json.dumps(relayoutData, indent=2)


if __name__ == '__main__':
    app.run_server(debug=True)
