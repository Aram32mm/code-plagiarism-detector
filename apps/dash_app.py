"""Dash web interface for Code Plagiarism Detector."""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import base64
import io
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from code_plagiarism_detector import CodeHasher, CodeHashDatabase

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Code Plagiarism Detector"

# Initialize hasher
hasher = CodeHasher()
db_path = "web_reference.db"

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🔍 Code Plagiarism Detector", className="text-center mb-4"),
            html.P("AST-based plagiarism detection with perceptual hashing", 
                   className="text-center text-muted")
        ])
    ]),
    
    dbc.Tabs([
        dbc.Tab(label="Compare Files", tab_id="compare"),
        dbc.Tab(label="Batch Analysis", tab_id="batch"),
        dbc.Tab(label="Database", tab_id="database"),
        dbc.Tab(label="About", tab_id="about"),
    ], id="tabs", active_tab="compare"),
    
    html.Div(id="tab-content", className="mt-4")
], fluid=True, className="p-4")


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    """Render content based on active tab."""
    if active_tab == "compare":
        return compare_tab()
    elif active_tab == "batch":
        return batch_tab()
    elif active_tab == "database":
        return database_tab()
    else:
        return about_tab()


def compare_tab():
    """Compare files tab layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("File 1"),
                dcc.Upload(
                    id='upload-file1',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select File')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    }
                ),
                dcc.Dropdown(
                    id='lang1',
                    options=[
                        {'label': 'Python', 'value': 'python'},
                        {'label': 'Java', 'value': 'java'},
                        {'label': 'C++', 'value': 'cpp'}
                    ],
                    value='python',
                    className="mb-3"
                ),
                html.Div(id='file1-content', style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace'})
            ], md=6),
            
            dbc.Col([
                html.H3("File 2"),
                dcc.Upload(
                    id='upload-file2',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select File')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    }
                ),
                dcc.Dropdown(
                    id='lang2',
                    options=[
                        {'label': 'Python', 'value': 'python'},
                        {'label': 'Java', 'value': 'java'},
                        {'label': 'C++', 'value': 'cpp'}
                    ],
                    value='python',
                    className="mb-3"
                ),
                html.Div(id='file2-content', style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace'})
            ], md=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Button("🔍 Compare", id="compare-btn", color="primary", className="mt-3 mb-3", size="lg")
            ], className="text-center")
        ]),
        
        html.Div(id='compare-results')
    ])


def batch_tab():
    """Batch analysis tab layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Batch Analysis"),
                html.P("Upload multiple files to compare all pairs"),
                
                dcc.Upload(
                    id='upload-batch',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Multiple Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=True
                ),
                
                html.Label("Similarity Threshold:"),
                dcc.Slider(
                    id='batch-threshold',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.85,
                    marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                    className="mb-3"
                ),
                
                dbc.Button("🔍 Analyze", id="batch-btn", color="primary", className="mb-3", size="lg"),
                
                html.Div(id='batch-results')
            ])
        ])
    ])


def database_tab():
    """Database management tab layout."""
    return dbc.Container([
        dbc.Tabs([
            dbc.Tab(label="Add to Database", children=[
                dbc.Container([
                    html.H4("Add Reference Code", className="mt-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Source Identifier:"),
                            dbc.Input(id="db-source", placeholder="e.g., leetcode, official"),
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Language:"),
                            dcc.Dropdown(
                                id='db-lang',
                                options=[
                                    {'label': 'Python', 'value': 'python'},
                                    {'label': 'Java', 'value': 'java'},
                                    {'label': 'C++', 'value': 'cpp'}
                                ],
                                value='python'
                            ),
                        ], md=6)
                    ], className="mb-3"),
                    
                    dcc.Upload(
                        id='upload-db-file',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select File')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        }
                    ),
                    
                    dbc.Button("💾 Add to Database", id="db-add-btn", color="primary", className="mt-3"),
                    html.Div(id="db-add-result", className="mt-3")
                ])
            ]),
            
            dbc.Tab(label="Check Against Database", children=[
                dbc.Container([
                    html.H4("Check Code", className="mt-3"),
                    
                    dcc.Upload(
                        id='upload-check-file',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select File to Check')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        }
                    ),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Language:"),
                            dcc.Dropdown(
                                id='check-lang',
                                options=[
                                    {'label': 'Python', 'value': 'python'},
                                    {'label': 'Java', 'value': 'java'},
                                    {'label': 'C++', 'value': 'cpp'}
                                ],
                                value='python'
                            ),
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Threshold:"),
                            dcc.Slider(
                                id='check-threshold',
                                min=0,
                                max=1,
                                step=0.05,
                                value=0.85,
                                marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)}
                            ),
                        ], md=6)
                    ], className="mb-3"),
                    
                    dbc.Button("🔍 Check", id="db-check-btn", color="primary", className="mt-3"),
                    html.Div(id="db-check-result", className="mt-3")
                ])
            ]),
            
            dbc.Tab(label="Statistics", children=[
                dbc.Container([
                    html.H4("Database Statistics", className="mt-3"),
                    dbc.Button("🔄 Refresh", id="db-stats-btn", color="secondary", className="mb-3"),
                    html.Div(id="db-stats-result")
                ])
            ])
        ])
    ])


def about_tab():
    """About tab layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("About Code Plagiarism Detector"),
                html.Hr(),
                
                html.H5("Features"),
                html.Ul([
                    html.Li("AST-based analysis using tree-sitter"),
                    html.Li("Perceptual hashing with LSH"),
                    html.Li("Multi-language support (Python, Java, C++)"),
                    html.Li("Database for reference code storage"),
                    html.Li("Fast Hamming distance comparison")
                ]),
                
                html.H5("How It Works"),
                html.Ol([
                    html.Li("Parse code into Abstract Syntax Tree"),
                    html.Li("Extract structural features"),
                    html.Li("Generate k-shingles"),
                    html.Li("Create 256-bit perceptual hash"),
                    html.Li("Compare using Hamming distance")
                ]),
                
                html.H5("Similarity Thresholds"),
                html.Ul([
                    html.Li("95-100%: Very high risk - likely plagiarism"),
                    html.Li("85-95%: High risk - possible plagiarism"),
                    html.Li("70-85%: Moderate risk - similar structure"),
                    html.Li("0-70%: Low risk - different code")
                ]),
                
                html.Hr(),
                html.P("Built with Dash and Plotly", className="text-muted")
            ])
        ])
    ])


# Callbacks for file uploads and processing
@app.callback(
    Output('file1-content', 'children'),
    Input('upload-file1', 'contents'),
    State('upload-file1', 'filename')
)
def display_file1(contents, filename):
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        return html.Pre(decoded[:1000] + ('...' if len(decoded) > 1000 else ''))
    return "No file uploaded"


@app.callback(
    Output('file2-content', 'children'),
    Input('upload-file2', 'contents'),
    State('upload-file2', 'filename')
)
def display_file2(contents, filename):
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        return html.Pre(decoded[:1000] + ('...' if len(decoded) > 1000 else ''))
    return "No file uploaded"


@app.callback(
    Output('compare-results', 'children'),
    Input('compare-btn', 'n_clicks'),
    State('upload-file1', 'contents'),
    State('upload-file2', 'contents'),
    State('lang1', 'value'),
    State('lang2', 'value'),
    prevent_initial_call=True
)
def compare_files(n_clicks, contents1, contents2, lang1, lang2):
    if not contents1 or not contents2:
        return dbc.Alert("Please upload both files", color="warning")
    
    try:
        # Decode files
        _, content_string1 = contents1.split(',')
        code1 = base64.b64decode(content_string1).decode('utf-8')
        
        _, content_string2 = contents2.split(',')
        code2 = base64.b64decode(content_string2).decode('utf-8')
        
        # Hash and compare
        hash1 = hasher.hash_code(code1, lang1)
        hash2 = hasher.hash_code(code2, lang2)
        similarity, hamming_dist = hasher.compare(hash1, hash2)
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=similarity * 100,
            title={'text': "Similarity Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgreen"},
                    {'range': [70, 85], 'color': "yellow"},
                    {'range': [85, 95], 'color': "orange"},
                    {'range': [95, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        
        # Assessment
        if similarity > 0.95:
            alert_color = "danger"
            assessment = "⚠️ VERY HIGH RISK - Likely plagiarism"
        elif similarity > 0.85:
            alert_color = "warning"
            assessment = "⚠️ HIGH RISK - Possible plagiarism"
        elif similarity > 0.70:
            alert_color = "info"
            assessment = "ℹ️ MODERATE RISK - Similar structure"
        else:
            alert_color = "success"
            assessment = "✅ LOW RISK - Different code"
        
        return dbc.Container([
            html.Hr(),
            html.H4("Results"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Similarity"),
                            html.H2(f"{similarity:.2%}")
                        ])
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Hamming Distance"),
                            html.H2(f"{hamming_dist}/256")
                        ])
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Assessment"),
                            dbc.Alert(assessment, color=alert_color)
                        ])
                    ])
                ], md=4)
            ], className="mb-3"),
            
            dcc.Graph(figure=fig)
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")


if __name__ == '__main__':
    app.run(debug=True, port=8050)
