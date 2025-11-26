"""Dash web interface for Code Plagiarism Detector."""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import base64
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from code_plagiarism_detector import CodeHasher, CodeHashDatabase, load_code_bank

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
app.title = "Code Plagiarism Detector"

# Initialize hasher and database
hasher = CodeHasher()
db = CodeHashDatabase("reference_hashes.db")

# Load code_bank on startup
code_bank_path = Path(__file__).parent / "code_bank"
loaded_count = load_code_bank(str(code_bank_path), hasher, db)
print(f"Loaded {loaded_count} reference files from code_bank")


def infer_language(filename: str) -> str:
    """Infer language from file extension."""
    if not filename:
        return 'python'
    ext = Path(filename).suffix.lower()
    lang_map = {
        '.py': 'python',
        '.java': 'java',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c': 'cpp',
        '.h': 'cpp',
        '.hpp': 'cpp'
    }
    return lang_map.get(ext, 'python')


# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🔍 Code Plagiarism Detector", className="text-center mb-2"),
            html.P("AST-based plagiarism detection with perceptual hashing", 
                   className="text-center text-muted"),
            html.Div([
                dbc.Badge(f"{db.get_stats()['total_hashes']} reference files loaded", 
                         color="success", className="me-2"),
                dbc.Badge("Python • Java • C++", color="info")
            ], className="text-center mb-3")
        ])
    ]),
    
    dbc.Tabs([
        dbc.Tab(label="🆚 Compare Files", tab_id="compare"),
        dbc.Tab(label="📊 Batch Analysis", tab_id="batch"),
        dbc.Tab(label="🔍 Check Database", tab_id="check"),
        dbc.Tab(label="💾 Database", tab_id="database"),
        dbc.Tab(label="ℹ️ About", tab_id="about"),
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
    elif active_tab == "check":
        return check_tab()
    elif active_tab == "database":
        return database_tab()
    else:
        return about_tab()


def compare_tab():
    """Compare two files tab."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4("File 1"),
                dcc.Upload(
                    id='upload-file1',
                    children=dbc.Card([
                        dbc.CardBody([
                            "📁 Drag & Drop or Click to Select"
                        ], className="text-center")
                    ], style={'cursor': 'pointer'}),
                    style={'width': '100%'}
                ),
                dbc.Card([
                    dbc.CardBody([
                        html.Pre(id='file1-content', style={
                            'maxHeight': '300px', 
                            'overflow': 'auto',
                            'fontSize': '12px',
                            'margin': '0'
                        })
                    ])
                ], className="mt-2")
            ], md=6),
            
            dbc.Col([
                html.H4("File 2"),
                dcc.Upload(
                    id='upload-file2',
                    children=dbc.Card([
                        dbc.CardBody([
                            "📁 Drag & Drop or Click to Select"
                        ], className="text-center")
                    ], style={'cursor': 'pointer'}),
                    style={'width': '100%'}
                ),
                dbc.Card([
                    dbc.CardBody([
                        html.Pre(id='file2-content', style={
                            'maxHeight': '300px', 
                            'overflow': 'auto',
                            'fontSize': '12px',
                            'margin': '0'
                        })
                    ])
                ], className="mt-2")
            ], md=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Button("🔍 Compare", id="compare-btn", color="primary", 
                          className="mt-3 mb-3", size="lg")
            ], className="text-center")
        ]),
        
        html.Div(id='compare-results')
    ])


def batch_tab():
    """Batch analysis tab."""
    return dbc.Container([
        html.H4("Batch Analysis"),
        html.P("Upload multiple files to compare all pairs"),
        
        dcc.Upload(
            id='upload-batch',
            children=dbc.Card([
                dbc.CardBody([
                    "📁 Drag & Drop Multiple Files or Click to Select"
                ], className="text-center")
            ], style={'cursor': 'pointer'}),
            multiple=True
        ),
        
        html.Div(id='batch-files-list', className="mt-2"),
        
        html.Label("Similarity Threshold:", className="mt-3"),
        dcc.Slider(
            id='batch-threshold',
            min=0, max=1, step=0.05, value=0.60,
            marks={i/10: f'{i*10}%' for i in range(0, 11, 2)},
            className="mb-3"
        ),
        
        dbc.Button("🔍 Analyze All Pairs", id="batch-btn", color="primary", 
                  className="mb-3", size="lg"),
        
        html.Div(id='batch-results')
    ])


def check_tab():
    """Check against database tab."""
    stats = db.get_stats()
    return dbc.Container([
        html.H4("Check Against Reference Database"),
        html.P(f"Compare your code against {stats['total_hashes']} reference implementations"),
        
        dcc.Upload(
            id='upload-check',
            children=dbc.Card([
                dbc.CardBody([
                    "📁 Upload Code to Check"
                ], className="text-center")
            ], style={'cursor': 'pointer'})
        ),
        
        html.Div(id='check-file-info', className="mt-2"),
        
        html.Label("Minimum Similarity:", className="mt-3"),
        dcc.Slider(
            id='check-threshold',
            min=0, max=1, step=0.05, value=0.50,
            marks={i/10: f'{i*10}%' for i in range(0, 11, 2)}
        ),
        
        dbc.Button("🔍 Check for Matches", id="check-btn", color="primary", 
                  size="lg", className="mt-3"),
        
        html.Div(id='check-results', className="mt-3")
    ])


def database_tab():
    """Database management tab."""
    return dbc.Container([
        html.H4("Database Management"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Add Reference Code"),
                    dbc.CardBody([
                        dbc.Input(id="db-source", placeholder="Source (e.g., leetcode, homework1)"),
                        dcc.Upload(
                            id='upload-db',
                            children=html.Div(["📁 Upload File"]),
                            style={
                                'borderWidth': '1px', 'borderStyle': 'dashed',
                                'borderRadius': '5px', 'textAlign': 'center',
                                'padding': '10px', 'marginTop': '10px',
                                'cursor': 'pointer'
                            }
                        ),
                        html.Div(id='db-file-info', className="mt-2"),
                        dbc.Button("💾 Add to Database", id="db-add-btn", color="primary", className="mt-2"),
                        html.Div(id="db-add-result", className="mt-2")
                    ])
                ])
            ], md=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Statistics & Actions"),
                    dbc.CardBody([
                        dbc.Button("🔄 Refresh Stats", id="db-stats-btn", color="secondary"),
                        dbc.Button("🔄 Reload Code Bank", id="db-reload-btn", color="warning", className="ms-2"),
                        html.Div(id="db-stats-result", className="mt-3")
                    ])
                ])
            ], md=6)
        ])
    ])


def about_tab():
    """About tab."""
    return dbc.Container([
        html.H4("About Code Plagiarism Detector"),
        html.Hr(),
        
        dbc.Row([
            dbc.Col([
                html.H5("🎯 Features"),
                html.Ul([
                    html.Li("AST-based structural analysis"),
                    html.Li("Cross-language detection (Python ↔ Java ↔ C++)"),
                    html.Li("256-bit perceptual hashing (LSH)"),
                    html.Li("Reference database with code_bank"),
                    html.Li("Batch comparison support"),
                    html.Li("Auto language detection from file extension")
                ]),
            ], md=6),
            
            dbc.Col([
                html.H5("📊 How It Works"),
                html.Ol([
                    html.Li("Parse code → AST (tree-sitter)"),
                    html.Li("Extract control flow (LOOP, COND)"),
                    html.Li("Normalize patterns across languages"),
                    html.Li("Generate shingles → LSH hash"),
                    html.Li("Compare: Hamming (syntactic) + Jaccard (structural)")
                ]),
            ], md=6)
        ]),
        
        html.Hr(),
        
        html.H5("🚦 Confidence Levels"),
        dbc.Row([
            dbc.Col(dbc.Alert("🔴 HIGH: ≥60% structural match", color="danger"), md=4),
            dbc.Col(dbc.Alert("🟡 MEDIUM: 40-60% match", color="warning"), md=4),
            dbc.Col(dbc.Alert("🟢 LOW: <40% match", color="success"), md=4),
        ]),
        
        html.Hr(),
        
        html.H5("📁 Supported Languages"),
        dbc.Row([
            dbc.Col(dbc.Badge("Python (.py)", color="primary", className="me-2 p-2")),
            dbc.Col(dbc.Badge("Java (.java)", color="danger", className="me-2 p-2")),
            dbc.Col(dbc.Badge("C++ (.cpp, .cc, .h, .hpp)", color="info", className="p-2")),
        ])
    ])


# ============================================================================
# Callbacks
# ============================================================================

@app.callback(
    Output('file1-content', 'children'),
    Input('upload-file1', 'contents'),
    State('upload-file1', 'filename')
)
def display_file1(contents, filename):
    if contents:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        lang = infer_language(filename)
        return f"📄 {filename} [{lang}]\n{'─'*40}\n{decoded[:2000]}{'...' if len(decoded) > 2000 else ''}"
    return "No file uploaded"


@app.callback(
    Output('file2-content', 'children'),
    Input('upload-file2', 'contents'),
    State('upload-file2', 'filename')
)
def display_file2(contents, filename):
    if contents:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        lang = infer_language(filename)
        return f"📄 {filename} [{lang}]\n{'─'*40}\n{decoded[:2000]}{'...' if len(decoded) > 2000 else ''}"
    return "No file uploaded"


@app.callback(
    Output('compare-results', 'children'),
    Input('compare-btn', 'n_clicks'),
    State('upload-file1', 'contents'),
    State('upload-file1', 'filename'),
    State('upload-file2', 'contents'),
    State('upload-file2', 'filename'),
    prevent_initial_call=True
)
def compare_files(n_clicks, contents1, filename1, contents2, filename2):
    if not contents1 or not contents2:
        return dbc.Alert("Please upload both files", color="warning")
    
    try:
        _, content_string1 = contents1.split(',')
        code1 = base64.b64decode(content_string1).decode('utf-8')
        
        _, content_string2 = contents2.split(',')
        code2 = base64.b64decode(content_string2).decode('utf-8')
        
        # Infer languages from filenames
        lang1 = infer_language(filename1)
        lang2 = infer_language(filename2)
        
        # Compare
        result = hasher.compare(code1, lang1, code2, lang2)
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result.similarity * 100,
            title={'text': f"Similarity ({result.method_used})"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 70], 'color': "yellow"},
                    {'range': [70, 85], 'color': "orange"},
                    {'range': [85, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        
        color_map = {'high': 'danger', 'medium': 'warning', 'low': 'success'}
        alert_color = color_map.get(result.confidence, 'info')
        
        return dbc.Container([
            html.Hr(),
            html.H4("Results"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Files Compared"),
                            html.Small(f"📄 {filename1} ({lang1})"),
                            html.Br(),
                            html.Small(f"📄 {filename2} ({lang2})")
                        ])
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Overall Similarity"),
                            html.H2(f"{result.similarity:.1%}")
                        ])
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Structural Match"),
                            html.H2(f"{result.structural_similarity:.1%}")
                        ])
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Confidence"),
                            dbc.Badge(result.confidence.upper(), color=alert_color, className="fs-4")
                        ])
                    ])
                ], md=3)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig)
                ], md=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Detection Details"),
                        dbc.CardBody([
                            html.P([html.Strong("Syntactic: "), f"{result.syntactic_similarity:.1%}"]),
                            html.P([html.Strong("Structural: "), f"{result.structural_similarity:.1%}"]),
                            html.P([html.Strong("Hamming: "), f"{result.hamming_distance}/256"]),
                            html.P([html.Strong("Patterns: "), result.pattern_match_ratio]),
                            html.Hr(),
                            html.P("Matching Patterns:", className="fw-bold"),
                            html.Ul([html.Li(p, style={'fontSize': '11px'}) 
                                    for p in result.matching_patterns[:5]]) if result.matching_patterns else html.P("None", className="text-muted")
                        ])
                    ])
                ], md=4)
            ]),
            
            dbc.Alert(
                f"{'⚠️ PLAGIARISM DETECTED' if result.plagiarism_detected else '✅ No significant plagiarism detected'}",
                color=alert_color,
                className="mt-3"
            )
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")


@app.callback(
    Output('batch-files-list', 'children'),
    Input('upload-batch', 'filename')
)
def show_batch_files(filenames):
    if filenames:
        return dbc.Alert(f"Selected {len(filenames)} files: {', '.join(filenames[:5])}{'...' if len(filenames) > 5 else ''}", color="info")
    return ""


@app.callback(
    Output('batch-results', 'children'),
    Input('batch-btn', 'n_clicks'),
    State('upload-batch', 'contents'),
    State('upload-batch', 'filename'),
    State('batch-threshold', 'value'),
    prevent_initial_call=True
)
def batch_analysis(n_clicks, contents_list, filenames, threshold):
    if not contents_list or len(contents_list) < 2:
        return dbc.Alert("Please upload at least 2 files", color="warning")
    
    try:
        # Decode all files
        files = []
        for contents, filename in zip(contents_list, filenames):
            _, content_string = contents.split(',')
            code = base64.b64decode(content_string).decode('utf-8')
            lang = infer_language(filename)
            files.append({'name': filename, 'code': code, 'lang': lang})
        
        # Compare all pairs
        results = []
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                result = hasher.compare(
                    files[i]['code'], files[i]['lang'],
                    files[j]['code'], files[j]['lang']
                )
                
                if result.similarity >= threshold:
                    results.append({
                        'File 1': files[i]['name'],
                        'File 2': files[j]['name'],
                        'Similarity': f"{result.similarity:.1%}",
                        'Structural': f"{result.structural_similarity:.1%}",
                        'Confidence': result.confidence.upper()
                    })
        
        if not results:
            return dbc.Alert(f"✅ No pairs found above {threshold:.0%} threshold", color="success")
        
        results.sort(key=lambda x: x['Similarity'], reverse=True)
        
        import pandas as pd
        return dbc.Container([
            dbc.Alert(f"⚠️ Found {len(results)} suspicious pairs", color="warning"),
            dbc.Table.from_dataframe(
                pd.DataFrame(results),
                striped=True, bordered=True, hover=True, size='sm'
            )
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")


@app.callback(
    Output('check-file-info', 'children'),
    Input('upload-check', 'filename')
)
def show_check_file(filename):
    if filename:
        lang = infer_language(filename)
        return dbc.Alert(f"📄 {filename} [{lang}]", color="info")
    return ""


@app.callback(
    Output('check-results', 'children'),
    Input('check-btn', 'n_clicks'),
    State('upload-check', 'contents'),
    State('upload-check', 'filename'),
    State('check-threshold', 'value'),
    prevent_initial_call=True
)
def check_against_db(n_clicks, contents, filename, threshold):
    if not contents:
        return dbc.Alert("Please upload a file", color="warning")
    
    try:
        _, content_string = contents.split(',')
        code = base64.b64decode(content_string).decode('utf-8')
        lang = infer_language(filename)
        
        # Hash the code
        query_hash = hasher.hash_code(code, lang)
        
        # Search database
        matches = db.find_similar(query_hash, threshold=threshold)
        
        if not matches:
            return dbc.Alert("✅ No matches found - code appears original!", color="success")
        
        import pandas as pd
        rows = []
        for match in matches[:10]:
            rows.append({
                'Source': match['source'],
                'File': match['file_path'],
                'Similarity': f"{match['similarity']:.1%}",
                'Language': match['language']
            })
        
        return dbc.Container([
            dbc.Alert(f"⚠️ Found {len(matches)} potential matches!", color="warning"),
            dbc.Table.from_dataframe(
                pd.DataFrame(rows),
                striped=True, bordered=True, hover=True, size='sm'
            )
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")


@app.callback(
    Output('db-file-info', 'children'),
    Input('upload-db', 'filename')
)
def show_db_file(filename):
    if filename:
        lang = infer_language(filename)
        return html.Small(f"📄 {filename} [{lang}]")
    return ""


@app.callback(
    Output('db-add-result', 'children'),
    Input('db-add-btn', 'n_clicks'),
    State('upload-db', 'contents'),
    State('upload-db', 'filename'),
    State('db-source', 'value'),
    prevent_initial_call=True
)
def add_to_database(n_clicks, contents, filename, source):
    if not contents or not source:
        return dbc.Alert("Please upload file and enter source", color="warning")
    
    try:
        _, content_string = contents.split(',')
        code = base64.b64decode(content_string).decode('utf-8')
        lang = infer_language(filename)
        
        hash_value = hasher.hash_code(code, lang)
        
        db.add_hash(
            source=source,
            file_path=filename,
            language=lang,
            hash_value=hash_value,
            metadata={'added_via': 'web_ui'}
        )
        
        return dbc.Alert(f"✅ Added {filename} [{lang}] to database", color="success")
        
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")


@app.callback(
    Output('db-stats-result', 'children'),
    Input('db-stats-btn', 'n_clicks'),
    Input('db-reload-btn', 'n_clicks'),
    prevent_initial_call=True
)
def refresh_stats(stats_clicks, reload_clicks):
    ctx = dash.callback_context
    
    if ctx.triggered_id == 'db-reload-btn':
        loaded = load_code_bank(str(code_bank_path), hasher, db, force_reload=True)
        return dbc.Alert(f"✅ Reloaded {loaded} files from code_bank", color="success")
    
    stats = db.get_stats()
    
    return dbc.Container([
        html.H6(f"📁 Total files: {stats['total_hashes']}"),
        html.Hr(),
        html.P("By Language:", className="fw-bold mb-1"),
        html.Ul([html.Li(f"{lang}: {count}") for lang, count in stats.get('by_language', {}).items()]),
        html.P("By Source:", className="fw-bold mb-1"),
        html.Ul([html.Li(f"{src}: {count}") for src, count in stats.get('by_source', {}).items()])
    ])


if __name__ == '__main__':
    stats = db.get_stats()
    print(f"Starting server with {stats['total_hashes']} reference files...")
    print(f"Languages: {stats.get('by_language', {})}")
    print(f"Sources: {stats.get('by_source', {})}")
    app.run(debug=True, port=8050)
