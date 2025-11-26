"""Streamlit web interface for Code Plagiarism Detector."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from code_plagiarism_detector import CodeHasher, CodeHashDatabase, load_code_bank

# Page config
st.set_page_config(
    page_title="Code Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {border-radius: 10px;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .code-box {
        background-color: #1e1e1e;
        border-radius: 5px;
        padding: 10px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_detector():
    """Initialize hasher and database once."""
    hasher = CodeHasher()
    db = CodeHashDatabase("reference_hashes.db")
    code_bank_path = Path(__file__).parent / "code_bank"
    loaded = load_code_bank(str(code_bank_path), hasher, db)
    return hasher, db, loaded


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


def get_confidence_color(confidence: str) -> str:
    """Get color for confidence level."""
    return {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(confidence, '⚪')


def create_similarity_gauge(value: float, title: str = "Similarity") -> go.Figure:
    """Create a gauge chart for similarity."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': '%', 'font': {'size': 40}},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 40], 'color': '#d4edda'},
                {'range': [40, 60], 'color': '#fff3cd'},
                {'range': [60, 80], 'color': '#ffe5b4'},
                {'range': [80, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# Initialize
hasher, db, loaded_count = init_detector()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="margin: 0; font-size: 2.5rem;">🔍</h1>
        <h3 style="margin: 0.5rem 0;">Plagiarism Detector</h3>
        <p style="color: #888; font-size: 0.8rem; margin: 0;">AST-based code similarity</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Navigation buttons
    pages = {
        "compare": ("🆚", "Compare Files"),
        "batch": ("📊", "Batch Analysis"),
        "search": ("🔍", "Database Search"),
        "manager": ("💾", "Database Manager"),
        "about": ("ℹ️", "How It Works"),
    }
    
    if 'page' not in st.session_state:
        st.session_state.page = "compare"
    
    for key, (icon, label) in pages.items():
        is_active = st.session_state.page == key
        if st.button(
            f"{icon}  {label}",
            key=f"nav_{key}",
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            st.session_state.page = key
            st.rerun()
    
    # Footer stats (minimal)
    st.divider()
    stats = db.get_stats()
    langs = " • ".join([f"{l}: {c}" for l, c in stats.get('by_language', {}).items()])
    st.caption(f"📚 {stats['total_hashes']} refs | {langs}")
    
    page = st.session_state.page


# ============================================================================
# COMPARE FILES PAGE
# ============================================================================
if page == "compare":
    st.header("🆚 Compare Two Files")
    st.caption("Upload two code files to check for similarity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("File 1")
        file1 = st.file_uploader("Upload first file", type=['py', 'java', 'cpp', 'cc', 'c', 'h', 'hpp'], key="file1")
        if file1:
            code1 = file1.read().decode('utf-8')
            lang1 = infer_language(file1.name)
            st.success(f"📄 {file1.name} [{lang1}]")
            with st.expander("View Code", expanded=False):
                st.code(code1[:3000] + ('...' if len(code1) > 3000 else ''), language=lang1)
    
    with col2:
        st.subheader("File 2")
        file2 = st.file_uploader("Upload second file", type=['py', 'java', 'cpp', 'cc', 'c', 'h', 'hpp'], key="file2")
        if file2:
            code2 = file2.read().decode('utf-8')
            lang2 = infer_language(file2.name)
            st.success(f"📄 {file2.name} [{lang2}]")
            with st.expander("View Code", expanded=False):
                st.code(code2[:3000] + ('...' if len(code2) > 3000 else ''), language=lang2)
    
    if file1 and file2:
        if st.button("🔍 Analyze Similarity", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                result = hasher.compare(code1, lang1, code2, lang2)
            
            st.divider()
            
            # Main result
            col_result, col_gauge = st.columns([1, 1])
            
            with col_result:
                st.subheader("Analysis Result")
                
                # Confidence badge
                conf_emoji = get_confidence_color(result.confidence)
                if result.plagiarism_detected:
                    st.error(f"{conf_emoji} **PLAGIARISM LIKELY** - Confidence: {result.confidence.upper()}")
                else:
                    st.success(f"{conf_emoji} **No significant plagiarism** - Confidence: {result.confidence.upper()}")
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Syntactic Similarity", f"{result.syntactic_similarity:.1%}", 
                         help="Hash-based comparison - captures variable naming patterns, formatting")
                m2.metric("Structural Similarity", f"{result.structural_similarity:.1%}",
                         help="Control flow comparison - captures algorithm structure (loops, conditions)")
                m3.metric("Hamming Distance", f"{result.hamming_distance}/256",
                         help="Number of differing bits in the hash (lower = more similar)")
            
            with col_gauge:
                # Show the HIGHER of the two similarities
                main_sim = max(result.syntactic_similarity, result.structural_similarity)
                fig = create_similarity_gauge(main_sim, "Best Match")
                st.plotly_chart(fig, use_container_width=True)
            
            # Explanation
            with st.expander("📊 Understanding the Results", expanded=True):
                st.markdown(f"""
                **How similarity is calculated:**
                
                | Metric | Value | Meaning |
                |--------|-------|---------|
                | Syntactic | {result.syntactic_similarity:.1%} | Code structure hash comparison (catches renamed variables) |
                | Structural | {result.structural_similarity:.1%} | Algorithm pattern comparison (catches cross-language plagiarism) |
                | **Final Score** | **{result.similarity:.1%}** | Maximum of both methods |
                
                **Method used:** `{result.method_used}` gave the higher score
                
                **Why two methods?**
                - **Syntactic** works best for same-language, copy-paste plagiarism
                - **Structural** works best for cross-language or rewritten plagiarism
                
                **Confidence levels:**
                - 🔴 **HIGH**: ≥60% structural match - very likely plagiarism
                - 🟡 **MEDIUM**: 40-60% match - needs review
                - 🟢 **LOW**: <40% match - probably original
                """)
            
            # Matching patterns
            if result.matching_patterns:
                with st.expander(f"🔗 Matching Patterns ({result.pattern_match_ratio})"):
                    for pattern in result.matching_patterns:
                        st.code(pattern)
            
            # Debug info
            if result.debug_info:
                with st.expander("🔬 Debug: Extracted Patterns"):
                    dcol1, dcol2 = st.columns(2)
                    with dcol1:
                        st.caption(f"**{file1.name} ({lang1})**")
                        st.code('\n'.join(result.debug_info.get('patterns1', [])) or 'No patterns')
                    with dcol2:
                        st.caption(f"**{file2.name} ({lang2})**")
                        st.code('\n'.join(result.debug_info.get('patterns2', [])) or 'No patterns')

# ============================================================================
# BATCH ANALYSIS PAGE
# ============================================================================
elif page == "batch":
    st.header("📊 Batch Analysis")
    st.caption("Upload multiple files to find all similar pairs (e.g., check student submissions)")
    
    files = st.file_uploader(
        "Upload multiple files",
        type=['py', 'java', 'cpp', 'cc', 'c', 'h', 'hpp'],
        accept_multiple_files=True
    )
    
    if files and len(files) >= 2:
        st.info(f"📁 {len(files)} files uploaded - will compare {len(files) * (len(files)-1) // 2} pairs")
        
        threshold = st.slider("Minimum similarity to report", 0.0, 1.0, 0.5, 0.05, format="%.0f%%")
        
        if st.button("🔍 Analyze All Pairs", type="primary"):
            # Load all files
            file_data = []
            for f in files:
                code = f.read().decode('utf-8')
                lang = infer_language(f.name)
                file_data.append({'name': f.name, 'code': code, 'lang': lang})
            
            # Compare all pairs
            results = []
            progress = st.progress(0)
            total_pairs = len(file_data) * (len(file_data) - 1) // 2
            pair_count = 0
            
            for i in range(len(file_data)):
                for j in range(i + 1, len(file_data)):
                    result = hasher.compare(
                        file_data[i]['code'], file_data[i]['lang'],
                        file_data[j]['code'], file_data[j]['lang']
                    )
                    
                    if result.similarity >= threshold:
                        results.append({
                            'File 1': file_data[i]['name'],
                            'File 2': file_data[j]['name'],
                            'Similarity': result.similarity,
                            'Syntactic': result.syntactic_similarity,
                            'Structural': result.structural_similarity,
                            'Confidence': result.confidence.upper()
                        })
                    
                    pair_count += 1
                    progress.progress(pair_count / total_pairs)
            
            progress.empty()
            
            if results:
                st.warning(f"⚠️ Found {len(results)} suspicious pairs above {threshold:.0%} threshold")
                
                # Sort by similarity
                df = pd.DataFrame(results)
                df = df.sort_values('Similarity', ascending=False)
                df['Similarity'] = df['Similarity'].apply(lambda x: f"{x:.1%}")
                df['Syntactic'] = df['Syntactic'].apply(lambda x: f"{x:.1%}")
                df['Structural'] = df['Structural'].apply(lambda x: f"{x:.1%}")
                
                # Color code by confidence
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Confidence': st.column_config.TextColumn(
                            help="HIGH = likely plagiarism, MEDIUM = review needed, LOW = probably ok"
                        )
                    }
                )
                
                # Heatmap
                if len(file_data) <= 20:
                    st.subheader("Similarity Matrix")
                    
                    # Build matrix
                    n = len(file_data)
                    matrix = [[0.0] * n for _ in range(n)]
                    names = [f['name'][:20] for f in file_data]
                    
                    for i in range(n):
                        matrix[i][i] = 1.0
                        for j in range(i + 1, n):
                            r = hasher.compare(
                                file_data[i]['code'], file_data[i]['lang'],
                                file_data[j]['code'], file_data[j]['lang']
                            )
                            matrix[i][j] = r.similarity
                            matrix[j][i] = r.similarity
                    
                    fig = px.imshow(
                        matrix,
                        x=names,
                        y=names,
                        color_continuous_scale='RdYlGn_r',
                        zmin=0, zmax=1
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success(f"✅ No pairs found above {threshold:.0%} threshold - all files appear original!")
    
    elif files:
        st.warning("Please upload at least 2 files")


# ============================================================================
# DATABASE SEARCH PAGE
# ============================================================================
elif page == "search":
    st.header("🔍 Database Search")
    st.caption("Search and explore the reference code database")
    
    tab1, tab2 = st.tabs(["🔎 Check Against Database", "📚 Browse Database"])
    
    with tab1:
        st.subheader("Check Your Code")
        st.caption("Compare your code against all reference implementations")
        
        check_file = st.file_uploader(
            "Upload file to check",
            type=['py', 'java', 'cpp', 'cc', 'c', 'h', 'hpp'],
            key="check_file"
        )
        
        threshold = st.slider("Minimum similarity", 0.0, 1.0, 0.4, 0.05, key="check_thresh")
        
        search_method = st.radio(
            "Search method",
            ["Both (recommended)", "Syntactic only", "Structural only"],
            horizontal=True
        )
        
        if check_file:
            code = check_file.read().decode('utf-8')
            lang = infer_language(check_file.name)
            
            st.info(f"📄 {check_file.name} [{lang}]")
            
            with st.expander("🔬 View Extracted Patterns"):
                patterns = hasher.debug_patterns(code, lang)
                st.json(patterns)
            
            if st.button("🔍 Search Database", type="primary"):
                with st.spinner("Searching..."):
                    query_hash = hasher.hash_code(code, lang)
                    query_patterns = hasher.debug_patterns(code, lang)['control_flow']
                    
                    if search_method == "Both (recommended)":
                        matches = db.find_similar(query_hash, query_patterns, threshold=threshold)
                    elif search_method == "Syntactic only":
                        matches = db.find_similar_hash(query_hash, threshold=threshold)
                    else:
                        matches = db.find_similar_structural(query_patterns, threshold=threshold)
                
                if matches:
                    st.warning(f"⚠️ Found {len(matches)} potential matches!")
                    
                    for match in matches[:10]:
                        with st.expander(
                            f"**{match['file_path']}** - {match['similarity']:.1%} ({match.get('match_type', 'unknown')})"
                        ):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.caption("Match Info")
                                st.write(f"**Source:** {match['source']}")
                                st.write(f"**Language:** {match['language']}")
                                st.write(f"**Similarity:** {match['similarity']:.1%}")
                                if 'syntactic_similarity' in match:
                                    st.write(f"**Syntactic:** {match['syntactic_similarity']:.1%}")
                                if 'structural_similarity' in match:
                                    st.write(f"**Structural:** {match['structural_similarity']:.1%}")
                                if match.get('matching_patterns'):
                                    st.write("**Matching patterns:**")
                                    st.code('\n'.join(match['matching_patterns']))
                            with col2:
                                st.caption("Reference Code")
                                st.code(match['code'][:1000] + ('...' if len(match['code']) > 1000 else ''), 
                                       language=match['language'])
                else:
                    st.success("✅ No matches found - code appears original!")
    
    with tab2:
        st.subheader("Browse Reference Database")
        
        stats = db.get_stats()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Files", stats['total_hashes'])
        col2.metric("Languages", len(stats.get('by_language', {})))
        col3.metric("Sources", len(stats.get('by_source', {})))
        
        # Filters
        st.divider()
        
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            lang_filter = st.selectbox(
                "Filter by Language",
                ["All"] + list(stats.get('by_language', {}).keys())
            )
        with fcol2:
            source_filter = st.selectbox(
                "Filter by Source",
                ["All"] + list(stats.get('by_source', {}).keys())
            )
        with fcol3:
            search_term = st.text_input("Search file name", placeholder="e.g., bubble")
        
        # Get all entries
        cursor = db.conn.cursor()
        query = "SELECT id, source, file_path, language, code, patterns, created_at FROM code_hashes"
        conditions = []
        params = []
        
        if lang_filter != "All":
            conditions.append("language = ?")
            params.append(lang_filter)
        if source_filter != "All":
            conditions.append("source = ?")
            params.append(source_filter)
        if search_term:
            conditions.append("file_path LIKE ?")
            params.append(f"%{search_term}%")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC LIMIT 50"
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        
        if rows:
            st.caption(f"Showing {len(rows)} files")
            
            for row in rows:
                record_id, source, file_path, language, code, patterns_str, created_at = row
                patterns = patterns_str.split('|') if patterns_str else []
                
                with st.expander(f"📄 **{file_path}** [{language}] - {source}"):
                    tcol1, tcol2 = st.columns([2, 1])
                    
                    with tcol1:
                        st.caption("Source Code")
                        st.code(code[:2000] + ('...' if len(code) > 2000 else ''), language=language)
                    
                    with tcol2:
                        st.caption("Info")
                        st.write(f"**Source:** {source}")
                        st.write(f"**Language:** {language}")
                        st.write(f"**Size:** {len(code)} chars")
                        st.write(f"**Added:** {created_at}")
                        
                        st.caption("Control Flow Patterns")
                        if patterns:
                            st.code('\n'.join(patterns))
                        else:
                            st.write("No patterns")
        else:
            st.info("No entries found")
        
        # Visualizations
        st.divider()
        st.subheader("📊 Database Statistics")
        
        vcol1, vcol2 = st.columns(2)
        
        with vcol1:
            if stats.get('by_language'):
                fig = px.pie(
                    values=list(stats['by_language'].values()),
                    names=list(stats['by_language'].keys()),
                    title="By Language"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with vcol2:
            if stats.get('by_source'):
                fig = px.bar(
                    x=list(stats['by_source'].keys()),
                    y=list(stats['by_source'].values()),
                    title="By Source"
                )
                st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# DATABASE MANAGER PAGE
# ============================================================================
elif page == "manager":
    st.header("💾 Database Manager")
    
    tab1, tab2, tab3 = st.tabs(["➕ Add Files", "🔄 Reload Code Bank", "🗑️ Clear Database"])
    
    with tab1:
        st.subheader("Add Reference File")
        
        source = st.text_input("Source identifier", placeholder="e.g., leetcode, homework1, official")
        add_file = st.file_uploader(
            "Upload file",
            type=['py', 'java', 'cpp', 'cc', 'c', 'h', 'hpp'],
            key="add_file"
        )
        
        if add_file and source:
            code = add_file.read().decode('utf-8')
            lang = infer_language(add_file.name)
            
            st.code(code[:500] + ('...' if len(code) > 500 else ''), language=lang)

            patterns_info = hasher.debug_patterns(code, lang)
            with st.expander("🔬 Patterns to be indexed"):
                st.caption("Control flow patterns:")
                st.code('\n'.join(patterns_info['control_flow']) or 'No patterns detected')
            
            if st.button("💾 Add to Database", type="primary"):
                try:
                    hash_value = hasher.hash_code(code, lang)
                    patterns = patterns_info['control_flow']
                    
                    db.add_hash(
                        source=source,
                        file_path=add_file.name,
                        language=lang,
                        code=code,
                        hash_value=hash_value,
                        patterns=patterns,
                        metadata={'added_via': 'streamlit'}
                    )
                    st.success(f"✅ Added {add_file.name} to database!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab2:
        st.subheader("Reload Code Bank")
        st.caption("Reload all files from the code_bank folder")
        
        code_bank_path = Path(__file__).parent / "code_bank"
        
        if code_bank_path.exists():
            st.info(f"📁 Code bank path: `{code_bank_path}`")
            
            files_found = list(code_bank_path.rglob('*'))
            code_files = [f for f in files_found if f.suffix in ['.py', '.java', '.cpp', '.cc', '.c', '.h', '.hpp']]
            st.text(f"Found {len(code_files)} code files")
            
            if st.button("🔄 Reload All", type="primary"):
                with st.spinner("Reloading..."):
                    loaded = load_code_bank(str(code_bank_path), hasher, db, force_reload=True)
                st.success(f"✅ Reloaded {loaded} files!")
                st.rerun()
        else:
            st.warning(f"Code bank folder not found: `{code_bank_path}`")
    
    with tab3:
        st.subheader("Clear Database")
        st.warning("⚠️ This will delete ALL reference hashes from the database!")
        
        confirm = st.checkbox("I understand this cannot be undone")
        
        if st.button("🗑️ Clear All", type="secondary", disabled=not confirm):
            db.clear()
            st.success("✅ Database cleared!")
            st.rerun()


# ============================================================================
# HOW IT WORKS PAGE
# ============================================================================
elif page == "about":
    st.header("ℹ️ How It Works")
    
    st.markdown("""
    ## 🎯 Overview
    
    This tool detects code plagiarism using **AST (Abstract Syntax Tree) analysis** 
    and **perceptual hashing**. It can detect plagiarism even when:
    
    - Variables are renamed
    - Comments are changed
    - Code is reformatted
    - Code is translated to another language (Python ↔ Java ↔ C++)
    
    ---
    
    ## 🔬 Two Detection Methods
    
    ### 1. Syntactic Similarity (Hash-based)
    
    Generates a 256-bit "fingerprint" of the code structure:
    
    ```
    Code → AST → Extract Features → Shingles → LSH Hash → Hamming Distance
    ```
    
    **Best for:** Same-language copy-paste with renamed variables
    
    ### 2. Structural Similarity (Pattern-based)
    
    Compares control flow patterns across languages:
    
    ```
    Code → AST → Extract LOOP/COND patterns → Normalize depths → Jaccard similarity
    ```
    
    **Best for:** Cross-language plagiarism, algorithm detection
    
    ---
    
    ## 📊 Similarity Scores Explained
    
    | Score | Meaning |
    |-------|---------|
    | **Syntactic** | How similar the code "looks" (structure, patterns) |
    | **Structural** | How similar the algorithms are (control flow) |
    | **Final** | Maximum of both (best evidence of plagiarism) |
    
    ---
    
    ## 🚦 Confidence Levels
    
    | Level | Threshold | Meaning |
    |-------|-----------|---------|
    | 🔴 **HIGH** | ≥60% structural | Very likely plagiarism |
    | 🟡 **MEDIUM** | 40-60% | Needs manual review |
    | 🟢 **LOW** | <40% | Probably original |
    
    ---
    
    ## ⚠️ Limitations
    
    - Similar algorithms (like tree traversal) may show high structural similarity 
      even if independently written
    - Very short code snippets may give unreliable results
    - The tool detects similarity, not intent - manual review is still needed
    
    ---
    
    ## 📁 Supported Languages
    
    - **Python** (.py)
    - **Java** (.java)  
    - **C/C++** (.cpp, .cc, .c, .h, .hpp)
    """)


# Footer
st.divider()
st.caption("Code Plagiarism Detector • Built with Streamlit • AST-based analysis using tree-sitter")
