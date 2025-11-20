"""Streamlit web interface for Code Plagiarism Detector."""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from code_plagiarism_detector import CodeHasher, CodeHashDatabase


# Page config
st.set_page_config(
    page_title="Code Plagiarism Detector",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'hasher' not in st.session_state:
    st.session_state.hasher = CodeHasher()
if 'db_path' not in st.session_state:
    st.session_state.db_path = "web_reference.db"


def main():
    """Main application."""
    st.title("🔍 Code Plagiarism Detector")
    st.markdown("*AST-based plagiarism detection with perceptual hashing*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a feature:",
        ["🆚 Compare Files", "📊 Batch Analysis", "💾 Database Manager", "ℹ️ About"]
    )
    
    if page == "🆚 Compare Files":
        compare_files_page()
    elif page == "📊 Batch Analysis":
        batch_analysis_page()
    elif page == "💾 Database Manager":
        database_manager_page()
    else:
        about_page()


def compare_files_page():
    """Page for comparing two code files."""
    st.header("Compare Two Files")
    st.markdown("Upload two code files to check their similarity.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("File 1")
        file1 = st.file_uploader("Upload first file", type=['py', 'java', 'cpp', 'c', 'h', 'hpp'], key="file1")
        lang1 = st.selectbox("Language", ['python', 'java', 'cpp'], key="lang1")
        
        if file1:
            code1 = file1.read().decode('utf-8')
            st.code(code1, language=lang1)
    
    with col2:
        st.subheader("File 2")
        file2 = st.file_uploader("Upload second file", type=['py', 'java', 'cpp', 'c', 'h', 'hpp'], key="file2")
        lang2 = st.selectbox("Language", ['python', 'java', 'cpp'], key="lang2")
        
        if file2:
            code2 = file2.read().decode('utf-8')
            st.code(code2, language=lang2)
    
    if st.button("🔍 Compare", type="primary"):
        if file1 and file2:
            with st.spinner("Analyzing code..."):
                try:
                    hasher = st.session_state.hasher
                    
                    # Hash both files
                    hash1 = hasher.hash_code(code1, lang1)
                    hash2 = hasher.hash_code(code2, lang2)
                    
                    # Compare
                    similarity, hamming_dist = hasher.compare(hash1, hash2)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Results")
                    
                    # Similarity gauge
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Similarity", f"{similarity:.2%}")
                    with col2:
                        st.metric("Hamming Distance", f"{hamming_dist}/256")
                    with col3:
                        if similarity > 0.95:
                            st.error("⚠️ VERY HIGH RISK")
                        elif similarity > 0.85:
                            st.warning("⚠️ HIGH RISK")
                        elif similarity > 0.70:
                            st.info("ℹ️ MODERATE RISK")
                        else:
                            st.success("✅ LOW RISK")
                    
                    # Similarity gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=similarity * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Similarity Score"},
                        delta={'reference': 85},
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Assessment
                    st.markdown("### Assessment")
                    if similarity > 0.95:
                        st.error("**Nearly identical code detected.** This is very likely plagiarism or the same solution.")
                    elif similarity > 0.85:
                        st.warning("**High structural similarity.** Possible plagiarism - manual review recommended.")
                    elif similarity > 0.70:
                        st.info("**Similar structure detected.** Could be similar approaches to the same problem.")
                    else:
                        st.success("**Different code.** Low risk of plagiarism.")
                    
                except Exception as e:
                    st.error(f"Error analyzing code: {e}")
        else:
            st.warning("Please upload both files.")


def batch_analysis_page():
    """Page for batch analysis of multiple files."""
    st.header("Batch Analysis")
    st.markdown("Upload multiple files to compare all pairs.")
    
    # File uploader for multiple files
    files = st.file_uploader(
        "Upload code files",
        type=['py', 'java', 'cpp', 'c', 'h', 'hpp'],
        accept_multiple_files=True
    )
    
    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05,
        help="Only show matches above this threshold"
    )
    
    if files and st.button("🔍 Analyze All", type="primary"):
        if len(files) < 2:
            st.warning("Please upload at least 2 files.")
            return
        
        with st.spinner(f"Analyzing {len(files)} files..."):
            try:
                hasher = st.session_state.hasher
                
                # Hash all files
                hashes = {}
                file_contents = {}
                
                for file in files:
                    content = file.read().decode('utf-8')
                    file_contents[file.name] = content
                    
                    # Detect language from extension
                    ext = Path(file.name).suffix.lower()
                    if ext == '.py':
                        lang = 'python'
                    elif ext == '.java':
                        lang = 'java'
                    else:
                        lang = 'cpp'
                    
                    hashes[file.name] = hasher.hash_code(content, lang)
                
                # Compare all pairs
                results = []
                filenames = list(hashes.keys())
                
                for i in range(len(filenames)):
                    for j in range(i + 1, len(filenames)):
                        file1, file2 = filenames[i], filenames[j]
                        similarity, hamming_dist = hasher.compare(hashes[file1], hashes[file2])
                        
                        if similarity >= threshold:
                            results.append({
                                'File 1': file1,
                                'File 2': file2,
                                'Similarity': similarity,
                                'Hamming Distance': hamming_dist,
                                'Risk': get_risk_level(similarity)
                            })
                
                # Display results
                st.markdown("---")
                st.subheader("Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Files", len(files))
                with col2:
                    st.metric("Flagged Pairs", len(results))
                
                if results:
                    # Results table
                    df = pd.DataFrame(results)
                    df['Similarity'] = df['Similarity'].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Similarity matrix heatmap
                    st.markdown("### Similarity Matrix")
                    
                    # Create similarity matrix
                    n = len(filenames)
                    matrix = [[0.0] * n for _ in range(n)]
                    
                    for i in range(n):
                        for j in range(n):
                            if i == j:
                                matrix[i][j] = 1.0
                            else:
                                sim, _ = hasher.compare(hashes[filenames[i]], hashes[filenames[j]])
                                matrix[i][j] = sim
                    
                    fig = px.imshow(
                        matrix,
                        labels=dict(x="File", y="File", color="Similarity"),
                        x=filenames,
                        y=filenames,
                        color_continuous_scale="RdYlGn_r",
                        aspect="auto"
                    )
                    
                    fig.update_xaxes(side="bottom")
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.success("✅ No suspicious similarities detected above threshold.")
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")


def database_manager_page():
    """Page for managing reference database."""
    st.header("Database Manager")
    
    tab1, tab2, tab3 = st.tabs(["📥 Add to Database", "🔍 Check Against Database", "📊 Database Stats"])
    
    with tab1:
        st.subheader("Add Reference Code")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            source = st.text_input("Source Identifier", placeholder="e.g., leetcode, official, github:user/repo")
        
        with col2:
            language = st.selectbox("Language", ['python', 'java', 'cpp'])
        
        file = st.file_uploader("Upload reference code", type=['py', 'java', 'cpp', 'c', 'h', 'hpp'])
        
        metadata_json = st.text_area(
            "Metadata (optional JSON)",
            placeholder='{"problem": "Two Sum", "difficulty": "easy"}',
            height=100
        )
        
        if st.button("💾 Add to Database", type="primary"):
            if file and source:
                try:
                    import json
                    
                    hasher = st.session_state.hasher
                    db = CodeHashDatabase(st.session_state.db_path)
                    
                    # Read and hash code
                    code = file.read().decode('utf-8')
                    hash_value = hasher.hash_code(code, language)
                    
                    # Parse metadata
                    metadata = None
                    if metadata_json.strip():
                        metadata = json.loads(metadata_json)
                    
                    # Add to database
                    hash_id = db.add_hash(
                        source=source,
                        file_path=file.name,
                        language=language,
                        hash_value=hash_value,
                        metadata=metadata
                    )
                    
                    db.close()
                    
                    st.success(f"✅ Added to database (ID: {hash_id})")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please provide source and upload a file.")
    
    with tab2:
        st.subheader("Check Against Database")
        
        check_file = st.file_uploader("Upload file to check", type=['py', 'java', 'cpp', 'c', 'h', 'hpp'], key="check_file")
        check_lang = st.selectbox("Language", ['python', 'java', 'cpp'], key="check_lang")
        check_threshold = st.slider("Threshold", 0.0, 1.0, 0.85, 0.05, key="check_threshold")
        
        if st.button("🔍 Check", type="primary"):
            if check_file:
                try:
                    hasher = st.session_state.hasher
                    db = CodeHashDatabase(st.session_state.db_path)
                    
                    # Hash the code
                    code = check_file.read().decode('utf-8')
                    query_hash = hasher.hash_code(code, check_lang)
                    
                    # Search database
                    matches = db.find_similar(query_hash, threshold=check_threshold)
                    
                    db.close()
                    
                    st.markdown("---")
                    
                    if matches:
                        st.warning(f"⚠️ Found {len(matches)} potential matches")
                        
                        for i, match in enumerate(matches, 1):
                            with st.expander(f"Match #{i}: {match['file_path']} ({match['similarity']:.1%})"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Similarity", f"{match['similarity']:.2%}")
                                with col2:
                                    st.metric("Source", match['source'])
                                with col3:
                                    st.metric("Language", match['language'])
                                
                                st.markdown(f"**Hamming Distance:** {match['hamming_distance']}/256")
                                
                                if match['metadata']:
                                    st.json(match['metadata'])
                                
                                if match['similarity'] > 0.95:
                                    st.error("⚠️ VERY HIGH RISK - Likely copied")
                                elif match['similarity'] > 0.85:
                                    st.warning("⚠️ HIGH RISK - Possible plagiarism")
                                else:
                                    st.info("ℹ️ Similar structure detected")
                    else:
                        st.success("✅ No matches found - appears to be original code")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please upload a file to check.")
    
    with tab3:
        st.subheader("Database Statistics")
        
        if st.button("🔄 Refresh Stats"):
            try:
                db = CodeHashDatabase(st.session_state.db_path)
                stats = db.get_stats()
                db.close()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Hashes", stats['total_hashes'])
                    st.metric("Total Collections", stats['total_collections'])
                
                with col2:
                    if stats['by_language']:
                        st.markdown("**By Language:**")
                        for lang, count in stats['by_language'].items():
                            st.write(f"- {lang}: {count}")
                    
                    if stats['by_source']:
                        st.markdown("**By Source:**")
                        for source, count in stats['by_source'].items():
                            st.write(f"- {source}: {count}")
                
                # Language distribution chart
                if stats['by_language']:
                    fig = px.pie(
                        values=list(stats['by_language'].values()),
                        names=list(stats['by_language'].keys()),
                        title="Language Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")


def about_page():
    """About page."""
    st.header("About")
    
    st.markdown("""
    ### Code Plagiarism Detector
    
    A lightweight plagiarism detection tool using AST fingerprinting and perceptual hashing.
    
    #### Features
    - **AST-based Analysis**: Parses code structure using tree-sitter
    - **Perceptual Hashing**: Generates 256-bit hashes using LSH
    - **Multi-language**: Supports Python, Java, C++
    - **Database**: Store reference implementations for comparison
    - **Fast**: Efficient similarity detection using Hamming distance
    
    #### How It Works
    1. Parse code into Abstract Syntax Tree (AST)
    2. Extract structural features (functions, control flow, etc.)
    3. Generate k-shingles from features
    4. Create 256-bit perceptual hash using LSH
    5. Compare using Hamming distance
    
    #### Similarity Thresholds
    - **95-100%**: Very high risk - likely plagiarism
    - **85-95%**: High risk - possible plagiarism
    - **70-85%**: Moderate risk - similar structure
    - **0-70%**: Low risk - different code
    
    #### Use Cases
    - Academic integrity checking
    - Code review and duplicate detection
    - License compliance
    - Refactoring opportunities
    
    #### Technology Stack
    - **Backend**: Python, tree-sitter, numpy
    - **Frontend**: Streamlit
    - **Database**: SQLite
    - **Visualization**: Plotly
    
    ---
    
    Made with ❤️ using Streamlit
    """)


def get_risk_level(similarity):
    """Get risk level from similarity score."""
    if similarity > 0.95:
        return "🔴 VERY HIGH"
    elif similarity > 0.85:
        return "🟠 HIGH"
    elif similarity > 0.70:
        return "🟡 MODERATE"
    else:
        return "🟢 LOW"


if __name__ == '__main__':
    main()
