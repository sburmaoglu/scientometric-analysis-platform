"""Causal Analysis Page - Granger Causality and Causal Inference"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def render():
    """Render causal analysis page"""
    
    st.title("🔗 Causal Analysis")
    st.markdown("Discover causal relationships and temporal dependencies in scientometric data")
    
    if st.session_state.publications_data is None and st.session_state.patents_data is None:
        st.warning("⚠️ Please upload data first")
        return
    
    st.markdown("---")
    
    # Analysis selection
    analysis_type = st.selectbox(
        "Select Causal Analysis Method",
        [
            "🔄 Granger Causality Test",
            "📊 Cross-Correlation Analysis",
            "⏱️ Time Series Causality",
            "🌐 Transfer Entropy",
            "📈 Convergent Cross Mapping (CCM)",
            "🎯 Propensity Score Matching"
        ]
    )
    
    st.markdown("---")
    
    if analysis_type == "🔄 Granger Causality Test":
        render_granger_causality()
    
    elif analysis_type == "📊 Cross-Correlation Analysis":
        render_cross_correlation()
    
    elif analysis_type == "⏱️ Time Series Causality":
        render_time_series_causality()
    
    elif analysis_type == "🌐 Transfer Entropy":
        render_transfer_entropy()
    
    elif analysis_type == "📈 Convergent Cross Mapping (CCM)":
        render_ccm_analysis()
    
    elif analysis_type == "🎯 Propensity Score Matching":
        render_propensity_score()

def prepare_time_series_data(df, metric_col, dataset_type):
    """Prepare time series data for causal analysis"""
    
    if 'year' not in df.columns:
        return None
    
    # Group by year
    if metric_col == 'count':
        ts_data = df.groupby('year').size().reset_index(name='value')
    elif metric_col in df.columns:
        ts_data = df.groupby('year')[metric_col].sum().reset_index(name='value')
    else:
        return None
    
    # Ensure continuous years
    year_range = range(int(ts_data['year'].min()), int(ts_data['year'].max()) + 1)
    ts_data = ts_data.set_index('year').reindex(year_range, fill_value=0).reset_index()
    ts_data.columns = ['year', 'value']
    
    return ts_data

def render_granger_causality():
    """Granger causality test"""
    
    st.subheader("🔄 Granger Causality Test")
    st.markdown("""
    **Granger Causality** tests whether one time series can predict another.
    
    **Interpretation:**
    - X "Granger-causes" Y if past values of X help predict Y
    - Does not imply true causation, only predictive utility
    - Requires stationary time series
    
    **Null Hypothesis:** X does NOT Granger-cause Y
    """)
    
    # Need both datasets
    if st.session_state.publications_data is None or st.session_state.patents_data is None:
        st.warning("⚠️ Both publications and patents data required for cross-domain analysis")
        
        # Allow single dataset analysis
        st.info("Or analyze causality within a single dataset using different metrics")
        
        dataset = st.radio("Select Dataset", ["Publications", "Patents"], horizontal=True)
        
        if dataset == "Publications":
            if st.session_state.publications_data is None:
                return
            df = st.session_state.publications_data
            dataset_type = 'publications'
        else:
            if st.session_state.patents_data is None:
                return
            df = st.session_state.patents_data
            dataset_type = 'patents'
        
        # Single dataset mode
        render_single_dataset_granger(df, dataset_type)
        return
    
    # Two dataset mode
    pubs_df = st.session_state.publications_data
    pats_df = st.session_state.patents_data
    
    st.markdown("---")
    
    # Variable selection
    st.markdown("### 📊 Select Variables for Causality Test")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Variable X (Cause)**")
        x_dataset = st.selectbox("X Dataset", ["Publications", "Patents"], key='x_dataset')
        
        if x_dataset == "Publications":
            x_metrics = ["Publication Count", "Total Citations", "Unique Authors", "Unique Journals"]
        else:
            x_metrics = ["Patent Count", "Total Citations", "Unique Inventors", "Unique Organizations"]
        
        x_metric = st.selectbox("X Metric", x_metrics, key='x_metric')
    
    with col2:
        st.markdown("**Variable Y (Effect)**")
        y_dataset = st.selectbox("Y Dataset", ["Publications", "Patents"], key='y_dataset')
        
        if y_dataset == "Publications":
            y_metrics = ["Publication Count", "Total Citations", "Unique Authors", "Unique Journals"]
        else:
            y_metrics = ["Patent Count", "Total Citations", "Unique Inventors", "Unique Organizations"]
        
        y_metric = st.selectbox("Y Metric", y_metrics, key='y_metric')
    
    st.markdown("---")
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        max_lag = st.slider("Maximum Lag (years)", 1, 10, 5,
                           help="Test lags from 1 to this value")
    
    with col2:
        significance_level = st.selectbox("Significance Level", [0.01, 0.05, 0.10], index=1)
    
    if st.button("🚀 Run Granger Causality Test", type="primary"):
        run_granger_test(pubs_df, pats_df, x_dataset, x_metric, y_dataset, y_metric, 
                        max_lag, significance_level)

def run_granger_test(pubs_df, pats_df, x_dataset, x_metric, y_dataset, y_metric, 
                     max_lag, significance_level):
    """Execute Granger causality test"""
    
    with st.spinner("Running Granger causality test..."):
        # Prepare X time series
        x_df = pubs_df if x_dataset == "Publications" else pats_df
        x_ts = prepare_metric_timeseries(x_df, x_metric, x_dataset)
        
        # Prepare Y time series
        y_df = pubs_df if y_dataset == "Publications" else pats_df
        y_ts = prepare_metric_timeseries(y_df, y_metric, y_dataset)
        
        if x_ts is None or y_ts is None:
            st.error("Could not prepare time series data")
            return
        
        # Merge on common years
        merged = pd.merge(x_ts, y_ts, on='year', suffixes=('_x', '_y'))
        
        if len(merged) < max_lag + 5:
            st.error(f"Insufficient data points. Need at least {max_lag + 5} years.")
            return
        
        st.success(f"✅ Prepared time series with {len(merged)} years")
        
        # Visualize time series
        st.markdown("### 📈 Time Series Visualization")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=merged['year'],
            y=merged['value_x'],
            mode='lines+markers',
            name=f'{x_dataset}: {x_metric}',
            line=dict(color='#3498db', width=2),
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=merged['year'],
            y=merged['value_y'],
            mode='lines+markers',
            name=f'{y_dataset}: {y_metric}',
            line=dict(color='#e74c3c', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Time Series Comparison",
            xaxis_title="Year",
            yaxis=dict(title=f'{x_dataset}: {x_metric}', side='left'),
            yaxis2=dict(title=f'{y_dataset}: {y_metric}', side='right', overlaying='y'),
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Run Granger causality test
        st.markdown("### 🔬 Granger Causality Test Results")
        
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            # Prepare data for Granger test
            test_data = merged[['value_y', 'value_x']].values
            
            # Run test for multiple lags
            results = grangercausalitytests(test_data, max_lag, verbose=False)
            
            # Extract results
            test_results = []
            
            for lag in range(1, max_lag + 1):
                # Get F-test result
                f_test = results[lag][0]['ssr_ftest']
                f_stat = f_test[0]
                p_value = f_test[1]
                
                # Get chi-square test result
                chi2_test = results[lag][0]['ssr_chi2test']
                chi2_stat = chi2_test[0]
                chi2_p = chi2_test[1]
                
                test_results.append({
                    'Lag': lag,
                    'F-Statistic': f_stat,
                    'F-Test p-value': p_value,
                    'χ² Statistic': chi2_stat,
                    'χ² p-value': chi2_p,
                    'Significant (α=0.05)': '✅ Yes' if p_value < significance_level else '❌ No'
                })
            
            results_df = pd.DataFrame(test_results)
            
            # Display results table
            st.dataframe(
                results_df.style.format({
                    'F-Statistic': '{:.4f}',
                    'F-Test p-value': '{:.4f}',
                    'χ² Statistic': '{:.4f}',
                    'χ² p-value': '{:.4f}'
                }).apply(
                    lambda x: ['background-color: #d4edda' if v == '✅ Yes' else '' 
                              for v in x], 
                    subset=['Significant (α=0.05)']
                ),
                use_container_width=True,
                hide_index=True
            )
            
            # Visualize p-values
            st.markdown("### 📊 P-Values Across Lags")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=results_df['Lag'],
                y=results_df['F-Test p-value'],
                mode='lines+markers',
                name='F-Test p-value',
                line=dict(width=3, color='#3498db'),
                marker=dict(size=10)
            ))
            
            # Add significance threshold line
            fig.add_hline(
                y=significance_level,
                line_dash="dash",
                line_color="red",
                annotation_text=f"α = {significance_level}",
                annotation_position="right"
            )
            
            fig.update_layout(
                title="Granger Causality Test: P-Values vs Lag",
                xaxis_title="Lag (years)",
                yaxis_title="P-Value",
                yaxis_type="log",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown("---")
            st.markdown("### 🎯 Interpretation")
            
            significant_lags = results_df[results_df['F-Test p-value'] < significance_level]['Lag'].tolist()
            
            if significant_lags:
                st.success(f"""
                **✅ Granger Causality Detected!**
                
                **Finding:** {x_dataset} {x_metric} **Granger-causes** {y_dataset} {y_metric}
                
                **Significant at lags:** {', '.join(map(str, significant_lags))} years
                
                **Interpretation:**
                - Past values of {x_dataset} {x_metric} help predict {y_dataset} {y_metric}
                - The effect appears after {min(significant_lags)} year(s)
                - This suggests a temporal relationship but does NOT prove true causation
                
                **Practical meaning:**
                - Changes in {x_dataset.lower()} activity may lead to changes in {y_dataset.lower()} activity
                - Time lag: approximately {min(significant_lags)}-{max(significant_lags)} years
                """)
            else:
                st.info(f"""
                **❌ No Granger Causality Detected**
                
                **Finding:** {x_dataset} {x_metric} does NOT Granger-cause {y_dataset} {y_metric}
                
                **Interpretation:**
                - Past values of {x_dataset} {x_metric} do not improve prediction of {y_dataset} {y_metric}
                - No evidence of temporal predictive relationship
                - This does not rule out other types of relationships (correlation, bidirectional causality)
                
                **Recommendations:**
                - Try reversing X and Y (test if Y Granger-causes X)
                - Try different metrics or time aggregations
                - Consider longer time lags
                """)
            
            # Bidirectional test suggestion
            st.markdown("---")
            st.markdown("### 🔄 Bidirectional Causality")
            
            if st.button("🔀 Test Reverse Direction (Y → X)"):
                st.info("Testing if Y Granger-causes X...")
                
                # Reverse test
                test_data_reverse = merged[['value_x', 'value_y']].values
                results_reverse = grangercausalitytests(test_data_reverse, max_lag, verbose=False)
                
                reverse_results = []
                for lag in range(1, max_lag + 1):
                    f_test = results_reverse[lag][0]['ssr_ftest']
                    reverse_results.append({
                        'Lag': lag,
                        'p-value': f_test[1],
                        'Significant': '✅ Yes' if f_test[1] < significance_level else '❌ No'
                    })
                
                reverse_df = pd.DataFrame(reverse_results)
                st.dataframe(reverse_df, use_container_width=True, hide_index=True)
                
                reverse_significant = reverse_df[reverse_df['p-value'] < significance_level]
                
                if len(reverse_significant) > 0:
                    st.success(f"✅ **Bidirectional causality detected!** Both directions show Granger causality.")
                else:
                    st.info(f"ℹ️ **Unidirectional causality:** Only X → Y is significant.")
        
        except Exception as e:
            st.error(f"Error running Granger test: {str(e)}")
            st.info("""
            **Common issues:**
            - Need at least 10-15 time points
            - Time series should be stationary
            - Try installing: `pip install statsmodels`
            """)

def prepare_metric_timeseries(df, metric, dataset):
    """Prepare time series for a specific metric"""
    
    if 'year' not in df.columns:
        return None
    
    if metric.endswith('Count'):
        # Count metrics
        ts_data = df.groupby('year').size().reset_index(name='value')
    
    elif 'Citations' in metric:
        # Citation metrics
        citation_col = 'citations' if dataset == 'Publications' else 'forward_citations'
        if citation_col in df.columns:
            ts_data = df.groupby('year')[citation_col].sum().reset_index(name='value')
        else:
            return None
    
    elif 'Authors' in metric or 'Inventors' in metric:
        # Unique entity counts
        entity_col = 'author' if dataset == 'Publications' else 'inventor'
        if entity_col in df.columns:
            yearly_entities = []
            for year in sorted(df['year'].dropna().unique()):
                year_df = df[df['year'] == year]
                all_entities = []
                for entities_str in year_df[entity_col].dropna():
                    entities = str(entities_str).split(';')
                    all_entities.extend([e.strip() for e in entities if e.strip()])
                yearly_entities.append({'year': year, 'value': len(set(all_entities))})
            ts_data = pd.DataFrame(yearly_entities)
        else:
            return None
    
    elif 'Journals' in metric:
        if 'journal' in df.columns:
            ts_data = df.groupby('year')['journal'].nunique().reset_index(name='value')
        else:
            return None
    
    elif 'Organizations' in metric:
        if 'assignee' in df.columns:
            yearly_orgs = []
            for year in sorted(df['year'].dropna().unique()):
                year_df = df[df['year'] == year]
                all_orgs = []
                for orgs_str in year_df['assignee'].dropna():
                    orgs = str(orgs_str).split(';')
                    all_orgs.extend([o.strip() for o in orgs if o.strip()])
                yearly_orgs.append({'year': year, 'value': len(set(all_orgs))})
            ts_data = pd.DataFrame(yearly_orgs)
        else:
            return None
    else:
        return None
    
    # Ensure continuous years
    if len(ts_data) > 0:
        year_range = range(int(ts_data['year'].min()), int(ts_data['year'].max()) + 1)
        ts_data = ts_data.set_index('year').reindex(year_range, fill_value=0).reset_index()
        ts_data.columns = ['year', 'value']
    
    return ts_data

def render_single_dataset_granger(df, dataset_type):
    """Granger causality within single dataset"""
    
    st.markdown("### 📊 Single Dataset Mode")
    st.info("Analyze causal relationships between different metrics within the same dataset")
    
    # Select two metrics
    if dataset_type == 'publications':
        available_metrics = ["Publication Count", "Total Citations", "Unique Authors", "Unique Journals"]
    else:
        available_metrics = ["Patent Count", "Total Citations", "Unique Inventors", "Unique Organizations"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_metric = st.selectbox("X Metric (Cause)", available_metrics, key='single_x')
    
    with col2:
        y_metric = st.selectbox("Y Metric (Effect)", 
                               [m for m in available_metrics if m != x_metric],
                               key='single_y')
    
    max_lag = st.slider("Maximum Lag", 1, 10, 5)
    
    if st.button("🚀 Run Test", type="primary"):
        st.info("Feature under development - requires separate metric preparation")

def render_cross_correlation():
    """Cross-correlation analysis"""
    
    st.subheader("📊 Cross-Correlation Analysis")
    st.markdown("""
    **Cross-Correlation** measures the similarity between two time series as a function of time lag.
    
    **Use Cases:**
    - Identify time lags between related phenomena
    - Detect leading/lagging indicators
    - Measure strength of temporal associations
    """)
    
    # Need both datasets
    if st.session_state.publications_data is None or st.session_state.patents_data is None:
        st.warning("⚠️ Both publications and patents data required")
        return
    
    pubs_df = st.session_state.publications_data
    pats_df = st.session_state.patents_data
    
    st.markdown("---")
    
    # Prepare time series
    pubs_ts = prepare_time_series_data(pubs_df, 'count', 'publications')
    pats_ts = prepare_time_series_data(pats_df, 'count', 'patents')
    
    if pubs_ts is None or pats_ts is None:
        st.error("Could not prepare time series")
        return
    
    # Merge
    merged = pd.merge(pubs_ts, pats_ts, on='year', suffixes=('_pubs', '_pats'))
    
    if len(merged) < 10:
        st.error("Insufficient time points for correlation analysis")
        return
    
    st.success(f"✅ {len(merged)} years of data available")
    
    # Parameters
    max_lag = st.slider("Maximum Lag (years)", 1, min(20, len(merged)//2), 10)
    
    if st.button("🚀 Calculate Cross-Correlation", type="primary"):
        with st.spinner("Computing cross-correlation..."):
            # Normalize time series
            pubs_normalized = (merged['value_pubs'] - merged['value_pubs'].mean()) / merged['value_pubs'].std()
            pats_normalized = (merged['value_pats'] - merged['value_pats'].mean()) / merged['value_pats'].std()
            
            # Calculate cross-correlation
            lags = range(-max_lag, max_lag + 1)
            correlations = []
            
            for lag in lags:
                if lag < 0:
                    # Pubs leads pats
                    corr = np.corrcoef(pubs_normalized[:lag], pats_normalized[-lag:])[0, 1]
                elif lag > 0:
                    # Pats leads pubs
                    corr = np.corrcoef(pubs_normalized[lag:], pats_normalized[:-lag])[0, 1]
                else:
                    # No lag
                    corr = np.corrcoef(pubs_normalized, pats_normalized)[0, 1]
                
                correlations.append(corr)
            
            # Plot
            st.markdown("### 📈 Cross-Correlation Function")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(lags),
                y=correlations,
                marker=dict(
                    color=correlations,
                    colorscale='RdBu',
                    cmid=0,
                    line=dict(width=1, color='white')
                )
            ))
            
            fig.update_layout(
                title="Cross-Correlation: Publications vs Patents",
                xaxis_title="Lag (years) - Negative = Publications Lead, Positive = Patents Lead",
                yaxis_title="Correlation Coefficient",
                template='plotly_white',
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find maximum correlation
            max_corr_idx = np.argmax(np.abs(correlations))
            optimal_lag = list(lags)[max_corr_idx]
            max_corr = correlations[max_corr_idx]
            
            st.markdown("### 🎯 Key Findings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Optimal Lag", f"{optimal_lag} years")
            
            with col2:
                st.metric("Max Correlation", f"{max_corr:.3f}")
            
            with col3:
                if optimal_lag < 0:
                    st.metric("Leading Indicator", "Publications")
                elif optimal_lag > 0:
                    st.metric("Leading Indicator", "Patents")
                else:
                    st.metric("Relationship", "Synchronous")
            
            # Interpretation
            if optimal_lag < 0:
                st.success(f"""
                **📚 Publications lead Patents by {abs(optimal_lag)} years**
                
                - Scientific publications appear to precede patent activity
                - Maximum correlation: {max_corr:.3f}
                - Suggests knowledge flows from academia to industry
                """)
            elif optimal_lag > 0:
                st.success(f"""
                **💡 Patents lead Publications by {optimal_lag} years**
                
                - Patent activity appears to precede scientific publications
                - Maximum correlation: {max_corr:.3f}
                - Suggests industry developments influence academic research
                """)
            else:
                st.success(f"""
                **🔄 Synchronous Relationship**
                
                - Publications and patents move together without lag
                - Correlation: {max_corr:.3f}
                - Suggests simultaneous development in science and technology
                """)

def render_time_series_causality():
    """Advanced time series causality"""
    
    st.subheader("⏱️ Time Series Causality Analysis")
    st.markdown("""
    Comprehensive causality analysis including:
    - **VAR (Vector Autoregression)** - Multivariate time series modeling
    - **Impulse Response Functions** - Dynamic effects of shocks
    - **Forecast Error Variance Decomposition**
    """)
    
    st.info("🚧 Advanced time series methods under development")
    
    st.markdown("""
    **Coming Soon:**
    - VAR model estimation
    - Impulse response analysis
    - Variance decomposition
    - Structural break detection
    """)

def render_transfer_entropy():
    """Transfer entropy analysis"""
    
    st.subheader("🌐 Transfer Entropy")
    st.markdown("""
    **Transfer Entropy** is an information-theoretic measure of directed information flow.
    
    **Advantages over Granger Causality:**
    - Captures non-linear relationships
    - More general measure of information transfer
    - Does not assume linearity
    
    **Formula:** TE(X→Y) = I(Y_future; X_past | Y_past)
    """)
    
    st.info("🚧 Transfer entropy calculation under development")
    
    st.markdown("""
    **Implementation Requirements:**
    - Install: `pip install pyinform` or `jpype`
    - Discretization of continuous variables
    - Proper lag selection
    """)

def render_ccm_analysis():
    """Convergent Cross Mapping"""
    
    st.subheader("📈 Convergent Cross Mapping (CCM)")
    st.markdown("""
    **CCM** detects causality in dynamical systems using state space reconstruction.
    
    **Use Cases:**
    - Detecting causality in complex systems
    - Works with shorter time series than Granger
    - Identifies bidirectional causality
    
    **Based on:** Takens' embedding theorem
    """)
    
    st.info("🚧 CCM implementation under development")
    
    st.markdown("""
    **Key Features:**
    - State space reconstruction
    - Cross-mapping skill calculation
    - Convergence analysis
    - Bidirectional causality detection
    """)

def render_propensity_score():
    """Propensity score matching"""
    
    st.subheader("🎯 Propensity Score Matching")
    st.markdown("""
    **Propensity Score Matching** estimates causal effects by balancing treatment and control groups.
    
    **Use Cases:**
    - Estimate effect of interventions
    - Compare treated vs untreated groups
    - Control for confounding variables
    
    **Example:** Effect of collaboration on citation impact
    """)
    
    dataset = st.radio("Select Dataset", ["Publications", "Patents"], horizontal=True)
    
    if dataset == "Publications":
        if st.session_state.publications_data is None:
            st.warning("⚠️ Upload publications data")
            return
        df = st.session_state.publications_data
    else:
        if st.session_state.patents_data is None:
            st.warning("⚠️ Upload patents data")
            return
        df = st.session_state.patents_data
    
    st.markdown("---")
    
    st.markdown("### 🎯 Define Treatment and Outcome")
    
    # Treatment definition
    st.markdown("**Treatment Variable:**")
    
    entity_col = 'author' if dataset == "Publications" else 'inventor'
    
    if entity_col in df.columns:
        # Define treatment as collaboration (multiple authors/inventors)
        df['treatment'] = df[entity_col].fillna('').apply(
            lambda x: 1 if len(str(x).split(';')) > 1 else 0
        )
        
        treatment_count = df['treatment'].sum()
        control_count = len(df) - treatment_count
        
        st.info(f"**Treatment:** Collaborative work (multiple {entity_col}s)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Treated Group", treatment_count)
        
        with col2:
            st.metric("Control Group", control_count)
    
    # Outcome definition
    st.markdown("**Outcome Variable:**")
    
    outcome_col = 'citations' if dataset == "Publications" else 'forward_citations'
    
    if outcome_col in df.columns:
        st.info(f"**Outcome:** {outcome_col.replace('_', ' ').title()}")
        
        # Simple comparison
        treated_outcome = df[df['treatment'] == 1][outcome_col].mean()
        control_outcome = df[df['treatment'] == 0][outcome_col].mean()
        
        st.markdown("### 📊 Naive Comparison (Without Matching)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Treated Mean", f"{treated_outcome:.2f}")
        
        with col2:
            st.metric("Control Mean", f"{control_outcome:.2f}")
        
        with col3:
            effect = treated_outcome - control_outcome
            st.metric("Naive Effect", f"{effect:+.2f}")
        
        # Visualization
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=df[df['treatment'] == 1][outcome_col],
            name='Treated (Collaborative)',
            marker_color='#3498db'
        ))
        
        fig.add_trace(go.Box(
            y=df[df['treatment'] == 0][outcome_col],
            name='Control (Solo)',
            marker_color='#e74c3c'
        ))
        
        fig.update_layout(
            title=f"{outcome_col.title()} Distribution by Treatment Status",
            yaxis_title=outcome_col.title(),
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning("""
        **⚠️ Caution:**
        
        This is a naive comparison without controlling for confounders.
        True propensity score matching would:
        - Estimate propensity scores using logistic regression
        - Match treated and control units with similar propensity scores
        - Calculate average treatment effect on the treated (ATT)
        - Assess balance after matching
        
        Full implementation requires additional covariates.
        """)