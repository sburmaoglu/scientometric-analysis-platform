"""Patents Analysis Module - Full Implementation"""

from core.base_module import BaseModule
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

class PatentsAnalysisModule(BaseModule):
    """Comprehensive patent data analysis"""
    
    def render(self):
        st.title("💡 Patents Analysis")
        st.markdown("Comprehensive patent data analysis with interactive visualizations")
        
        if not self.check_data_availability('patents'):
            self.show_data_required_message()
            return
        
        df = st.session_state.patents_data
        
        st.markdown("---")
        
        # Overview Metrics
        st.subheader("📊 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patents", f"{len(df):,}")
        
        with col2:
            if 'year' in df.columns:
                years = df['year'].dropna()
                if len(years) > 0:
                    st.metric("Year Range", f"{int(years.min())}-{int(years.max())}")
        
        with col3:
            if 'forward_citations' in df.columns:
                total_cites = df['forward_citations'].sum()
                if pd.notna(total_cites):
                    st.metric("Total Citations", f"{int(total_cites):,}")
        
        with col4:
            if 'family_size' in df.columns:
                avg_size = df['family_size'].mean()
                if pd.notna(avg_size):
                    st.metric("Avg Family Size", f"{avg_size:.1f}")
        
        st.markdown("---")
        
        # Analysis Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Temporal Trends",
            "🏢 Organizations", 
            "👥 Inventors",
            "🗺️ Geographic",
            "🔬 Technology"
        ])
        
        with tab1:
            self.render_temporal_analysis(df)
        
        with tab2:
            self.render_organization_analysis(df)
        
        with tab3:
            self.render_inventor_analysis(df)
        
        with tab4:
            self.render_geographic_analysis(df)
        
        with tab5:
            self.render_technology_analysis(df)
    
    def render_temporal_analysis(self, df):
        """Temporal trends analysis"""
        st.subheader("📈 Patent Filings Over Time")
        
        if 'year' not in df.columns:
            st.warning("Year data not available")
            return
        
        yearly = df.groupby('year').size().reset_index(name='count')
        
        # Line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly['year'],
            y=yearly['count'],
            mode='lines+markers',
            marker=dict(size=10, color='#e74c3c', line=dict(width=2, color='white')),
            line=dict(width=3, color='#c0392b'),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.1)',
            hovertemplate='<b>Year: %{x}</b><br>Patents: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Patent Filings per Year",
            xaxis_title="Year",
            yaxis_title="Number of Patents",
            template='plotly_white',
            hovermode='x unified',
            height=450,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("#### 📊 Temporal Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average/Year", f"{yearly['count'].mean():.1f}")
        
        with col2:
            peak_idx = yearly['count'].idxmax()
            st.metric("Peak Year", f"{yearly.loc[peak_idx, 'year']:.0f}")
        
        with col3:
            st.metric("Peak Count", f"{yearly['count'].max():.0f}")
        
        with col4:
            if len(yearly) > 1:
                growth = ((yearly['count'].iloc[-1] - yearly['count'].iloc[0]) / yearly['count'].iloc[0] * 100)
                st.metric("Total Growth", f"{growth:+.1f}%")
        
        # Download data
        csv = yearly.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Temporal Data",
            csv,
            "patent_temporal_data.csv",
            "text/csv",
            key='download_temporal'
        )
    
    def render_organization_analysis(self, df):
        """Organization analysis"""
        st.subheader("🏢 Top Patent Holders")
        
        if 'assignee' not in df.columns:
            st.warning("Assignee data not available")
            return
        
        # Parse organizations
        all_orgs = []
        for orgs_str in df['assignee'].dropna():
            if pd.notna(orgs_str):
                orgs = str(orgs_str).split(';')
                all_orgs.extend([o.strip() for o in orgs if o.strip()])
        
        if not all_orgs:
            st.info("No organization data available")
            return
        
        # Count and create dataframe
        org_counts = Counter(all_orgs)
        top_orgs = pd.DataFrame(
            org_counts.most_common(20),
            columns=['Organization', 'Patent Count']
        )
        
        # Horizontal bar chart
        fig = px.bar(
            top_orgs.head(15),
            x='Patent Count',
            y='Organization',
            orientation='h',
            title="Top 15 Patent Holders",
            color='Patent Count',
            color_continuous_scale='Reds',
            text='Patent Count'
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Unique Organizations", f"{len(org_counts):,}")
        
        with col2:
            top_org_pct = (top_orgs.iloc[0]['Patent Count'] / len(df)) * 100
            st.metric("Top Org Share", f"{top_org_pct:.1f}%")
        
        with col3:
            top10_pct = (top_orgs.head(10)['Patent Count'].sum() / len(df)) * 100
            st.metric("Top 10 Share", f"{top10_pct:.1f}%")
        
        # Full table
        with st.expander("📊 View Complete Table"):
            st.dataframe(top_orgs, use_container_width=True, hide_index=True)
        
        # Download
        csv = top_orgs.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Organization Data",
            csv,
            "top_organizations.csv",
            "text/csv",
            key='download_orgs'
        )
    
    def render_inventor_analysis(self, df):
        """Inventor analysis"""
        st.subheader("👥 Top Inventors")
        
        if 'inventor' not in df.columns:
            st.warning("Inventor data not available")
            return
        
        # Parse inventors
        all_inventors = []
        for inv_str in df['inventor'].dropna():
            if pd.notna(inv_str):
                inventors = str(inv_str).split(';')
                all_inventors.extend([i.strip() for i in inventors if i.strip()])
        
        if not all_inventors:
            st.info("No inventor data available")
            return
        
        # Count
        inv_counts = Counter(all_inventors)
        top_inventors = pd.DataFrame(
            inv_counts.most_common(20),
            columns=['Inventor', 'Patent Count']
        )
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Inventors", f"{len(inv_counts):,}")
        
        with col2:
            st.metric("Most Prolific", top_inventors.iloc[0]['Inventor'][:30] + "...")
        
        with col3:
            st.metric("Their Patents", f"{top_inventors.iloc[0]['Patent Count']:.0f}")
        
        # Table
        st.dataframe(top_inventors.head(15), use_container_width=True, hide_index=True)
        
        # Collaboration analysis
        st.markdown("#### 🤝 Collaboration Metrics")
        
        inventors_per_patent = []
        for inv_str in df['inventor'].dropna():
            if pd.notna(inv_str):
                count = len([i for i in str(inv_str).split(';') if i.strip()])
                inventors_per_patent.append(count)
        
        if inventors_per_patent:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_inv = sum(inventors_per_patent) / len(inventors_per_patent)
                st.metric("Avg Inventors/Patent", f"{avg_inv:.2f}")
            
            with col2:
                single = inventors_per_patent.count(1)
                single_pct = (single / len(inventors_per_patent)) * 100
                st.metric("Single-Inventor %", f"{single_pct:.1f}%")
            
            with col3:
                max_inv = max(inventors_per_patent)
                st.metric("Max Inventors", f"{max_inv}")
    
    def render_geographic_analysis(self, df):
        """Geographic distribution"""
        st.subheader("🗺️ Geographic Distribution")
        
        if 'jurisdiction' not in df.columns:
            st.warning("Jurisdiction data not available")
            return
        
        # Count by jurisdiction
        juris_counts = df['jurisdiction'].value_counts().head(20)
        
        geo_df = pd.DataFrame({
            'Jurisdiction': juris_counts.index,
            'Patent Count': juris_counts.values
        })
        
        # Bar chart
        fig = px.bar(
            geo_df.head(15),
            x='Jurisdiction',
            y='Patent Count',
            title="Patents by Jurisdiction (Top 15)",
            color='Patent Count',
            color_continuous_scale='Blues',
            text='Patent Count'
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(
            template='plotly_white',
            height=450,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Jurisdictions", len(df['jurisdiction'].unique()))
        
        with col2:
            st.metric("Top Jurisdiction", geo_df.iloc[0]['Jurisdiction'])
        
        with col3:
            top_pct = (geo_df.iloc[0]['Patent Count'] / len(df)) * 100
            st.metric("Top Share", f"{top_pct:.1f}%")
        
        with col4:
            top5_pct = (geo_df.head(5)['Patent Count'].sum() / len(df)) * 100
            st.metric("Top 5 Share", f"{top5_pct:.1f}%")
        
        # Full table
        with st.expander("📊 All Jurisdictions"):
            st.dataframe(geo_df, use_container_width=True, hide_index=True)
    
    def render_technology_analysis(self, df):
        """Technology classification"""
        st.subheader("🔬 Technology Classification")
        
        # Check available classifications
        has_ipc = 'ipc_class' in df.columns
        has_cpc = 'cpc_class' in df.columns
        
        if not has_ipc and not has_cpc:
            st.warning("No classification data available")
            return
        
        # Choose classification
        class_col = 'ipc_class' if has_ipc else 'cpc_class'
        class_name = 'IPC' if has_ipc else 'CPC'
        
        st.markdown(f"**Analyzing {class_name} Classifications**")
        
        # Parse classifications
        all_classes = []
        for class_str in df[class_col].dropna():
            if pd.notna(class_str):
                classes = str(class_str).split(';')
                main_classes = [c.strip()[:4] for c in classes if c.strip()]
                all_classes.extend(main_classes)
        
        if not all_classes:
            st.info("No classification data found")
            return
        
        # Count
        class_counts = Counter(all_classes)
        top_classes = pd.DataFrame(
            class_counts.most_common(15),
            columns=['Classification', 'Patent Count']
        )
        
        # Pie chart
        fig = px.pie(
            top_classes.head(10),
            values='Patent Count',
            names='Classification',
            title=f"Top 10 {class_name} Classifications",
            hole=0.3
        )
        
        fig.update_traces(textposition='inside', textinfo='label+percent')
        fig.update_layout(template='plotly_white', height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Unique Classes", f"{len(class_counts):,}")
        
        with col2:
            st.metric("Most Common", top_classes.iloc[0]['Classification'])
        
        with col3:
            top_pct = (top_classes.iloc[0]['Patent Count'] / len(all_classes)) * 100
            st.metric("Top Class Share", f"{top_pct:.1f}%")
        
        # Table
        with st.expander("📊 Complete Classification List"):
            st.dataframe(top_classes, use_container_width=True, hide_index=True)
   Summary: Fully implemented Patents Analysis module
   Description: Added 5 analysis tabs with interactive visualizations and download functionality
