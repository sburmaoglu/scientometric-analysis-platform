"""Patents Analysis Module - Full Implementation"""

from core.base_module import BaseModule
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

class PatentsAnalysisModule(BaseModule):
    """Comprehensive patent data analysis module"""
    
    def render(self):
        st.title("💡 Patents Analysis")
        st.markdown("Comprehensive analysis of patent data with statistical validation")
        
        if not self.check_data_availability('patents'):
            self.show_data_required_message()
            return
        
        df = st.session_state.patents_data
        
        st.markdown("---")
        
        # Overview Metrics
        self.show_overview_metrics(df)
        
        st.markdown("---")
        
        # Analysis Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Temporal Trends", 
            "🏢 Organizations", 
            "👥 Inventors",
            "🗺️ Geographic Distribution",
            "🔍 Technology Classification"
        ])
        
        with tab1:
            self.temporal_analysis(df)
        
        with tab2:
            self.organization_analysis(df)
        
        with tab3:
            self.inventor_analysis(df)
        
        with tab4:
            self.geographic_analysis(df)
        
        with tab5:
            self.technology_analysis(df)
    
    def show_overview_metrics(self, df):
        """Display key metrics"""
        st.subheader("📊 Overview Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patents", f"{len(df):,}")
        
        with col2:
            if 'year' in df.columns:
                years = df['year'].dropna()
                if len(years) > 0:
                    year_range = f"{int(years.min())}-{int(years.max())}"
                    st.metric("Year Range", year_range)
        
        with col3:
            if 'forward_citations' in df.columns:
                total_cites = df['forward_citations'].sum()
                st.metric("Total Citations", f"{int(total_cites):,}")
        
        with col4:
            if 'family_size' in df.columns:
                avg_family = df['family_size'].mean()
                st.metric("Avg Family Size", f"{avg_family:.1f}")
    
    def temporal_analysis(self, df):
        """Analyze temporal trends"""
        st.subheader("📈 Patents Over Time")
        
        if 'year' not in df.columns:
            st.warning("Year column not available")
            return
        
        # Group by year
        yearly = df.groupby('year').size().reset_index(name='count')
        
        # Create line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly['year'],
            y=yearly['count'],
            mode='lines+markers',
            marker=dict(size=8, color='#e74c3c'),
            line=dict(width=3, color='#c0392b'),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.1)'
        ))
        
        fig.update_layout(
            title="Patent Filings per Year",
            xaxis_title="Year",
            yaxis_title="Number of Patents",
            template='plotly_white',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average per Year", f"{yearly['count'].mean():.1f}")
        
        with col2:
            st.metric("Peak Year", f"{yearly.loc[yearly['count'].idxmax(), 'year']:.0f}")
        
        with col3:
            st.metric("Peak Count", f"{yearly['count'].max():.0f}")
        
        # Growth analysis
        if len(yearly) > 1:
            st.markdown("#### 📊 Growth Analysis")
            
            growth_rate = ((yearly['count'].iloc[-1] - yearly['count'].iloc[0]) / 
                          yearly['count'].iloc[0] * 100)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Overall Growth", f"{growth_rate:.1f}%")
            
            with col2:
                recent_5yr = yearly.tail(5)['count'].mean()
                early_5yr = yearly.head(5)['count'].mean()
                trend = "Increasing" if recent_5yr > early_5yr else "Decreasing"
                st.metric("5-Year Trend", trend)
    
    def organization_analysis(self, df):
        """Analyze organizations/assignees"""
        st.subheader("🏢 Top Organizations")
        
        if 'assignee' not in df.columns:
            st.warning("Assignee column not available")
            return
        
        # Parse semicolon-separated assignees
        all_assignees = []
        for assignees_str in df['assignee'].dropna():
            if pd.notna(assignees_str):
                assignees = str(assignees_str).split(';')
                all_assignees.extend([a.strip() for a in assignees if a.strip()])
        
        if not all_assignees:
            st.info("No assignee data available")
            return
        
        # Count and get top organizations
        assignee_counts = Counter(all_assignees)
        top_assignees = pd.DataFrame(
            assignee_counts.most_common(20),
            columns=['Organization', 'Patent Count']
        )
        
        # Bar chart
        fig = px.bar(
            top_assignees.head(15),
            x='Patent Count',
            y='Organization',
            orientation='h',
            title="Top 15 Patent Holders",
            color='Patent Count',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show table
        with st.expander("📊 View Full Table"):
            st.dataframe(
                top_assignees,
                use_container_width=True,
                hide_index=True
            )
        
        # Download button
        csv = top_assignees.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Organization Data",
            csv,
            "top_organizations.csv",
            "text/csv"
        )
    
    def inventor_analysis(self, df):
        """Analyze inventors"""
        st.subheader("👥 Top Inventors")
        
        if 'inventor' not in df.columns:
            st.warning("Inventor column not available")
            return
        
        # Parse semicolon-separated inventors
        all_inventors = []
        for inventors_str in df['inventor'].dropna():
            if pd.notna(inventors_str):
                inventors = str(inventors_str).split(';')
                all_inventors.extend([i.strip() for i in inventors if i.strip()])
        
        if not all_inventors:
            st.info("No inventor data available")
            return
        
        # Count
        inventor_counts = Counter(all_inventors)
        top_inventors = pd.DataFrame(
            inventor_counts.most_common(20),
            columns=['Inventor', 'Patent Count']
        )
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Unique Inventors", f"{len(inventor_counts):,}")
        
        with col2:
            st.metric("Most Prolific", top_inventors.iloc[0]['Inventor'])
        
        with col3:
            st.metric("Their Patents", f"{top_inventors.iloc[0]['Patent Count']:.0f}")
        
        # Table
        st.dataframe(
            top_inventors.head(15),
            use_container_width=True,
            hide_index=True
        )
        
        # Collaboration analysis
        st.markdown("#### 🤝 Collaboration Analysis")
        
        # Count patents by number of inventors
        inventor_counts_per_patent = []
        for inventors_str in df['inventor'].dropna():
            if pd.notna(inventors_str):
                count = len([i for i in str(inventors_str).split(';') if i.strip()])
                inventor_counts_per_patent.append(count)
        
        if inventor_counts_per_patent:
            avg_inventors = sum(inventor_counts_per_patent) / len(inventor_counts_per_patent)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Avg Inventors per Patent", f"{avg_inventors:.2f}")
            
            with col2:
                single_inventor = inventor_counts_per_patent.count(1)
                pct_single = (single_inventor / len(inventor_counts_per_patent)) * 100
                st.metric("Single-Inventor Patents", f"{pct_single:.1f}%")
    
    def geographic_analysis(self, df):
        """Analyze geographic distribution"""
        st.subheader("🗺️ Geographic Distribution")
        
        if 'jurisdiction' not in df.columns:
            st.warning("Jurisdiction column not available")
            return
        
        # Count by jurisdiction
        jurisdiction_counts = df['jurisdiction'].value_counts().head(20)
        
        # Create dataframe
        geo_df = pd.DataFrame({
            'Jurisdiction': jurisdiction_counts.index,
            'Patent Count': jurisdiction_counts.values
        })
        
        # Bar chart
        fig = px.bar(
            geo_df.head(15),
            x='Jurisdiction',
            y='Patent Count',
            title="Patents by Jurisdiction",
            color='Patent Count',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Jurisdictions", len(df['jurisdiction'].unique()))
        
        with col2:
            top_country = geo_df.iloc[0]['Jurisdiction']
            st.metric("Top Jurisdiction", top_country)
        
        with col3:
            top_count = geo_df.iloc[0]['Patent Count']
            pct = (top_count / len(df)) * 100
            st.metric("Their Share", f"{pct:.1f}%")
        
        # Full table
        with st.expander("📊 View All Jurisdictions"):
            st.dataframe(geo_df, use_container_width=True, hide_index=True)
    
    def technology_analysis(self, df):
        """Analyze technology classifications"""
        st.subheader("🔬 Technology Classification Analysis")
        
        # Check which classification system is available
        has_ipc = 'ipc_class' in df.columns
        has_cpc = 'cpc_class' in df.columns
        
        if not has_ipc and not has_cpc:
            st.warning("No classification data available (IPC or CPC)")
            return
        
        # Choose which classification to use
        classification_col = 'ipc_class' if has_ipc else 'cpc_class'
        classification_name = 'IPC' if has_ipc else 'CPC'
        
        st.markdown(f"**Using {classification_name} Classifications**")
        
        # Parse classifications (semicolon-separated)
        all_classes = []
        for classes_str in df[classification_col].dropna():
            if pd.notna(classes_str):
                classes = str(classes_str).split(';')
                # Take first 4 characters for main class
                main_classes = [c.strip()[:4] for c in classes if c.strip()]
                all_classes.extend(main_classes)
        
        if not all_classes:
            st.info("No classification data found")
            return
        
        # Count top classes
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
            title=f"Top 10 {classification_name} Classifications"
        )
        
        fig.update_layout(template='plotly_white', height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.dataframe(top_classes, use_container_width=True, hide_index=True)
        
        # Statistics
        st.markdown("#### 📊 Classification Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Unique Classifications", f"{len(class_counts):,}")
        
        with col2:
            st.metric("Most Common", top_classes.iloc[0]['Classification'])
```

---

## 🚀 How to Deploy:

1. **Replace** `src/modules/patents_analysis/module.py` with the code above
2. **Commit:**
```
   Summary: Implemented full Patents Analysis module
   Description: Added temporal, organization, inventor, geographic, and technology classification analyses
