"""Patents Analysis Module"""

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
        st.markdown("Comprehensive analysis of patent data")
        
        if not self.check_data_availability('patents'):
            self.show_data_required_message()
            return
        
        df = st.session_state.patents_data
        
        st.markdown("---")
        
        # Overview Metrics
        st.subheader("📊 Overview Metrics")
        
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
                st.metric("Total Citations", f"{int(df['forward_citations'].sum()):,}")
        
        with col4:
            if 'family_size' in df.columns:
                st.metric("Avg Family Size", f"{df['family_size'].mean():.1f}")
        
        st.markdown("---")
        
        # Analysis Tabs
        tab1, tab2, tab3 = st.tabs(["📈 Temporal", "🏢 Organizations", "🗺️ Geographic"])
        
        with tab1:
            st.subheader("📈 Patents Over Time")
            
            if 'year' in df.columns:
                yearly = df.groupby('year').size().reset_index(name='count')
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=yearly['year'],
                    y=yearly['count'],
                    mode='lines+markers',
                    marker=dict(size=8, color='#e74c3c'),
                    line=dict(width=3)
                ))
                
                fig.update_layout(
                    title="Patent Filings per Year",
                    xaxis_title="Year",
                    yaxis_title="Count",
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg per Year", f"{yearly['count'].mean():.1f}")
                with col2:
                    st.metric("Peak Year", f"{yearly.loc[yearly['count'].idxmax(), 'year']:.0f}")
                with col3:
                    st.metric("Peak Count", f"{yearly['count'].max():.0f}")
            else:
                st.warning("Year column not available")
        
        with tab2:
            st.subheader("🏢 Top Organizations")
            
            if 'assignee' in df.columns:
                all_assignees = []
                for assignees_str in df['assignee'].dropna():
                    if pd.notna(assignees_str):
                        assignees = str(assignees_str).split(';')
                        all_assignees.extend([a.strip() for a in assignees if a.strip()])
                
                if all_assignees:
                    assignee_counts = Counter(all_assignees)
                    top_assignees = pd.DataFrame(
                        assignee_counts.most_common(15),
                        columns=['Organization', 'Patents']
                    )
                    
                    fig = px.bar(
                        top_assignees,
                        x='Patents',
                        y='Organization',
                        orientation='h',
                        title="Top 15 Patent Holders"
                    )
                    
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        template='plotly_white',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(top_assignees, use_container_width=True, hide_index=True)
                else:
                    st.info("No assignee data available")
            else:
                st.warning("Assignee column not available")
        
        with tab3:
            st.subheader("🗺️ Geographic Distribution")
            
            if 'jurisdiction' in df.columns:
                jurisdiction_counts = df['jurisdiction'].value_counts().head(15)
                
                geo_df = pd.DataFrame({
                    'Jurisdiction': jurisdiction_counts.index,
                    'Count': jurisdiction_counts.values
                })
                
                fig = px.bar(
                    geo_df,
                    x='Jurisdiction',
                    y='Count',
                    title="Patents by Jurisdiction"
                )
                
                fig.update_layout(template='plotly_white', height=400)
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Jurisdictions", len(df['jurisdiction'].unique()))
                with col2:
                    st.metric("Top Jurisdiction", geo_df.iloc[0]['Jurisdiction'])
                with col3:
                    pct = (geo_df.iloc[0]['Count'] / len(df)) * 100
                    st.metric("Their Share", f"{pct:.1f}%")
            else:
                st.warning("Jurisdiction column not available")
```

---

## 🔧 Step-by-Step Fix:

### 1. **Open the file:**
- Navigate to `src/modules/patents_analysis/module.py`

### 2. **Delete ALL content**
- Select all (Cmd+A)
- Delete

### 3. **Copy the code above**
- Copy the ENTIRE code block above

### 4. **Paste into the file**
- Paste (Cmd+V)

### 5. **Save the file**
- Save (Cmd+S)

### 6. **Verify the file:**
- Check that the first line is: `"""Patents Analysis Module"""`
- Check that there's a blank line after it
- Check that the class name is: `class PatentsAnalysisModule(BaseModule):`
- Check that `def render(self):` exists

---

## 🚀 Deploy:

1. **Commit in GitHub Desktop:**
```
   Summary: Fixed Patents Analysis module implementation
