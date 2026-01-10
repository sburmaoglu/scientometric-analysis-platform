"""Publications Analysis Module"""

from core.base_module import BaseModule
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.statistical_tests import calculate_descriptive_stats, calculate_correlation
from visualizations.temporal_viz import create_timeline_chart
from visualizations.statistical_viz import create_distribution_plot

class PublicationsAnalysisModule(BaseModule):
    """Module for analyzing publication data"""

    def render(self):
        """Render the publications analysis interface"""

        st.title("📚 Publications Analysis")
        st.markdown("Comprehensive analysis of publication data with statistical validation")

        if not self.check_data_availability('publications'):
            self.show_data_required_message()
            return

        df = st.session_state.publications_data

        st.markdown("---")

        # Overview metrics
        st.subheader("📊 Overview Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Publications", f"{len(df):,}")

        with col2:
            if 'year' in df.columns:
                year_range = f"{df['year'].min():.0f}-{df['year'].max():.0f}"
                st.metric("Year Range", year_range)

        with col3:
            if 'citations' in df.columns:
                total_cites = df['citations'].sum()
                st.metric("Total Citations", f"{total_cites:,.0f}")

        with col4:
            if 'author' in df.columns:
                unique_authors = df['author'].nunique()
                st.metric("Unique Authors", f"{unique_authors:,}")

        st.markdown("---")

        # Analysis tabs
        tab1, tab2, tab3 = st.tabs(["📈 Temporal Trends", "📊 Statistical Analysis", "🔍 Detailed View"])

        with tab1:
            st.subheader("📈 Publications Over Time")

            if 'year' in df.columns:
                yearly_counts = df.groupby('year').size().reset_index(name='count')

                fig = create_timeline_chart(yearly_counts, 'year', 'count',
                                            "Publications per Year")
                st.plotly_chart(fig, use_container_width=True)

                # Statistical summary
                st.markdown("#### 📊 Temporal Statistics")

                col1, col2 = st.columns(2)

                with col1:
                    stats = calculate_descriptive_stats(yearly_counts['count'])

                    st.markdown(f"""
                    **Publications per Year:**
                    - Mean: {stats['mean']:.2f}
                    - Median: {stats['median']:.0f}
                    - Std Dev: {stats['std']:.2f}
                    - Range: {stats['min']:.0f} - {stats['max']:.0f}
                    """)

                with col2:
                    growth_rate = ((yearly_counts['count'].iloc[-1] - yearly_counts['count'].iloc[0]) /
                                  yearly_counts['count'].iloc[0] * 100)

                    st.metric("Overall Growth", f"{growth_rate:.1f}%")

                    if len(yearly_counts) > 1:
                        recent_trend = yearly_counts['count'].iloc[-5:].mean() - yearly_counts['count'].iloc[:5].mean()
                        st.metric("Recent Trend",
                                 f"{'↑' if recent_trend > 0 else '↓'} {abs(recent_trend):.1f}")
            else:
                st.warning("Year column not found in data")

        with tab2:
            st.subheader("📊 Citation Analysis")

            if 'citations' in df.columns:
                citations = df['citations'].dropna()

                if len(citations) > 0:
                    # Distribution plot
                    fig = create_distribution_plot(citations, "Citation Distribution")
                    st.plotly_chart(fig, use_container_width=True)

                    # Descriptive statistics
                    st.markdown("#### 📈 Citation Statistics")

                    stats = calculate_descriptive_stats(citations)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Mean", f"{stats['mean']:.2f}")
                        st.metric("Median", f"{stats['median']:.0f}")

                    with col2:
                        st.metric("Std Dev", f"{stats['std']:.2f}")
                        st.metric("Total", f"{citations.sum():,.0f}")

                    with col3:
                        st.metric("Min", f"{stats['min']:.0f}")
                        st.metric("Max", f"{stats['max']:.0f}")

                    # Top cited publications
                    st.markdown("#### 🏆 Top Cited Publications")

                    top_pubs = df.nlargest(10, 'citations')[['title', 'year', 'citations']]
                    st.dataframe(top_pubs, use_container_width=True, hide_index=True)

                    # Correlation with year
                    if 'year' in df.columns:
                        st.markdown("#### 📉 Citation-Year Correlation")

                        valid_data = df[['year', 'citations']].dropna()

                        if len(valid_data) >= 3:
                            corr_result = calculate_correlation(valid_data['year'],
                                                               valid_data['citations'],
                                                               'pearson')

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown(f"""
                                **Pearson Correlation:**
                                - r = {corr_result['correlation']:.4f}
                                - p-value = {corr_result['p_value']:.4f}
                                - n = {corr_result['n']}
                                - Significant: {'Yes ✓' if corr_result['significant'] else 'No ✗'}
                                """)

                            with col2:
                                # Scatter plot
                                fig = px.scatter(valid_data, x='year', y='citations',
                                               title="Citations vs Year",
                                               trendline="ols",
                                               template='plotly_white')
                                fig.update_traces(marker=dict(color='#667eea'))
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Citations column not found in data")

        with tab3:
            st.subheader("🔍 Detailed Data View")

            # Filters
            col1, col2 = st.columns(2)

            with col1:
                if 'year' in df.columns:
                    year_min = int(df['year'].min())
                    year_max = int(df['year'].max())
                    year_filter = st.slider("Filter by Year", year_min, year_max,
                                           (year_min, year_max))
                    df_filtered = df[(df['year'] >= year_filter[0]) &
                                    (df['year'] <= year_filter[1])]
                else:
                    df_filtered = df

            with col2:
                if 'citations' in df.columns:
                    min_citations = st.number_input("Minimum Citations", 0,
                                                   int(df['citations'].max()), 0)
                    df_filtered = df_filtered[df_filtered['citations'] >= min_citations]

            st.markdown(f"**Showing {len(df_filtered):,} of {len(df):,} publications**")

            st.dataframe(df_filtered, use_container_width=True, height=400)

            # Export options
            st.markdown("#### 💾 Export Data")

            col1, col2 = st.columns(2)

            with col1:
                csv = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Filtered Data (CSV)",
                    csv,
                    "publications_filtered.csv",
                    "text/csv"
                )

            with col2:
                if st.button("📊 Generate Full Report"):
                    st.info("Report generation feature coming soon!")
