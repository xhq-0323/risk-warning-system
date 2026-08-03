"""
V2 全功能升级版 - 跨行重复融资风险预警系统
面向银行风控场景的专业化设计
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import random
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ========== 页面配置 ==========
st.set_page_config(
    page_title="跨行重复融资风险预警系统 V2",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CSS 自定义样式（银行风控风格） ==========
st.markdown("""
<style>
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-medium { color: #fd7e14; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# ========== 标题与导航 ==========
st.title("🏛️ 跨行重复融资风险预警系统 V2")
st.caption(f"⏱️ 系统运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 适用于银行票据业务风控场景")

st.markdown("---")

# ========== 侧边栏 ==========
with st.sidebar:
    st.header("📂 数据上传")
    uploaded_file = st.file_uploader("上传企业融资数据 (CSV)", type=['csv'])
    
    st.markdown("---")
    st.header("⚙️ 系统设置")
    
    low_threshold = st.slider("低风险阈值", 0.0, 0.5, 0.3, 0.05)
    high_threshold = st.slider("高风险阈值", 0.5, 1.0, 0.7, 0.05)
    
    st.markdown("---")
    st.header("📌 使用说明")
    st.info("""
    1. 上传 CSV 格式的融资数据
    2. 系统自动训练模型并预测风险
    3. 查看多维度风险分析面板
    4. 一键导出高风险企业名单
    """)

# ========== 主功能区 ==========
if uploaded_file is not None:
    # ---------- 加载数据 ----------
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)
    
    df = load_data(uploaded_file)
    
    # ---------- 数据概览 ----------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">📊 企业总数</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        bank_col = 'bank_count' if 'bank_count' in df.columns else None
        avg_banks = df[bank_col].mean() if bank_col else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_banks:.1f}</div>
            <div class="metric-label">🏦 平均关联银行数</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        revenue_col = 'revenue_loan_ratio' if 'revenue_loan_ratio' in df.columns else None
        avg_revenue = df[revenue_col].mean() if revenue_col else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_revenue:.2f}</div>
            <div class="metric-label">💰 平均营收贷款比</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-label">📋 特征维度</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ---------- 数据预览 ----------
    with st.expander("📊 原始数据预览", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"共 {len(df)} 行，{df.shape[1]} 列")
    
    # ---------- 模型训练与预测 ----------
    st.subheader("🤖 智能风险评估")
    
    with st.spinner("🔄 正在训练风险模型，请稍候..."):
        # 准备特征 - 自动过滤非数值列
        exclude_cols = ['label', 'high_invoice_reuse', 'multi_bank', 'frequent_drawdown']
        
        # 只选择数值类型的列
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 如果特征列为空，报错提示
        if len(feature_cols) == 0:
            st.error("❌ 未找到可用于训练的数值特征列，请检查数据格式")
            st.stop()
        
        X = df[feature_cols]
        
        # 处理标签
        if 'label' in df.columns:
            y = df['label']
        else:
            # 用营收贷款比和银行数模拟风险标签
            risk_score = df['revenue_loan_ratio'] / 5 + df['bank_count'] / 10
            y = (risk_score > risk_score.median()).astype(int)
        
        # 训练模型
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        prob = model.predict_proba(X)[:, 1]
        
        # 特征重要性
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    # 风险等级划分
    df['risk_prob'] = prob
    df['risk_level'] = pd.cut(
        prob, 
        bins=[0, low_threshold, high_threshold, 1.0], 
        labels=['低风险', '中风险', '高风险']
    )
    
    # ---------- 风险概览仪表板 ----------
    col1, col2, col3 = st.columns(3)
    risk_counts = df['risk_level'].value_counts()
    
    with col1:
        high_count = risk_counts.get('高风险', 0)
        st.metric("🔴 高风险企业", high_count, 
                  delta=f"占比 {high_count/len(df)*100:.1f}%" if high_count > 0 else "0%")
    with col2:
        medium_count = risk_counts.get('中风险', 0)
        st.metric("🟡 中风险企业", medium_count,
                  delta=f"占比 {medium_count/len(df)*100:.1f}%" if medium_count > 0 else "0%")
    with col3:
        low_count = risk_counts.get('低风险', 0)
        st.metric("🟢 低风险企业", low_count,
                  delta=f"占比 {low_count/len(df)*100:.1f}%" if low_count > 0 else "0%")
    
    st.markdown("---")
    
    # ---------- 风险分布可视化 ----------
    st.subheader("📈 风险分布分析")
    
    tab1, tab2, tab3, tab4 = st.tabs(["风险概率分布", "行业风险分析", "企业详情查询", "关联网络分析"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.histogram(
                df, x='risk_prob', nbins=20,
                title="企业风险概率分布",
                color_discrete_sequence=['#1f77b4'],
                labels={'risk_prob': '风险概率', 'count': '企业数量'},
                color='risk_level'
            )
            fig.add_vline(x=low_threshold, line_dash="dash", line_color="green", annotation_text="低/中阈值")
            fig.add_vline(x=high_threshold, line_dash="dash", line_color="red", annotation_text="中/高阈值")
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            pie_data = df['risk_level'].value_counts().reset_index()
            pie_data.columns = ['风险等级', '数量']
            fig_pie = px.pie(
                pie_data, values='数量', names='风险等级',
                title="风险等级占比",
                color='风险等级',
                color_discrete_map={'高风险': '#dc3545', '中风险': '#fd7e14', '低风险': '#28a745'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        if 'industry_risk' in df.columns:
            industry_map = {0: '低风险行业', 1: '中风险行业', 2: '高风险行业'}
            df['industry_label'] = df['industry_risk'].map(industry_map)
            
            col1, col2 = st.columns(2)
            with col1:
                industry_stats = df.groupby('industry_label').agg(
                    企业数量=('risk_prob', 'count'),
                    平均风险概率=('risk_prob', 'mean'),
                    高风险占比=('risk_level', lambda x: (x == '高风险').sum() / len(x) * 100)
                ).reset_index()
                
                fig2 = px.bar(
                    industry_stats, x='industry_label', y='平均风险概率',
                    color='平均风险概率', color_continuous_scale='Reds',
                    title="各行业平均风险概率",
                    labels={'industry_label': '行业类别', '平均风险概率': '平均风险概率'}
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                st.dataframe(industry_stats, use_container_width=True)
        else:
            st.warning("⚠️ 数据中无 'industry_risk' 字段，无法进行行业风险分析")
    
    with tab3:
        st.subheader("🔍 单企业风险详情查询")
        
        if 'id' in df.columns:
            id_list = df['id'].tolist()
        else:
            id_list = list(range(len(df)))
        
        selected_id = st.selectbox("选择企业", id_list, format_func=lambda x: f"企业 {x}")
        
        if 'id' in df.columns:
            row = df[df['id'] == selected_id].iloc[0]
        else:
            row = df.iloc[selected_id]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_color = "🔴" if row['risk_level'] == '高风险' else "🟡" if row['risk_level'] == '中风险' else "🟢"
            st.metric("风险等级", f"{risk_color} {row['risk_level']}")
        with col2:
            st.metric("风险概率", f"{row['risk_prob']:.4f}")
        with col3:
            st.metric("行业风险等级", row.get('industry_risk', 'N/A'))
        
        st.write("**企业特征明细：**")
        st.dataframe(pd.DataFrame({
            '特征': feature_cols,
            '数值': [row[col] for col in feature_cols]
        }), use_container_width=True)
    
    with tab4:
        if 'bank_count' in df.columns:
            st.subheader("🏦 银企融资关联网络")
            
            col1, col2 = st.columns(2)
            with col1:
                fig3 = px.histogram(
                    df, x='bank_count', 
                    title="银行关联度分布",
                    labels={'bank_count': '关联银行数量', 'count': '企业数'},
                    color_discrete_sequence=['#2ca02c']
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                bank_counts = df['bank_count'].value_counts().sort_index()
                st.write("**银行关联度统计：**")
                st.dataframe(pd.DataFrame({
                    '关联银行数': bank_counts.index,
                    '企业数量': bank_counts.values,
                    '占比': (bank_counts.values / len(df) * 100).round(2)
                }), use_container_width=True)
            
            st.markdown("**企业-银行关联网络图（前30条）**")
            sample_df = df.head(30).copy()
            bank_names = ['工商银行', '建设银行', '农业银行', '中国银行', '交通银行', '招商银行', '浦发银行', '中信银行']
            
            edges = []
            for idx, row in sample_df.iterrows():
                n_banks = int(row['bank_count']) if not pd.isna(row['bank_count']) else 0
                n_banks = min(n_banks, len(bank_names))
                chosen = random.sample(bank_names, n_banks) if n_banks > 0 else []
                for b in chosen:
                    edges.append((f"企业{idx}", b))
            
            G = nx.Graph()
            G.add_edges_from(edges)
            
            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, seed=42)
                edge_x, edge_y = [], []
                for e in G.edges():
                    x0, y0 = pos[e[0]]
                    x1, y1 = pos[e[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                node_x, node_y = [], []
                node_colors = []
                for n in G.nodes():
                    x, y = pos[n]
                    node_x.append(x)
                    node_y.append(y)
                    if n.startswith('企业'):
                        node_colors.append('#1f77b4')
                    else:
                        node_colors.append('#ff7f0e')
                
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=edge_x, y=edge_y, 
                    mode='lines', 
                    line=dict(width=1, color='#888'), 
                    hoverinfo='none'
                ))
                fig4.add_trace(go.Scatter(
                    x=node_x, y=node_y, 
                    mode='markers+text', 
                    marker=dict(size=15, color=node_colors),
                    text=list(G.nodes()), 
                    textposition="top center"
                ))
                fig4.update_layout(
                    title='企业-银行关联网络图',
                    showlegend=False, 
                    hovermode='closest',
                    height=500
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("无关联边可绘制")
        else:
            st.warning("数据中无 'bank_count' 字段")
    
    # ---------- 特征重要性 ----------
    st.subheader("📊 特征重要性分析")
    fig_importance = px.bar(
        importance.head(10), x='importance', y='feature',
        orientation='h',
        title="Top 10 风险特征",
        color='importance',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # ---------- 高风险名单导出 ----------
    st.markdown("---")
    st.subheader("📋 高风险企业名单")
    
    high_risk_df = df[df['risk_level'] == '高风险'].copy()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"共 **{len(high_risk_df)}** 家高风险企业")
    with col2:
        if len(high_risk_df) > 0:
            csv = high_risk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 下载高风险名单",
                csv,
                f"high_risk_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    if len(high_risk_df) > 0:
        st.dataframe(high_risk_df[feature_cols[:5] + ['risk_prob', 'risk_level']], use_container_width=True)
    else:
        st.success("🎉 恭喜！当前数据中未发现高风险企业")
    
else:
    # ---------- 未上传数据时的欢迎界面 ----------
    st.info("👈 请从左侧上传 CSV 数据文件开始分析")
    
    st.markdown("""
    ### 🏛️ 系统功能概览
    
    | 功能模块 | 说明 |
    | :--- | :--- |
    | **📊 数据概览** | 展示企业数量、平均关联银行数、营收贷款比等关键指标 |
    | **🤖 智能预测** | 基于随机森林算法实时评估企业重复融资风险 |
    | **📈 风险分布** | 概率分布图、行业风险对比、等级占比饼图 |
    | **🔍 企业详情** | 查询任意企业的风险评分和特征明细 |
    | **🏦 关联网络** | 可视化展示企业与银行之间的融资关联关系 |
    | **📋 名单导出** | 一键导出高风险企业名单（CSV格式） |
    
    ### 📌 数据格式要求
    - 文件格式：CSV
    - 必须包含数值类型的特征列
    - 可选字段：`label`（风险标签，用于模型训练）
    """)

# ========== 页脚 ==========
st.markdown("---")
st.caption("🏛️ 跨行重复融资风险预警系统 V2 | 金融科技创新大赛 | 基于 Streamlit 构建")
