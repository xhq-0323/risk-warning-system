import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import random
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="票据重复融资风险预警系统",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 跨行重复融资风险预警系统")
st.markdown("---")

st.sidebar.header("📂 数据上传")
uploaded_file = st.sidebar.file_uploader("上传企业融资数据 (CSV)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ 数据上传成功！")
    
    st.subheader("📊 原始数据预览")
    st.dataframe(df.head(10), use_container_width=True)

    exclude_cols = ['label', 'high_invoice_reuse', 'multi_bank', 'frequent_drawdown']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    
    if 'label' in df.columns:
        y = df['label']
    else:
        risk_score = df['revenue_loan_ratio'] / 5 + df['bank_count'] / 10
        y = (risk_score > risk_score.median()).astype(int)
    
    with st.spinner("🔄 正在训练模型，请稍候..."):
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        prob = model.predict_proba(X)[:, 1]
    
    df['risk_prob'] = prob
    df['risk_level'] = pd.cut(prob, bins=[0, 0.3, 0.7, 1.0], labels=['低', '中', '高'])

    st.subheader("🔍 风险预测结果（前10条）")
    st.dataframe(df[['risk_prob', 'risk_level']].head(10), use_container_width=True)

    fig = px.histogram(df, x='risk_prob', nbins=20, 
                       title="企业风险概率分布",
                       color_discrete_sequence=['#1f77b4'])
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

    high_risk = df[df['risk_level'] == '高']
    st.write(f"⚠️ 高风险企业数量：**{len(high_risk)}**")
    if len(high_risk) > 0:
        csv = high_risk.to_csv(index=False).encode('utf-8')
        st.download_button("📥 下载高风险名单", csv, "high_risk.csv", "text/csv")
    else:
        st.info("🎉 暂无高风险企业")

    st.markdown("---")
    st.header("🔎 高级分析功能")

    with st.expander("🧑‍💼 单企业风险评分查询", expanded=False):
        if 'id' in df.columns:
            id_list = df['id'].tolist()
        else:
            id_list = list(range(len(df)))
        selected_id = st.selectbox("选择企业", id_list, format_func=lambda x: f"企业 {x}")
        if 'id' in df.columns:
            row = df[df['id'] == selected_id].iloc[0]
        else:
            row = df.iloc[selected_id]
        st.write("**风险概率**:", f"{row['risk_prob']:.4f}")
        st.write("**风险等级**:", row['risk_level'])

    with st.expander("🏭 行业风险热力图", expanded=False):
        if 'industry_risk' in df.columns:
            industry_map = {0: '低风险行业', 1: '中风险行业', 2: '高风险行业'}
            df['industry_label'] = df['industry_risk'].map(industry_map)
            industry_stats = df.groupby('industry_label').agg(
                avg_risk=('risk_prob', 'mean'),
                count=('risk_prob', 'count')
            ).reset_index()
            fig2 = px.bar(industry_stats, x='industry_label', y='avg_risk',
                          color='avg_risk', color_continuous_scale='Reds',
                          title='各行业平均风险概率')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("数据中无 'industry_risk' 字段")

    with st.expander("🏦 多银行主体融资关联图谱", expanded=False):
        if 'bank_count' in df.columns:
            fig3 = px.histogram(df, x='bank_count', title='涉及银行数量分布')
            st.plotly_chart(fig3, use_container_width=True)
            
            st.markdown("**（模拟企业-银行关联图，前20条）**")
            sample_df = df.head(20).copy()
            bank_names = ['工商银行', '建设银行', '农业银行', '中国银行', '交通银行', '招商银行']
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
                for n in G.nodes():
                    x, y = pos[n]
                    node_x.append(x)
                    node_y.append(y)
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#888'), hoverinfo='none'))
                fig4.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', marker=dict(size=12, color='#1f77b4'),
                                          text=list(G.nodes()), textposition="top center"))
                fig4.update_layout(title='企业-银行关联示意图（前20条）', showlegend=False)
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("无关联边可绘制")
        else:
            st.warning("数据中无 'bank_count' 字段")

else:
    st.info("👈 请从左侧上传 CSV 数据文件开始分析")
