import streamlit as st
import pandas as pd
import json
import os
import altair as alt  # 고급 시각화를 위한 라이브러리

# 1. 페이지 설정
st.set_page_config(page_title="LMM Benchmark Dashboard", layout="wide")
st.title("📊 대규모 LMM 검색 모델 성능 평가")
st.markdown("여러 텍스트 임베딩 모델(S-BERT, BERT, CLIP, KoCLIP 등)의 검색 성능과 결과를 한눈에 비교합니다.")

# 2. 데이터 로드
RESULT_FILE = "evaluation_results.json"
IMAGE_DIR = "images"

if not os.path.exists(RESULT_FILE):
    st.error(f"'{RESULT_FILE}' 파일이 없습니다. 먼저 search_engine.py를 실행하여 평가 데이터를 생성해주세요.")
    st.stop()

with open(RESULT_FILE, 'r', encoding='utf-8') as f:
    results = json.load(f)

# 데이터프레임 생성 및 정렬
metrics_data = []
for name, data in results.items():
    metrics_data.append({
        "Model": name,
        "Type": data.get("type", "N/A"),
        "Latency (ms)": data["avg_latency_ms"],
        "Indexing (s)": data["indexing_time_sec"]
    })

df = pd.DataFrame(metrics_data).sort_values(by="Latency (ms)", ascending=True)


# 3. 성능 지표 시각화
st.header("1. 성능 지표 비교 (Speed & Efficiency)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚡ 검색 속도 (Latency)")
    # Altair를 사용한 가로 막대 그래프 + 모델별 색상 적용
    chart_latency = alt.Chart(df).mark_bar().encode(
        x=alt.X('Latency (ms)', title='평균 검색 시간 (ms)'),
        y=alt.Y('Model', sort='x', title='모델명'),  # 속도순 정렬
        color=alt.Color('Model', legend=None, scale=alt.Scale(scheme='category20')), # 다양한 색상
        tooltip=['Model', 'Latency (ms)', 'Type']
    ).properties(height=300)
    
    st.altair_chart(chart_latency, width='stretch')
    st.caption("※ 막대가 짧을수록 더 빠릅니다.")

with col2:
    st.subheader("🏗️ 인덱싱 속도 (Indexing Time)")
    # 인덱싱 속도 그래프 추가
    chart_indexing = alt.Chart(df).mark_bar().encode(
        x=alt.X('Indexing (s)', title='인덱싱 소요 시간 (초)'),
        y=alt.Y('Model', sort='x', title=''), # Y축 라벨 숨김 (왼쪽과 동일하므로)
        color=alt.Color('Model', legend=None, scale=alt.Scale(scheme='category20')),
        tooltip=['Model', 'Indexing (s)', 'Type']
    ).properties(height=300)
    
    st.altair_chart(chart_indexing, width='stretch')
    st.caption("※ 데이터 벡터화 및 저장에 걸린 시간입니다.")

# 상세 데이터 테이블 (정렬됨)
with st.expander("📋 상세 수치 데이터 보기"):
    st.dataframe(df.style.background_gradient(subset=['Latency (ms)', 'Indexing (s)'], cmap='Oranges'), width='stretch')


# 4. 검색 품질 비교 
st.divider()
st.header("2. 검색 결과 품질 비교 (Search Quality)")

# 컨트롤 패널
c1, c2 = st.columns([3, 1])
with c1:
    # 쿼리 선택
    sample_model = list(results.keys())[0]
    queries = [ex["query"] for ex in results[sample_model]["search_examples"]]
    selected_query = st.selectbox("🔍 비교할 검색어를 선택하세요:", queries)
with c2:
    # 몇 위까지 볼지 선택
    top_k = st.radio("보여줄 결과 개수", [1, 3, 5], index=0, horizontal=True)

st.markdown(f"### 👉 검색어: **'{selected_query}'**")

# 모델별 결과 출력 (그리드 레이아웃)
# 화면 너비에 따라 컬럼 수 자동 조정은 어렵지만, 5개씩 끊어서 보여주기
model_names = df["Model"].tolist() # 정렬된 순서대로 표시
cols_per_row = 5
rows = [model_names[i:i + cols_per_row] for i in range(0, len(model_names), cols_per_row)]

for row_models in rows:
    cols = st.columns(len(row_models))
    for idx, model_name in enumerate(row_models):
        with cols[idx]:
            with st.container(border=True):
                st.subheader(f"{model_name}")
                
                # 해당 쿼리의 결과 찾기
                model_result = results[model_name]
                query_data = next((item for item in model_result["search_examples"] if item["query"] == selected_query), None)
                
                if query_data:
                    # 선택한 개수(top_k)만큼 반복 출력
                    for i, res in enumerate(query_data["results"][:top_k]):
                        if i > 0: st.divider() # 결과 사이 구분선
                        
                        img_path = os.path.join(IMAGE_DIR, res['image'])
                        
                        # 순위와 상품명 표시
                        st.markdown(f"**{res['rank']}위**")
                        
                        if os.path.exists(img_path):
                            st.image(img_path, width='stretch')
                        else:
                            st.warning(f"이미지 없음\n({res['image']})")
                        
                        # 상품명 및 유사도
                        st.caption(f"{res['product_name']}")
                        st.caption(f"유사도: {res['score']:.4f}")