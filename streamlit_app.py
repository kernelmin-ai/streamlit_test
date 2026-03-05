import streamlit as st
import pickle
import numpy as np
from src.model.movie_predictor import MoviePredictor

# 페이지 설정
st.set_page_config(page_title="Movie Rating Predictor", page_icon="🎬")

st.title("🎬 Movie Rating Prediction")
st.write("영화 정보를 입력하면 예상 평점을 예측합니다.")

# 모델 로드
@st.cache_resource
def load_model():
    with open("models/movie_predictor/E10_T260304180143.pkl", "rb") as f:
        ck = pickle.load(f)

    model = MoviePredictor(**ck["model_params"])
    model.load_state_dict(ck["model_state_dict"])
    scaler = ck["scaler"]

    return model, scaler

model, scaler = load_model()

# 입력 UI
st.sidebar.header("Movie Features")

budget = st.sidebar.slider("Budget", 1_000_000, 200_000_000, 50_000_000)
popularity = st.sidebar.slider("Popularity", 0.0, 100.0, 50.0)
runtime = st.sidebar.slider("Runtime (minutes)", 60, 200, 120)
vote_count = st.sidebar.slider("Vote Count", 0, 10000, 1000)

st.subheader("Input Features")

st.write(
    {
        "Budget": budget,
        "Popularity": popularity,
        "Runtime": runtime,
        "Vote Count": vote_count,
    }
)

# 예측 버튼
if st.button("Predict Rating"):

    x = np.array([[budget, popularity, runtime, vote_count]])
    x = scaler.transform(x)

    pred = model.forward(x)
    rating = float(pred[0][0])

    st.success(f"🎥 Predicted Movie Rating: **{rating:.2f}**")

    # 간단한 해석
    if rating > 7:
        st.info("🔥 높은 평점을 받을 가능성이 있습니다.")
    elif rating > 5:
        st.info("🙂 평균적인 영화일 가능성이 있습니다.")
    else:
        st.warning("😅 낮은 평점이 예상됩니다.")