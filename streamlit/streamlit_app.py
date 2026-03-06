import streamlit as st
import pickle
import numpy as np
import sys
import os
import pandas as pd
import streamlit.components.v1 as components
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.movie_predictor import MoviePredictor

TEXT = {
    "page_title": "TMDB 영화 평점 예측기",
    "page_icon": "🎬",
    "title": "🎬 영화 평점 예측 서비스",
    "desc": "영화 정보를 입력하면 예상 평점을 예측합니다.",

    "sidebar_header": "영화 특징 입력",

    "budget": "예산(만 달러)",
    "popularity": "인기도(Popularity)",
    "runtime": "상영 시간(분)",
    "vote_count": "투표 수(Vote Count)",

    "input_features": "입력값 확인",
    "predict_btn": "평점 예측하기",

    "success_prefix": "예측 평점",
    "high": "🔥지금 당장 영화관으로 달려가세요! 팝콘 필수입니다!🍿🔥",
    "mid": "🙂괜찮은 영화입니다. 시간 여유 있을 때 극장이나 OTT로 즐겨보세요.🙂",
    "low": "😅급하게 볼 필요는 없습니다. OTT에 나오면 편하게 보셔도 될 것 같아요.😅",
}

# 페이지 설정
st.set_page_config(page_title=TEXT["page_title"], page_icon=TEXT["page_icon"])

st.title(TEXT["title"])
st.write(TEXT["desc"])

# 모델 로드
@st.cache_resource
def load_model():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(
        BASE_DIR,
        "models",
        "movie_predictor",
        "E10_T260304180143.pkl"
    )

    with open(model_path, "rb") as f:
        ck = pickle.load(f)

    model = MoviePredictor(**ck["model_params"])
    model.load_state_dict(ck["model_state_dict"])
    scaler = ck["scaler"]

    return model, scaler

model, scaler = load_model()

# 입력 UI
st.sidebar.header(TEXT["sidebar_header"])

budget_man = st.sidebar.slider(TEXT["budget"], 100, 20000, 5000, step=50)
budget = budget_man * 10000
popularity = st.sidebar.slider(TEXT["popularity"], 0, 100, 50)
runtime = st.sidebar.slider(TEXT["runtime"], 60, 200, 120)
vote_count = st.sidebar.slider(TEXT["vote_count"], 0, 10000, 1000)

st.subheader("입력값 확인")

c1, c2 = st.columns(2)
c1.metric("예산", f"{budget_man:,} (만 달러)")
c2.metric("인기도", f"{popularity:d}")

c3, c4 = st.columns(2)
c3.metric("상영 시간", f"{runtime:d} (분)")
c4.metric("투표 수", f"{vote_count:,}")


# 예측 버튼
if st.button(TEXT["predict_btn"]):

    x = np.array([[budget, popularity, runtime, vote_count]])
    x = scaler.transform(x)

    pred = model.forward(x)
    rating = float(pred[0][0])

    st.success(f"🎥 {TEXT['success_prefix']}: {rating:.2f} / 10")

    if rating > 7:
        st.info(f"{TEXT['high']}")
    elif rating > 5:
        st.info(f"{TEXT['mid']}")
    else:
        st.warning(f"{TEXT['low']}")







# ================================
# Demo Movie Dataset Load
# ================================
@st.cache_data
def load_movies():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = os.path.join(
        BASE_DIR,
        "dataset",
        "movies.csv"
    )

    return pd.read_csv(data_path)


df = load_movies()


# ================================
# Random Movie Evaluation Section
# ================================

st.divider()
st.header("🎬 Random Movie Evaluation")

# ✅ 세션에서 1번만 샘플링해서 고정
if "random_movie_ids" not in st.session_state:
    st.session_state.random_movie_ids = df.sample(10, random_state=None).index.tolist()

movies = df.loc[st.session_state.random_movie_ids].copy()

# feature 추출
features = movies[["budget", "popularity", "runtime", "vote_count"]].values
features = scaler.transform(features)

# 모델 예측
pred = model.forward(features)

movies["predicted_rating"] = pred.flatten()

# 오차 계산
movies["error"] = abs(movies["vote_average"] - movies["predicted_rating"])


# 추천 태그
def opinion(row):

    if row["predicted_rating"] >= 7:
        return "🔥 Recommended"

    elif row["predicted_rating"] >= 5:
        return "🙂 Watchable"

    else:
        return "😅 Skip"


movies["opinion"] = movies.apply(opinion, axis=1)

# poster url 생성
def to_poster_url(p):
    p = "" if p is None else str(p).strip()
    if p.startswith("http://") or p.startswith("https://"):
        return p  # 이미 완전한 URL
    if p == "" or p == "0":
        return "" # 없으면 빈값
    # 혹시 /xxxx.jpg 같은 path면 CDN 붙이기
    if not p.startswith("/"):
        p = "/" + p
    return "https://image.tmdb.org/t/p/w500" + p

movies["poster_url"] = movies["poster_path"].apply(to_poster_url)


# ================================
# Poster Grid
# ================================

st.subheader("🍿Team First 극장 금일 상영작🍿")

uid = uuid.uuid4().hex
wrap_id = f"cf_wrap_{uid}"
track_id = f"cf_track_{uid}"
prev_id = f"cf_prev_{uid}"
next_id = f"cf_next_{uid}"

cards = ""
for _, row in movies.iterrows():
    poster = row["poster_url"] or ""
    title = str(row["title"]).replace('"', "&quot;")
    opinion = str(row.get("opinion", ""))
    ar = row.get("vote_average", 0)
    try:
        ar = float(ar)
    except:
        ar = 0.0
    actual_text = f"⭐평점 {ar:.1f} / 10"

    cards += f"""
      <div class="cf-card" aria-label="{title}">
        <img src="{poster}" alt="{title}">
        <div class="cf-rating">{actual_text}</div>
        <div class="cf-title">{title}</div>
      </div>
    """

html = f"""
<style>
/* 컨테이너 "안 보이게" — 배경/테두리 없음 */
.coverflow {{
  position: relative;
  width: 100%;
  padding: 12px 0 22px 0;
  background: transparent;
}}

.coverflow .stage {{
  position: relative;
  height: 420px;            /* 카드 높이 공간 */
  display: grid;
  place-items: center;
  overflow: visible;
}}

.coverflow .track {{
  position: relative;
  width: 100%;
  height: 100%;
  perspective: 1200px;      /* 3D 느낌 */
  transform-style: preserve-3d;
  overflow: visible;
}}

.cf-card {{
  position: absolute;
  top: 50%;
  left: 50%;
  width: 220px;             /* ✅ 카드 폭 (여기 조절) */
  aspect-ratio: 2 / 3;
  transform: translate(-50%, -50%);
  border-radius: 18px;
  overflow: hidden;
  background: #111;
  box-shadow: 0 18px 40px rgba(0,0,0,.45);
  transition: transform 420ms cubic-bezier(.2,.8,.2,1),
              opacity 420ms cubic-bezier(.2,.8,.2,1),
              filter 420ms cubic-bezier(.2,.8,.2,1);
  will-change: transform, opacity, filter;
}}

.cf-card img {{
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}}

.cf-title {{
  position: absolute;
  left: 0; right: 0; bottom: 0;
  padding: 12px 12px 12px 12px;
  font-size: 14px;
  font-weight: 800;
  color: #fff;
  background: linear-gradient(to top, rgba(0,0,0,.88), rgba(0,0,0,0));
  line-height: 1.2;
}}

.cf-rating {{
  position: absolute;
  top: 12px; left: 12px;
  padding: 6px 10px;
  font-size: 12px;
  font-weight: 800;
  border-radius: 999px;
  background: rgba(0,0,0,.55);
  color: #fff;
  backdrop-filter: blur(8px);
}}

/* 좌/우 버튼: 컨테이너만 보이게, 배경은 최소 */
.cf-btn {{
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 44px;
  height: 84px;
  border: 0;
  border-radius: 16px;
  cursor: pointer;
  background: rgba(0,0,0,.35);
  color: #fff;
  font-size: 24px;
  display: grid;
  place-items: center;
  backdrop-filter: blur(10px);
  transition: background .15s ease, transform .15s ease;
  z-index: 50;
}}
.cf-btn:hover {{
  background: rgba(0,0,0,.55);
  transform: translateY(-50%) scale(1.04);
}}
.cf-prev {{ left: 8px; }}
.cf-next {{ right: 8px; }}

/* (선택) 버튼 옆 페이드 — 컨테이너가 아니라 카드만 돋보이게 */
.cf-edge {{
  position:absolute;
  top:0; bottom:0;
  width:80px;
  pointer-events:none;
  z-index: 20;
}}
.cf-edge.left {{
  left:0;
  background: linear-gradient(to right, rgba(0,0,0,.55), rgba(0,0,0,0));
}}
.cf-edge.right {{
  right:0;
  background: linear-gradient(to left, rgba(0,0,0,.55), rgba(0,0,0,0));
}}

</style>

<div class="coverflow" id="{wrap_id}">
  <div class="cf-edge left"></div>
  <div class="cf-edge right"></div>

  <button class="cf-btn cf-prev" id="{prev_id}" aria-label="Previous">‹</button>
  <button class="cf-btn cf-next" id="{next_id}" aria-label="Next">›</button>

  <div class="stage">
    <div class="track" id="{track_id}">
      {cards}
    </div>
  </div>
</div>

<script>
(function() {{
  const track = document.getElementById("{track_id}");
  const prev = document.getElementById("{prev_id}");
  const next = document.getElementById("{next_id}");
  if (!track || !prev || !next) return;

  const cards = Array.from(track.querySelectorAll(".cf-card"));
  const n = cards.length;
  if (n === 0) return;

  // 현재 인덱스
  let current = 0;

  // ✅ 뒤에 깔리는 카드 개수(좌/우 몇 장 보일지)
  const depth = 3;  // 2~3 추천 (3이면 좌우 최대 3장까지 배경으로)

  function normDelta(i) {{
    // i - current를 -n/2..n/2 범위로 정규화(원형)
    let d = i - current;
    if (d > n/2) d -= n;
    if (d < -n/2) d += n;
    return d;
  }}

  function render() {{
    for (let i = 0; i < n; i++) {{
      const d = normDelta(i);

      // depth 밖은 거의 안 보이게(성능+깔끔)
      if (Math.abs(d) > depth) {{
        cards[i].style.opacity = "0";
        cards[i].style.pointerEvents = "none";
        cards[i].style.transform = "translate(-50%, -50%) translateX(" + (d * 140) + "px) scale(0.7)";
        cards[i].style.filter = "blur(6px)";
        cards[i].style.zIndex = "0";
        continue;
      }}

      // 중심 카드: 크게, 선명, 최상단
      // 옆 카드: x로 밀고, 살짝 작게, 블러+투명, 약간 회전
      const absD = Math.abs(d);
      const dir = d < 0 ? -1 : 1;

      const x = d * 160;                 // ✅ 좌우 간격 (값 키우면 더 벌어짐)
      const scale = 1 - absD * 0.14;     // ✅ 뒤로 갈수록 작아짐
      const rotY = dir * absD * 18;      // ✅ 3D 회전
      const z = 200 - absD * 80;         // ✅ 깊이감
      const opacity = 1 - absD * 0.22;   // ✅ 뒤로 갈수록 흐림
      const blur = absD * 1.8;           // ✅ 뒤로 갈수록 블러

      cards[i].style.opacity = String(Math.max(0, opacity));
      cards[i].style.pointerEvents = (absD === 0) ? "auto" : "none"; // 가운데만 클릭 가능하게
      cards[i].style.zIndex = String(100 - absD);

      cards[i].style.transform =
        "translate(-50%, -50%) " +
        "translateX(" + x + "px) " +
        "translateZ(" + z + "px) " +
        "rotateY(" + rotY + "deg) " +
        "scale(" + scale + ")";

      cards[i].style.filter = "blur(" + blur + "px)";
    }}
  }}

  function goNext() {{
    current = (current + 1) % n;
    render();
  }}
  function goPrev() {{
    current = (current - 1 + n) % n;
    render();
  }}

  next.addEventListener("click", goNext);
  prev.addEventListener("click", goPrev);

  // 키보드 지원(선택)
  window.addEventListener("keydown", (e) => {{
    if (e.key === "ArrowRight") goNext();
    if (e.key === "ArrowLeft") goPrev();
  }});

  render();
}})();
</script>
"""

components.html(html, height=460, scrolling=False)



# ================================
# Result Table
# ================================

st.subheader("🎬🏆Team First의 평가는?")

table_df = movies[
    [
        "title",
        "vote_average",
        "predicted_rating",
        "error",
        "opinion"
    ]
].rename(
    columns={
        "title": "Title",
        "vote_average": "Actual Rating",
        "predicted_rating": "Predicted Rating",
        "error": "Error",
        "opinion": "Opinion"
    }
)

st.dataframe(table_df, use_container_width=True, hide_index=True)