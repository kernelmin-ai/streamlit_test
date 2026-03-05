import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.utils import project_path

class TMDBRatingDataset:
    def __init__(self, df, scaler=None):
        self.df = df
        self.features = None
        self.labels = None
        self.scaler = scaler
        self._preprocessing()

    def _preprocessing(self):
        # Feature 및 Label 분리
        # budget(예산), popularity(인기도), runtime(러닝타임), vote_count(투표수) 등을 Feature로 사용
        feature_columns = ['budget', 'popularity', 'runtime', 'vote_count']
        target_column = 'vote_average'

        self.labels = self.df[target_column].values.astype(np.float32)
        features = self.df[feature_columns].values.astype(np.float32)

        # 결측치 평균으로 채우기
        features = np.nan_to_num(features, nan=np.nanmean(features, axis=0))

        # Feature Scaling
        if self.scaler:
            self.features = self.scaler.transform(features)
        else:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features)

    @property
    def features_dim(self):
        return self.features.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def fetch_tmdb_data(api_key, n_pages=2):
    """TMDB API에서 최고 평점 영화 데이터를 추출 (Extraction)"""
    print(f"Fetchting data from TMDB API... ({n_pages} pages)")
    movies = []
    base_url = os.environ.get("TMDB_BASE_URL", "https://api.themoviedb.org/3/movie")
    
    for page in range(1, n_pages + 1):
        params = {
            "api_key": api_key,
            "language": "ko-KR",
            "region": "KR",
            "page": page
        }
        res = requests.get(f"{base_url}/popular", params=params)
        if res.status_code == 200:
            results = res.json().get("results", [])
            for m in results:
                # Top Rated 리스트에는 예산/런타임이 없으므로 세부 정보 한 번 더 Fetch (선택적)
                # 구현 편의상 이번엔 목록 데이터의 popularity, vote_count, vote_average만 먼저 씁니다.
                movies.append({
                    "id": m["id"],
                    "title": m["title"],
                    "popularity": m["popularity"],
                    "vote_count": m["vote_count"],
                    "vote_average": m["vote_average"],
                    # 임의 생성 피처 (실제 API 상세 콜을 줄이기 위함)
                    "budget": np.random.randint(1000000, 100000000), 
                    "runtime": np.random.randint(80, 180)
                })
        else:
            print(f"TMDB API Error: {res.status_code}")
            
    return pd.DataFrame(movies)


def read_dataset():
    """데이터가 없으면 API에서 새로 받고, 있으면 로컬에서 로드"""
    dataset_dir = os.path.join(project_path(), "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    tmdb_path = os.path.join(dataset_dir, "tmdb_rating.csv")
    
    if not os.path.exists(tmdb_path):
        api_key = os.environ.get("TMDB_API_KEY")
        if not api_key:
            # 팀장님이 TMDB 키 입력 전이라면 일단 테스트용 더미프레임 생성
            print("Waring: TMDB_API_KEY가 없습니다. 100개의 더미 데이터를 생성합니다.")
            df = pd.DataFrame({
                "budget": np.random.randint(1000000, 50000000, 100),
                "popularity": np.random.uniform(10.0, 100.0, 100),
                "runtime": np.random.randint(80, 180, 100),
                "vote_count": np.random.randint(100, 10000, 100),
                "vote_average": np.random.uniform(3.0, 9.0, 100)
            })
        else:
            df = fetch_tmdb_data(api_key, n_pages=5) # 약 100여개 수집
            
        df.to_csv(tmdb_path, index=False)
        print(f"Data saved to {tmdb_path}")
        return df
    else:
        print(f"Loading cached data from {tmdb_path}")
        return pd.read_csv(tmdb_path)


def split_dataset(df):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
    return train_df, val_df, test_df


def get_datasets(scaler=None):
    df = read_dataset()
    train_df, val_df, test_df = split_dataset(df)
    train_dataset = TMDBRatingDataset(train_df, scaler)
    val_dataset = TMDBRatingDataset(val_df, scaler=train_dataset.scaler)
    test_dataset = TMDBRatingDataset(test_df, scaler=train_dataset.scaler)
    return train_dataset, val_dataset, test_dataset
