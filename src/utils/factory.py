from src.model.movie_predictor import MoviePredictor

class ModelFactory:
    _models = {
        "movie_predictor": MoviePredictor,
    }

    @classmethod
    def validate_and_get(cls, model_name: str):
        """이름을 검증하고 해당 모델 클래스를 반환"""
        name_lower = model_name.lower()
        if name_lower not in cls._models:
            valid_options = list(cls._models.keys())
            raise ValueError(
                f"Invalid model name: '{model_name}'. "
                f"Available: {valid_options}"
            )
        return cls._models[name_lower]

    @classmethod
    def create(cls, model_name: str, **kwargs):
        model_class = cls.validate_and_get(model_name)
        return model_class(**kwargs)
