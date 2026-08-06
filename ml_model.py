from indicators import engineer_features
import pandas as pd

HAS_LIGHTGBM = False
HAS_XGBOOST = False
HAS_SKLEARN = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError: pass
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError: pass
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    HAS_SKLEARN = True
except ImportError: pass

class MLPredictor:
    FEATURE_COLS = [
        'Returns', 'Returns_5d', 'Returns_10d', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_PctB', 'ATR_Pct', 'Volume_Ratio', 'SMA_Diff', 'Momentum_5', 'Momentum_10',
        'Volatility_20', 'Volatility_5', 'Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'VIX_Close', 'VIX_MA_5'
    ]

    def __init__(self):
        self.model = None
        self.model_name = "None"
        self.accuracy = 0.0
        self.feature_importance = {}
        self.is_trained = False
        self.split_idx = 0

    def _select_backend(self):
        if HAS_LIGHTGBM: return "lightgbm"
        if HAS_XGBOOST: return "xgboost"
        if HAS_SKLEARN: return "sklearn"
        return None

    def train(self, df, test_size=0.3, optimize_hyperparams=True):
        backend = self._select_backend()
        if backend is None: return {"error": "ML kütüphanesi bulunamadı."}

        feature_df = engineer_features(df)
        if len(feature_df) < 100: return {"error": f"Yetersiz veri: {len(feature_df)} satır"}

        self.split_idx = int(len(feature_df) * (1 - test_size))
        train_df = feature_df.iloc[:self.split_idx]
        test_df = feature_df.iloc[self.split_idx:]
        X_train, y_train = train_df[self.FEATURE_COLS], train_df['Target_Direction']
        X_test, y_test = test_df[self.FEATURE_COLS], test_df['Target_Direction']
        best_params = {}

        if backend == "lightgbm":
            base_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
            self.model_name = "LightGBM"
            if optimize_hyperparams and HAS_SKLEARN and len(X_train) > 200:
                param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1], 'num_leaves': [15, 31, 63], 'subsample': [0.7, 0.8, 1.0]}
                search = RandomizedSearchCV(base_model, param_grid, n_iter=8, scoring='accuracy', cv=TimeSeriesSplit(n_splits=5), random_state=42, n_jobs=-1)
                search.fit(X_train, y_train)
                self.model = search.best_estimator_
                best_params = search.best_params_
            else:
                self.model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
                self.model.fit(X_train, y_train)

        elif backend == "xgboost":
            self.model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss')
            self.model_name = "XGBoost"
            self.model.fit(X_train, y_train)
            
        else:
            self.model = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
            self.model_name = "sklearn-GradientBoosting"
            self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(self.FEATURE_COLS, self.model.feature_importances_.tolist()))
        self.is_trained = True

        return {
            "model": self.model_name, "accuracy": round(self.accuracy, 4), "train_size": len(train_df),
            "test_size": len(test_df), "best_params": best_params if best_params else None,
            "feature_importance": self.feature_importance,
            "classification_report": classification_report(y_test, y_pred, output_dict=True) if HAS_SKLEARN else None
        }

    def predict_latest(self, df):
        if not self.is_trained: return None
        feature_df = engineer_features(df, include_target=False)
        if len(feature_df) == 0: return None
        latest = feature_df[self.FEATURE_COLS].iloc[-1:]
        pred = int(self.model.predict(latest)[0])
        proba = float(self.model.predict_proba(latest)[0].max())
        return {"direction": pred, "probability": round(proba, 4)}
