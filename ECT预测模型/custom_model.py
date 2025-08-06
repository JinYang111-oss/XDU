import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import joblib  # 用于模型持久化

class CustomModel:
    def __init__(self, weak_models=None, standardize=True):
        """
        残差堆叠集成模型
        :param weak_models: 弱模型名称列表，默认 ["SVR", "Lasso", "Tree", "Ridge"]
        :param standardize: 是否对特征做标准化
        """
        self.weak_models = weak_models or ["SVR", "Lasso", "Tree", "Ridge"]
        self.standardize = standardize
        # 训练后属性
        self.scaler_ = None
        self.ridge_features_ = None
        self.ridge_model_ = None
        self.residual_models_ = []  # 列表：[(模型名, 特征索引, 模型对象), ...]

    def _instantiate_model(self, name):
        """根据名称实例化模型"""
        if name == "Ridge":
            return Ridge(alpha=1.0, random_state=42)
        elif name == "Lasso":
            return Lasso(alpha=1.0, max_iter=10000, random_state=42)
        elif name == "SVR":
            return SVR(kernel='rbf', C=1.0)
        elif name in ("Tree", "DecisionTree"):
            return DecisionTreeRegressor(random_state=42)
        else:
            raise ValueError(f"未知模型类型：{name}")

    def _forward_feature_selection(self, X, y, model_name):
        """
        前向逐步特征选择：在当前残差 y 上选特征并训练 model_name 模型
        返回：selected_feats（特征索引列表）、fitted_model（已训练模型）
        """
        n_feats = X.shape[1]
        selected = []
        best_mse = float('inf')
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        while len(selected) < n_feats:
            improved = False
            candidate = None
            for j in range(n_feats):
                if j in selected:
                    continue
                idxs = selected + [j]
                model = self._instantiate_model(model_name)
                scores = cross_val_score(model, X[:, idxs], y,
                                         cv=cv, scoring='neg_mean_squared_error')
                mse = -scores.mean()
                if mse < best_mse:
                    best_mse = mse
                    candidate = j
                    improved = True
            if not improved:
                break
            selected.append(candidate)

        # 训练最终模型
        final = self._instantiate_model(model_name)
        final.fit(X[:, selected], y)
        final.selected_features = selected
        return selected, final

    def fit(self, X, y):
        """
        训练集成模型：
        1. 可选标准化
        2. 训练并保存 Ridge 基模型
        3. 依次对残差训练弱模型
        """
        X_train = X.astype(float)
        if self.standardize:
            self.scaler_ = StandardScaler().fit(X_train)
            X_train = self.scaler_.transform(X_train)

        # —— ① 训练 Ridge 基模型，并保存
        self.ridge_features_, self.ridge_model_ = \
            self._forward_feature_selection(X_train, y, "Ridge")
        joblib.dump(self.ridge_model_, "ridge_model.pkl")
        print(f"Ridge 模型已训练，选择特征 {self.ridge_features_}，并保存为 'ridge_model.pkl'。")

        # —— ② 计算初始残差
        base_pred = self.ridge_model_.predict(X_train[:, self.ridge_features_])
        residual = y - base_pred

        # —— ③ 依次训练弱模型
        self.residual_models_.clear()
        for name in self.weak_models:
            feats, model = self._forward_feature_selection(X_train, residual, name)
            self.residual_models_.append((name, feats, model))
            pred = model.predict(X_train[:, feats])
            residual -= pred
            print(f"{name} 模型训练完毕，选择特征 {feats}。")

        return self

    def predict(self, X):
        """
        对新数据进行预测：Ridge+各残差模型预测之和
        """
        if self.ridge_model_ is None:
            raise RuntimeError("模型未训练，请先调用 fit()！")
        X_test = X.astype(float)
        if self.standardize and self.scaler_ is not None:
            X_test = self.scaler_.transform(X_test)

        # Ridge 预测
        y_pred = self.ridge_model_.predict(X_test[:, self.ridge_features_])
        # 累加残差模型预测
        for name, feats, model in self.residual_models_:
            y_pred += model.predict(X_test[:, feats])
        return y_pred

    def score(self, X, y_true):
        """
        计算皮尔逊相关系数 r 作为评价
        """
        y_pred = self.predict(X)
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        # 若常数序列，则相关系数设为 0
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return 0.0
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        return corr
