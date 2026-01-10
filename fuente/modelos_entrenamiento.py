from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def train_all_models(X, y):
    models = {}

    models["logistic"] = GridSearchCV(
        LogisticRegression(max_iter=1000),
        {"C": [0.1, 1, 10]},
        cv=5,
        scoring="f1"
    )

    models["rf"] = GridSearchCV(
        RandomForestClassifier(),
        {"n_estimators": [100, 200], "max_depth": [None, 10]},
        cv=5,
        scoring="f1"
    )

    models["mlp"] = GridSearchCV(
        MLPClassifier(max_iter=1000),
        {"hidden_layer_sizes": [(50,), (50,50)]},
        cv=5,
        scoring="f1"
    )

    for model in models.values():
        model.fit(X, y)

    return models
