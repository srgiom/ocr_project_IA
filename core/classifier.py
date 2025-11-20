from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


@dataclass
class OCRModel:
    model: object
    charset: str

    def predict(self, X: np.ndarray) -> List[str]:
        labels_idx = self.model.predict(X)
        # Si el modelo devuelve ya caracteres, no hacemos índice
        if isinstance(labels_idx[0], str):
            return labels_idx.tolist()
        return [self.charset[i] for i in labels_idx]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "charset": self.charset}, f)

    @staticmethod
    def load(path) -> "OCRModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return OCRModel(model=data["model"], charset=data["charset"])


def train_knn(X: np.ndarray, y: Iterable[int], n_neighbors: int = 5) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    knn.fit(X, y)
    return knn


def train_linear_svc(X: np.ndarray, y: Iterable[int]) -> LinearSVC:
    svc = LinearSVC()
    svc.fit(X, y)
    return svc
