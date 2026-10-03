"""Universe machinery: a set of landmark embeddings defines a local metric.

Two distances are computed for any query against universe members:

- raw:      cosine similarity in the full embedding space. Universe-independent,
            dominated by whatever structure all members share.
- relative: cosine similarity after centering on the universe mean and projecting
            onto the top principal components of the (centered) members. Centering
            removes the shared component; the PCA basis keeps only the directions
            along which *this universe* actually varies.
"""

from dataclasses import dataclass

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(a @ b / denom)


@dataclass
class Universe:
    phrases: list[str]
    vectors: np.ndarray          # (n, d) raw embeddings
    mean: np.ndarray             # (d,)
    components: np.ndarray       # (k, d) top-k principal axes
    explained: np.ndarray        # (k,) fraction of variance per kept axis
    projected: np.ndarray        # (n, k) members in universe coordinates

    @classmethod
    def build(cls, phrases: list[str], vectors: np.ndarray, variance: float = 0.90) -> "Universe":
        mean = vectors.mean(axis=0)
        centered = vectors - mean
        # SVD of centered members: rows of Vt are principal axes.
        _u, s, vt = np.linalg.svd(centered, full_matrices=False)
        var = s**2
        # Every member identical (one phrase repeated): no direction varies,
        # and dividing by zero would make every fraction NaN.
        total = var.sum()
        frac = var / total if total > 0 else np.zeros_like(var)
        k = int(np.searchsorted(np.cumsum(frac), variance) + 1)
        k = min(k, len(phrases) - 1)
        components = vt[:k]
        return cls(
            phrases=phrases,
            vectors=vectors,
            mean=mean,
            components=components,
            explained=frac[:k],
            projected=centered @ components.T,
        )

    @property
    def n_components(self) -> int:
        return self.components.shape[0]

    def project(self, vec: np.ndarray) -> np.ndarray:
        """Map a raw embedding into universe coordinates."""
        return (vec - self.mean) @ self.components.T

    def residual_norm(self, vec: np.ndarray) -> float:
        """How much of the (centered) vector lies outside the universe subspace."""
        centered = vec - self.mean
        within = self.project(vec) @ self.components
        return float(np.linalg.norm(centered - within))

    def rank(self, query_vec: np.ndarray, exclude: str | None = None) -> list[dict]:
        """Rank all members against a query, by both metrics.

        Returns one dict per member with raw/relative similarities and ranks.
        """
        q_proj = self.project(query_vec)
        rows = []
        for phrase, vec, member_proj in zip(self.phrases, self.vectors, self.projected):
            if exclude is not None and phrase == exclude:
                continue
            rows.append(
                {
                    "phrase": phrase,
                    "raw": cosine(query_vec, vec),
                    "relative": cosine(q_proj, member_proj),
                }
            )
        for metric in ("raw", "relative"):
            for rank, row in enumerate(
                sorted(rows, key=lambda r: r[metric], reverse=True), start=1
            ):
                row[f"{metric}_rank"] = rank
        rows.sort(key=lambda r: r["relative"], reverse=True)
        return rows
