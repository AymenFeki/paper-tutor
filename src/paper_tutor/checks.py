"""Data-quality checks on the corpus (report only, nothing is deleted): see scripts/13_data_checks.py."""

import re

import numpy as np

# Words in a title, or phrases in an abstract, that mark reviews, surveys, tutorials and lecture notes (#19)
TITLE_PATTERNS = [
    r"\breview\b", r"\bsurvey\b", r"\btutorial\b", r"\boverview\b", r"\bintroduction to\b", r"\bprimer\b",
    r"\blecture notes?\b", r"\bguide\b", r"\bhandbook\b", r"\btextbook\b",
]
ABSTRACT_PATTERNS = [
    r"\bthis (paper|article|chapter) (reviews|surveys)\b", r"\bwe (review|survey)\b",
    r"\bthis (review|survey|tutorial)\b",
    r"\b(a|this) (comprehensive|systematic|critical) (review|survey|overview)\b", r"\blecture notes\b",
]


def review_signals(title, abstract):
    """The review/tutorial patterns found in a paper's title and abstract (empty list if none)."""
    found = [p for p in TITLE_PATTERNS if re.search(p, (title or "").lower())]
    found += [p for p in ABSTRACT_PATTERNS if re.search(p, (abstract or "").lower())]
    return found


def cosine_rows(a, b):
    """Cosine similarity of each row of a with the same row of b."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(axis=1)


def lowest_share(scores, share):
    """Indices of the lowest `share` of the scores (at least one), lowest first."""
    n = max(1, round(len(scores) * share))
    return list(np.argsort(scores)[:n])


def area_centroids(vectors, areas):
    """Mean direction of each area's paper vectors, as unit vectors: {area: vector}."""
    areas = np.array(areas)
    centroids = {}
    for area in sorted(set(areas)):
        mean = vectors[areas == area].mean(axis=0)
        centroids[area] = mean / np.linalg.norm(mean)
    return centroids


def nearest_area(vector, centroids):
    """The area whose centroid is most similar to the vector, and all the similarities."""
    vector = vector / np.linalg.norm(vector)
    similarities = {area: float(vector @ centroid) for area, centroid in centroids.items()}
    return max(similarities, key=similarities.get), similarities
