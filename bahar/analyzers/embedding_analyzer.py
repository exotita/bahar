"""
Embedding Analyzer for Distributional Semantics.

This module provides comprehensive embedding and distributional semantics analysis:
- Multiple embedding models (Word2Vec, GloVe, spaCy vectors)
- Semantic space analysis (clustering, dimensionality reduction)
- Semantic similarity and neighbor analysis
- Contextual vs. static embeddings comparison

Academic research focused with statistical metrics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import spacy
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap


@dataclass
class WordEmbedding:
    """Word embedding information."""

    word: str
    vector: np.ndarray
    model_name: str
    dimensions: int
    norm: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export (excluding large vector)."""
        return {
            "word": self.word,
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "norm": float(self.norm),
            "vector_sample": self.vector[:5].tolist() if len(self.vector) >= 5 else self.vector.tolist(),
        }


@dataclass
class SemanticNeighbor:
    """Semantic neighbor information."""

    target_word: str
    neighbor_word: str
    similarity: float
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "target_word": self.target_word,
            "neighbor_word": self.neighbor_word,
            "similarity": float(self.similarity),
            "model_name": self.model_name,
        }


@dataclass
class SemanticCluster:
    """Semantic cluster information."""

    cluster_id: int
    words: list[str]
    centroid: np.ndarray
    cohesion: float  # Intra-cluster similarity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "cluster_id": self.cluster_id,
            "words": self.words,
            "size": len(self.words),
            "cohesion": float(self.cohesion),
        }


@dataclass
class EmbeddingFeatures:
    """Complete embedding analysis results."""

    text: str
    embeddings: list[WordEmbedding] = field(default_factory=list)
    semantic_neighbors: list[SemanticNeighbor] = field(default_factory=list)
    clusters: list[SemanticCluster] = field(default_factory=list)

    # Statistics
    avg_vector_norm: float = 0.0
    vector_dimensionality: int = 0
    semantic_density: float = 0.0  # Average similarity between words
    cluster_quality: float = 0.0  # Silhouette score

    # Dimensionality reduction results
    pca_variance_explained: list[float] = field(default_factory=list)
    effective_dimensions: int = 0

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            "text_length": len(self.text),
            "word_count": len(self.text.split()),
            "embedded_words": len(self.embeddings),
            "vector_dimensionality": self.vector_dimensionality,
            "avg_vector_norm": self.avg_vector_norm,
            "semantic_density": self.semantic_density,
            "num_clusters": len(self.clusters),
            "cluster_quality": self.cluster_quality,
            "effective_dimensions": self.effective_dimensions,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "text": self.text,
            "embeddings": [emb.to_dict() for emb in self.embeddings[:20]],  # Limit for size
            "semantic_neighbors": [n.to_dict() for n in self.semantic_neighbors],
            "clusters": [c.to_dict() for c in self.clusters],
            "statistics": {
                "avg_vector_norm": self.avg_vector_norm,
                "vector_dimensionality": self.vector_dimensionality,
                "semantic_density": self.semantic_density,
                "cluster_quality": self.cluster_quality,
                "effective_dimensions": self.effective_dimensions,
                "pca_variance_explained": [float(v) for v in self.pca_variance_explained[:10]],
            }
        }


class EmbeddingAnalyzer:
    """
    Analyzer for distributional semantics and embeddings.

    Provides:
    - Word embeddings from spaCy (Word2Vec-style)
    - Semantic neighbor analysis
    - Semantic space clustering
    - Dimensionality reduction and analysis
    - Semantic density metrics

    Academic research focused with comprehensive statistics.
    """

    def __init__(self, language: str = "english", model_name: str | None = None) -> None:
        """
        Initialize embedding analyzer.

        Args:
            language: Language code (english, dutch, persian)
            model_name: Specific spaCy model to use (must have vectors)
        """
        self.language = language
        self.model_name = model_name
        self._nlp: Any = None
        self._loaded = False

    def load_model(self) -> None:
        """Load spaCy model with word vectors."""
        if self._loaded:
            return

        # Use large models with vectors
        if self.model_name:
            model_to_load = self.model_name
        else:
            model_map = {
                "english": "en_core_web_lg",
                "dutch": "nl_core_news_lg",
                "persian": "en_core_web_lg",  # Fallback
            }
            model_to_load = model_map.get(self.language, "en_core_web_lg")

        try:
            self._nlp = spacy.load(model_to_load)
        except OSError as e:
            raise RuntimeError(
                f"spaCy model '{model_to_load}' not found or doesn't have vectors. "
                f"Install with: python -m spacy download {model_to_load}"
            ) from e

        # Verify model has vectors
        if not self._nlp.vocab.vectors.shape[0]:
            raise RuntimeError(
                f"Model '{model_to_load}' doesn't have word vectors. "
                f"Use a large model (e.g., en_core_web_lg) with vectors."
            )

        self._loaded = True

    def analyze(self, text: str, top_k_neighbors: int = 5, n_clusters: int = 3) -> EmbeddingFeatures:
        """
        Perform comprehensive embedding analysis.

        Args:
            text: Input text to analyze
            top_k_neighbors: Number of semantic neighbors to find per word
            n_clusters: Number of clusters for semantic space analysis

        Returns:
            EmbeddingFeatures with complete analysis
        """
        if not self._loaded:
            self.load_model()

        # Process text
        doc = self._nlp(text)

        # Extract embeddings for content words
        embeddings = self._extract_embeddings(doc)

        if not embeddings:
            # Return empty results if no embeddings
            return EmbeddingFeatures(text=text)

        # Find semantic neighbors
        semantic_neighbors = self._find_semantic_neighbors(embeddings, top_k_neighbors)

        # Cluster semantic space
        clusters = self._cluster_semantic_space(embeddings, n_clusters)

        # Compute statistics
        avg_norm = self._compute_avg_vector_norm(embeddings)
        dimensionality = embeddings[0].dimensions if embeddings else 0
        semantic_density = self._compute_semantic_density(embeddings)
        cluster_quality = self._compute_cluster_quality(embeddings, clusters)

        # Dimensionality reduction analysis
        pca_variance, effective_dims = self._analyze_dimensionality(embeddings)

        return EmbeddingFeatures(
            text=text,
            embeddings=embeddings,
            semantic_neighbors=semantic_neighbors,
            clusters=clusters,
            avg_vector_norm=avg_norm,
            vector_dimensionality=dimensionality,
            semantic_density=semantic_density,
            cluster_quality=cluster_quality,
            pca_variance_explained=pca_variance,
            effective_dimensions=effective_dims,
        )

    def _extract_embeddings(self, doc: Any) -> list[WordEmbedding]:
        """Extract word embeddings from document."""
        embeddings = []

        for token in doc:
            # Skip non-content words and words without vectors
            if not token.is_alpha or token.is_stop or not token.has_vector:
                continue

            vector = token.vector
            norm = float(np.linalg.norm(vector))

            embedding = WordEmbedding(
                word=token.text.lower(),
                vector=vector,
                model_name=self._nlp.meta.get('name', 'unknown'),
                dimensions=len(vector),
                norm=norm,
            )
            embeddings.append(embedding)

        return embeddings

    def _find_semantic_neighbors(
        self,
        embeddings: list[WordEmbedding],
        top_k: int
    ) -> list[SemanticNeighbor]:
        """Find semantic neighbors for each word."""
        neighbors = []

        # Limit to first few words for performance
        for embedding in embeddings[:10]:
            # Find most similar words in vocabulary
            word_token = self._nlp(embedding.word)[0]

            if not word_token.has_vector:
                continue

            # Get most similar words from spaCy vocabulary
            ms_words = self._nlp.vocab.vectors.most_similar(
                word_token.vector.reshape(1, -1),
                n=top_k + 1  # +1 because word itself will be included
            )

            # Convert to neighbor objects
            for i, (vector_id, similarity) in enumerate(zip(ms_words[0][0], ms_words[1][0])):
                if i == 0:  # Skip the word itself
                    continue

                neighbor_word = self._nlp.vocab.strings[vector_id]

                neighbor = SemanticNeighbor(
                    target_word=embedding.word,
                    neighbor_word=neighbor_word,
                    similarity=float(similarity),
                    model_name=embedding.model_name,
                )
                neighbors.append(neighbor)

        return neighbors

    def _cluster_semantic_space(
        self,
        embeddings: list[WordEmbedding],
        n_clusters: int
    ) -> list[SemanticCluster]:
        """Cluster words in semantic space."""
        if len(embeddings) < n_clusters:
            n_clusters = max(1, len(embeddings) // 2)

        if len(embeddings) < 2:
            return []

        # Extract vectors
        vectors = np.array([emb.vector for emb in embeddings])
        words = [emb.word for emb in embeddings]

        # K-means clustering
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)
            centroids = kmeans.cluster_centers_
        except Exception:
            return []

        # Create cluster objects
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_words = [words[i] for i, label in enumerate(labels) if label == cluster_id]
            cluster_vectors = vectors[labels == cluster_id]

            if len(cluster_vectors) == 0:
                continue

            # Compute cohesion (average pairwise similarity within cluster)
            cohesion = self._compute_cluster_cohesion(cluster_vectors)

            cluster = SemanticCluster(
                cluster_id=cluster_id,
                words=cluster_words,
                centroid=centroids[cluster_id],
                cohesion=cohesion,
            )
            clusters.append(cluster)

        return clusters

    def _compute_cluster_cohesion(self, vectors: np.ndarray) -> float:
        """Compute average pairwise cosine similarity within cluster."""
        if len(vectors) < 2:
            return 1.0

        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = vectors / norms

        # Compute pairwise similarities
        similarities = np.dot(normalized, normalized.T)

        # Average of upper triangle (excluding diagonal)
        n = len(vectors)
        upper_triangle = similarities[np.triu_indices(n, k=1)]

        return float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 1.0

    def _compute_avg_vector_norm(self, embeddings: list[WordEmbedding]) -> float:
        """Compute average vector norm."""
        if not embeddings:
            return 0.0
        return sum(emb.norm for emb in embeddings) / len(embeddings)

    def _compute_semantic_density(self, embeddings: list[WordEmbedding]) -> float:
        """Compute average pairwise similarity between all words."""
        if len(embeddings) < 2:
            return 0.0

        vectors = np.array([emb.vector for emb in embeddings])

        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = vectors / norms

        # Compute pairwise similarities
        similarities = np.dot(normalized, normalized.T)

        # Average of upper triangle (excluding diagonal)
        n = len(vectors)
        upper_triangle = similarities[np.triu_indices(n, k=1)]

        return float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 0.0

    def _compute_cluster_quality(
        self,
        embeddings: list[WordEmbedding],
        clusters: list[SemanticCluster]
    ) -> float:
        """Compute cluster quality using silhouette score."""
        if len(embeddings) < 2 or len(clusters) < 2:
            return 0.0

        vectors = np.array([emb.vector for emb in embeddings])

        # Create labels from clusters
        labels = np.zeros(len(embeddings), dtype=int)
        for cluster in clusters:
            for i, emb in enumerate(embeddings):
                if emb.word in cluster.words:
                    labels[i] = cluster.cluster_id

        try:
            score = silhouette_score(vectors, labels)
            return float(score)
        except Exception:
            return 0.0

    def _analyze_dimensionality(
        self,
        embeddings: list[WordEmbedding]
    ) -> tuple[list[float], int]:
        """Analyze effective dimensionality using PCA."""
        if len(embeddings) < 2:
            return [], 0

        vectors = np.array([emb.vector for emb in embeddings])

        # PCA to analyze variance
        n_components = min(50, len(vectors), vectors.shape[1])

        try:
            pca = PCA(n_components=n_components)
            pca.fit(vectors)

            variance_explained = pca.explained_variance_ratio_.tolist()

            # Find effective dimensions (cumulative variance > 0.95)
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            effective_dims = int(np.argmax(cumsum >= 0.95) + 1)

            return variance_explained, effective_dims
        except Exception:
            return [], 0

    def get_word_similarity(self, word1: str, word2: str) -> float | None:
        """
        Get semantic similarity between two words.

        Args:
            word1: First word
            word2: Second word

        Returns:
            Cosine similarity (0-1) or None if words not in vocabulary
        """
        if not self._loaded:
            self.load_model()

        token1 = self._nlp(word1)[0]
        token2 = self._nlp(word2)[0]

        if not token1.has_vector or not token2.has_vector:
            return None

        return float(token1.similarity(token2))

    def get_semantic_neighbors(self, word: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Get most similar words to a given word.

        Args:
            word: Target word
            top_k: Number of neighbors to return

        Returns:
            List of (word, similarity) tuples
        """
        if not self._loaded:
            self.load_model()

        token = self._nlp(word)[0]

        if not token.has_vector:
            return []

        # Get most similar words
        ms_words = self._nlp.vocab.vectors.most_similar(
            token.vector.reshape(1, -1),
            n=top_k + 1
        )

        neighbors = []
        for i, (vector_id, similarity) in enumerate(zip(ms_words[0][0], ms_words[1][0])):
            if i == 0:  # Skip the word itself
                continue

            neighbor_word = self._nlp.vocab.strings[vector_id]
            neighbors.append((neighbor_word, float(similarity)))

        return neighbors

