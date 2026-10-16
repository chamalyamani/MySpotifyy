"""Memory-conscious recommenders for the Million Song Dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


class MySpotify:
    REQUIRED_GENRES = (
        "Rock", "Rap", "Jazz", "Electronic", "Pop", "Blues", "Country",
        "Reggae", "New Age",
    )

    # These fallback terms are used only when the requested word is absent
    # from musiXmatch's 5,000-word vocabulary.
    COLLECTION_TERMS = {
        "happiness": ("happy", "happiness", "joy", "joyful", "smile", "laugh", "fun"),
        "loneliness": ("loneliness", "lonely", "alone", "solitude", "sad", "tears", "cry", "miss"),
    }

    def __init__(self, songs, lyrics=None, metadata=None, vocabulary=None):
        required = {"user_id", "song_id", "play_count"}
        missing = required.difference(songs.columns)
        if missing:
            raise ValueError(f"Listening data is missing columns: {sorted(missing)}")

        self.songs = songs
        self.lyrics = lyrics
        self.metadata = metadata
        self.vocabulary = set(vocabulary or ())
        self.user_encoder = None
        self.song_encoder = None
        self.train_data = None
        self.test_data = None
        self.user_item_matrix = None
        self.test_user_item_matrix = None
        self.item_user_matrix = None
        self.user_model = None
        self.item_model = None

    def _require_metadata(self, columns):
        if self.metadata is None:
            raise ValueError("Track metadata has not been provided")
        missing = set(columns).difference(self.metadata.columns)
        if missing:
            raise ValueError(f"Track metadata is missing columns: {sorted(missing)}")

    def _popularity_ranking(self, interactions, limit):
        self._require_metadata(["song_id", "artist", "title"])
        popularity = (
            interactions.groupby("song_id", observed=True)["play_count"]
            .sum()
            .rename("play_count")
            .reset_index()
        )
        metadata = self.metadata[["song_id", "artist", "title"]].drop_duplicates("song_id")
        result = (
            popularity.merge(metadata, on="song_id", how="left", validate="one_to_one")
            .sort_values("play_count", ascending=False)
            .head(limit)[["artist", "title", "play_count"]]
            .reset_index(drop=True)
        )
        result.index = np.arange(1, len(result) + 1)
        result.index.name = "rank"
        return result

    def top_250(self):
        """Return the 250 most-played tracks."""
        return self._popularity_ranking(self.songs, 250)

    def top_100_by_genre(self, genre):
        """Return the 100 most-played tracks for a major genre."""
        self._require_metadata(["song_id", "major_genre", "artist", "title"])
        available = set(self.metadata["major_genre"].dropna().unique())
        if genre not in available:
            raise ValueError(f"Genre {genre!r} not found. Available genres: {sorted(available)}")
        song_ids = self.metadata.loc[self.metadata["major_genre"] == genre, "song_id"].drop_duplicates()
        genre_interactions = self.songs[self.songs["song_id"].isin(song_ids)]
        return self._popularity_ranking(genre_interactions, 100)

    def load_mxm_dataset(self, filepath):
        """Load sparse musiXmatch lyrics and return the rows and vocabulary."""
        vocabulary = []
        rows = []
        with Path(filepath).open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("ls#"):
                    continue
                if line.startswith("%"):
                    vocabulary = line[1:].split(",")
                    continue
                if not vocabulary:
                    raise ValueError(f"Missing musiXmatch vocabulary before line {line_number}")

                parts = line.split(",")
                if len(parts) < 2:
                    continue
                counts = {}
                try:
                    for item in parts[2:]:
                        index, count = item.split(":", 1)
                        word = vocabulary[int(index) - 1]
                        counts[word] = int(count)
                except (IndexError, ValueError) as error:
                    raise ValueError(f"Malformed musiXmatch row at line {line_number}") from error
                rows.append({
                    "track_id": parts[0],
                    "mxm_track_id": parts[1],
                    "word_counts": counts,
                })

        self.vocabulary = set(vocabulary)
        return pd.DataFrame(rows), vocabulary

    def collections(self, keyword, threshold=1):
        """Return up to 50 popular songs containing a keyword or its fallback terms."""
        if self.lyrics is None:
            raise ValueError("Lyrics have not been provided")
        self._require_metadata(["track_id", "song_id", "artist", "title"])
        keyword = keyword.lower().strip()
        terms = [keyword] if keyword in self.vocabulary else list(self.COLLECTION_TERMS.get(keyword, (keyword,)))
        terms = [term for term in terms if term in self.vocabulary]
        if not terms:
            raise ValueError(f"No usable lyric words found for {keyword!r}")

        lyrics = self.lyrics[["track_id", "word_counts"]].copy()
        lyrics["keyword_count"] = lyrics["word_counts"].map(
            lambda counts: sum(counts.get(term, 0) for term in terms)
        )
        lyrics = lyrics[lyrics["keyword_count"] >= threshold]

        popularity = (
            self.songs.groupby("song_id", observed=True)["play_count"]
            .sum().rename("play_count").reset_index()
        )
        track_map = self.metadata[["track_id", "song_id"]].drop_duplicates("track_id")
        song_metadata = self.metadata[["song_id", "artist", "title"]].drop_duplicates("song_id")
        result = (
            lyrics.merge(track_map, on="track_id", how="inner")
            .merge(popularity, on="song_id", how="inner")
            .merge(song_metadata, on="song_id", how="left", validate="many_to_one")
            .sort_values(["play_count", "keyword_count"], ascending=False)
            .drop_duplicates("song_id")
            .head(50)[["artist", "title", "play_count"]]
            .reset_index(drop=True)
        )
        result.index = np.arange(1, len(result) + 1)
        result.index.name = "rank"
        return result

    @staticmethod
    def _category_encoder(series):
        encoder = LabelEncoder()
        if isinstance(series.dtype, pd.CategoricalDtype):
            encoder.classes_ = np.asarray(series.cat.categories, dtype=object)
        else:
            encoder.fit(series)
        return encoder

    def train_test_split(self, test_size=0.2, random_state=42):
        """Split listening events into train and test sets while retaining IDs."""
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        self.train_data, self.test_data = sklearn_train_test_split(
            self.songs, test_size=test_size, random_state=random_state
        )
        self.user_encoder = self._category_encoder(self.songs["user_id"])
        self.song_encoder = self._category_encoder(self.songs["song_id"])
        print("Train:", self.train_data.shape)
        print("Test :", self.test_data.shape)

    def build_user_item_matrix(self):
        """Build compact sparse train and test matrices from the split data."""
        if self.train_data is None or self.test_data is None:
            raise ValueError("Call train_test_split() before building the matrix")
        if not isinstance(self.songs["user_id"].dtype, pd.CategoricalDtype):
            # This path supports smaller external dataframes. The notebook reads
            # IDs as categoricals to avoid materializing tens of millions of strings.
            self.user_encoder.fit(self.songs["user_id"])
            self.song_encoder.fit(self.songs["song_id"])
            train_users = self.user_encoder.transform(self.train_data["user_id"])
            train_songs = self.song_encoder.transform(self.train_data["song_id"])
            test_users = self.user_encoder.transform(self.test_data["user_id"])
            test_songs = self.song_encoder.transform(self.test_data["song_id"])
        else:
            train_users = self.train_data["user_id"].cat.codes.to_numpy(dtype=np.int32)
            train_songs = self.train_data["song_id"].cat.codes.to_numpy(dtype=np.int32)
            test_users = self.test_data["user_id"].cat.codes.to_numpy(dtype=np.int32)
            test_songs = self.test_data["song_id"].cat.codes.to_numpy(dtype=np.int32)

        shape = (len(self.user_encoder.classes_), len(self.song_encoder.classes_))
        train_values = np.log1p(self.train_data["play_count"].to_numpy(dtype=np.float32))
        test_values = np.ones(len(self.test_data), dtype=np.float32)
        self.user_item_matrix = csr_matrix((train_values, (train_users, train_songs)), shape=shape)
        self.test_user_item_matrix = csr_matrix((test_values, (test_users, test_songs)), shape=shape)
        self.user_item_matrix.sum_duplicates()
        self.test_user_item_matrix.sum_duplicates()
        print("Matrix shape:", self.user_item_matrix.shape)
        print("Non-zero entries:", self.user_item_matrix.nnz)

    def fit_user_model(self):
        if self.user_item_matrix is None:
            raise ValueError("Build the user-item matrix before fitting the model")
        self.user_model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.user_model.fit(self.user_item_matrix)

    def fit_item_model(self):
        if self.user_item_matrix is None:
            raise ValueError("Build the user-item matrix before fitting the model")
        self.item_user_matrix = self.user_item_matrix.T.tocsr()
        self.item_model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.item_model.fit(self.item_user_matrix)

    def similar_users(self, user_id, n_neighbors=50):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        if self.user_model is None:
            raise ValueError("Fit the user model before requesting similar users")
        if user_id not in self.user_encoder.classes_:
            raise ValueError(f"Unknown user: {user_id}")
        user_index = self.user_encoder.transform([user_id])[0]
        row = self.user_item_matrix[user_index]
        if row.nnz == 0:
            raise ValueError("User has no listening history in the training data")
        count = min(n_neighbors + 1, self.user_item_matrix.shape[0])
        distances, indices = self.user_model.kneighbors(row, n_neighbors=count)
        neighbors = [(i, 1.0 - d) for i, d in zip(indices[0], distances[0]) if i != user_index]
        neighbors = neighbors[:n_neighbors]
        if not neighbors:
            raise ValueError("No other users are available for comparison")
        ids = np.asarray([item[0] for item in neighbors], dtype=np.int32)
        similarities = np.asarray([item[1] for item in neighbors], dtype=np.float32)
        return self.user_encoder.inverse_transform(ids), similarities

    def recommend_for_user(self, user_id, n_recommendations=10, n_neighbors=50):
        """Recommend tracks listened to by the user's nearest neighbors."""
        if n_recommendations < 1:
            raise ValueError("n_recommendations must be positive")
        self._require_metadata(["song_id", "artist", "title"])
        neighbors, similarities = self.similar_users(user_id, n_neighbors=n_neighbors)
        user_index = self.user_encoder.transform([user_id])[0]
        listened = set(self.user_item_matrix[user_index].indices)
        scores = {}
        for neighbor, similarity in zip(neighbors, similarities):
            neighbor_index = self.user_encoder.transform([neighbor])[0]
            row = self.user_item_matrix[neighbor_index]
            for song_index, weight in zip(row.indices, row.data):
                if song_index not in listened:
                    scores[song_index] = scores.get(song_index, 0.0) + float(similarity * weight)
        if not scores:
            return pd.DataFrame(columns=["song_id", "artist", "title", "likelihood"])

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:n_recommendations]
        song_ids = self.song_encoder.inverse_transform(np.asarray([i for i, _ in ranked], dtype=np.int32))
        recs = pd.DataFrame({"song_id": song_ids, "likelihood": [score for _, score in ranked]})
        meta = self.metadata[["song_id", "artist", "title"]].drop_duplicates("song_id")
        recs = recs.merge(meta, on="song_id", how="left", validate="one_to_one")
        recs = recs[["song_id", "artist", "title", "likelihood"]]
        recs.index = np.arange(1, len(recs) + 1)
        recs.index.name = "rank"
        return recs

    def recommend_for_track(self, song_id, n_recommendations=10):
        """Recommend tracks commonly listened to alongside the supplied track."""
        if n_recommendations < 1:
            raise ValueError("n_recommendations must be positive")
        self._require_metadata(["song_id", "artist", "title"])
        if self.item_model is None:
            raise ValueError("Fit the item model before requesting track recommendations")
        if song_id not in self.song_encoder.classes_:
            raise ValueError(f"Unknown track: {song_id}")
        song_index = self.song_encoder.transform([song_id])[0]
        row = self.item_user_matrix[song_index]
        if row.nnz == 0:
            raise ValueError(f"Track {song_id} has no listening history in the training data")
        count = min(n_recommendations + 1, self.item_user_matrix.shape[0])
        distances, indices = self.item_model.kneighbors(row, n_neighbors=count)
        pairs = [(i, 1.0 - d) for i, d in zip(indices[0], distances[0]) if i != song_index]
        pairs = pairs[:n_recommendations]
        ids = self.song_encoder.inverse_transform(np.asarray([i for i, _ in pairs], dtype=np.int32))
        result = pd.DataFrame({"song_id": ids, "likelihood": [score for _, score in pairs]})
        meta = self.metadata[["song_id", "artist", "title"]].drop_duplicates("song_id")
        result = result.merge(meta, on="song_id", how="left", validate="one_to_one")
        result = result[["song_id", "artist", "title", "likelihood"]]
        result.index = np.arange(1, len(result) + 1)
        result.index.name = "rank"
        return result

    def precision_at_k(self, k=10, sample_size=100, random_state=42):
        """Average user-level precision@k over users represented in both splits."""
        if k < 1:
            raise ValueError("k must be positive")
        if self.user_model is None:
            raise ValueError("Fit the user model before evaluating it")
        active = self.user_item_matrix.getnnz(axis=1) > 0
        held_out = self.test_user_item_matrix.getnnz(axis=1) > 0
        eligible = np.flatnonzero(active & held_out)
        if len(eligible) == 0:
            raise ValueError("No users have interactions in both train and test data")
        rng = np.random.default_rng(random_state)
        selected = rng.choice(eligible, size=min(sample_size, len(eligible)), replace=False)
        precisions = []
        for user_index in selected:
            user_id = self.user_encoder.inverse_transform([user_index])[0]
            recs = self.recommend_for_user(user_id, n_recommendations=k)
            recommended_ids = self.song_encoder.transform(recs["song_id"]) if not recs.empty else np.array([], dtype=np.int32)
            relevant = set(self.test_user_item_matrix[user_index].indices)
            precisions.append(len(set(recommended_ids) & relevant) / k)
        average = float(np.mean(precisions))
        print(f"User precision@{k} ({len(precisions)} users): {average:.4f}")
        return average

    def precision_at_k_for_tracks(self, k=10, sample_size=50, random_state=42,
                                  max_users_per_track=2000):
        """Track-level precision@k over held-out tracks of seed-track listeners."""
        if k < 1:
            raise ValueError("k must be positive")
        if self.item_model is None:
            raise ValueError("Fit the item model before evaluating it")
        train = self.user_item_matrix.tocsc()
        test = self.test_user_item_matrix
        eligible = np.flatnonzero((train.getnnz(axis=0) > 0) & (test.getnnz(axis=0) > 0))
        if len(eligible) == 0:
            raise ValueError("No tracks have interactions in both train and test data")
        rng = np.random.default_rng(random_state)
        selected = rng.choice(eligible, size=min(sample_size, len(eligible)), replace=False)
        precisions = []
        for seed_index in selected:
            users = train.indices[train.indptr[seed_index]:train.indptr[seed_index + 1]]
            if len(users) > max_users_per_track:
                users = rng.choice(users, size=max_users_per_track, replace=False)
            relevant_counts = np.asarray(test[users].getnnz(axis=0)).ravel()
            relevant = set(np.flatnonzero(relevant_counts))
            relevant.discard(seed_index)
            if not relevant:
                continue
            seed_id = self.song_encoder.inverse_transform([seed_index])[0]
            recs = self.recommend_for_track(seed_id, n_recommendations=k)
            rec_ids = self.song_encoder.transform(recs["song_id"]) if not recs.empty else np.array([], dtype=np.int32)
            precisions.append(len(set(rec_ids) & relevant) / k)
        if not precisions:
            raise ValueError("No sampled tracks had held-out co-listening events")
        average = float(np.mean(precisions))
        print(f"Track precision@{k} ({len(precisions)} tracks): {average:.4f}")
        return average
