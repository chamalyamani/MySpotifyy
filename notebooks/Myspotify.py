import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import numpy as np

class MySpotify:
    def __init__(self, songs, lyrics=None):
        self.songs = songs
        self.lyrics = lyrics
        self.user_encoder = None
        self.song_encoder = None
        self.user_item_matrix = None

    def _popularity_ranking(self, dataframe, limit):
        """Return the most popular tracks from a dataframe."""

        song_popularity = (
            dataframe.groupby("song_id")["play_count"]
            .sum()
            .reset_index()
            .rename(columns={"play_count": "total_play_count"})
        )

        metadata = (
            dataframe[["song_id", "artist", "title"]]
            .drop_duplicates()
        )

        top_tracks = (
            song_popularity
            .merge(metadata, on="song_id", how="left")
            .sort_values("total_play_count", ascending=False)
            .head(limit)
            [["artist", "title", "total_play_count"]]
            .reset_index(drop=True)
        )

        top_tracks.index += 1

        return top_tracks

    def top_250(self):
        """Return the 250 most popular tracks."""
        return self._popularity_ranking(self.songs, 250)

    def top_100_by_genre(self, genre):
        """Return the 100 most popular tracks for a given genre."""
        available_genres = self.songs["major_genre"].dropna().unique()

        if genre not in available_genres:
            raise ValueError(
                f"Genre '{genre}' not found. Available genres: {sorted(available_genres)}"
            )
        genre_songs = self.songs[
            self.songs["major_genre"] == genre
        ]

        return self._popularity_ranking(genre_songs, 100)

    def load_mxm_dataset(self, filepath):
        """
        Load the musiXmatch dataset.

        Returns:
            lyrics_df : DataFrame
                Columns:
                - track_id
                - mxm_track_id
                - word_counts (dict)
            vocabulary : list
                List of the 5000 words ordered by index.
        """

        vocabulary = []
        data = []

        with open(filepath, "r", encoding="utf-8") as f:

            for line in f:

                # Ignore comments
                if line.startswith("#"):
                    continue

                # Vocabulary line
                if line.startswith("%"):
                    vocabulary = line[1:].strip().split(",")
                    continue

                parts = line.strip().split(",")

                track_id = parts[0]
                mxm_track_id = parts[1]

                word_counts = {}

                for item in parts[2:]:
                    idx, count = item.split(":")
                    idx = int(idx) - 1        # vocabulary starts at 1
                    count = int(count)

                    word = vocabulary[idx]
                    word_counts[word] = count

                data.append({
                    "track_id": track_id,
                    "mxm_track_id": mxm_track_id,
                    "word_counts": word_counts
                })

        lyrics_df = pd.DataFrame(data)

        return lyrics_df, vocabulary

    def collections(self, keyword, threshold=1):
        """
        Return the top 50 songs related to a given keyword.

        Parameters
        ----------
        keyword : str
            Word to search for (love, war, happiness, money...)
        threshold : int
            Minimum number of occurrences required.
        """

        # Copy the lyrics dataframe
        lyrics = self.lyrics.copy()

        # Count how many times the keyword appears
        lyrics["keyword_count"] = lyrics["word_counts"].apply(
            lambda words: words.get(keyword.lower(), 0)
        )

        # Keep only songs containing the keyword
        lyrics = lyrics[lyrics["keyword_count"] >= threshold]

        # Calculate global popularity for each song
        popularity = (
            self.songs
            .groupby("track_id")["play_count"]
            .sum()
            .reset_index()
            .rename(columns={"play_count": "total_play_count"})
        )

        # Metadata
        metadata = (
            self.songs[
                ["track_id", "artist", "title"]
            ]
            .drop_duplicates()
        )

        # Merge everything
        collection = (
            lyrics
            .merge(popularity, on="track_id")
            .merge(metadata, on="track_id")
        )

        # Rank songs
        collection = (
            collection
            .sort_values(
                ["keyword_count", "total_play_count"],
                ascending=False
            )
            .head(50)
            [
                [
                    "artist",
                    "title",
                    "keyword_count",
                    "total_play_count",
                ]
            ]
            .reset_index(drop=True)
        )

        collection.index += 1

        return collection

    def build_user_item_matrix(self):
        self.user_encoder = LabelEncoder()
        self.song_encoder = LabelEncoder()

        user_ids = self.user_encoder.fit_transform(
            self.songs["user_id"]
        ).astype(np.int32)

        song_ids = self.song_encoder.fit_transform(
            self.songs["song_id"]
        ).astype(np.int32)

        play_counts = self.songs["play_count"].astype(np.float32)

        self.user_item_matrix = csr_matrix(
            (play_counts, (user_ids, song_ids))
        )

        print("Matrix shape:", self.user_item_matrix.shape)
        print("Non-zero entries:", self.user_item_matrix.nnz)
    
    def train_test_split(self, test_size=0.2, random_state=42):
        """
        Split the listening history into training and testing sets.
        """

        self.train_data, self.test_data = train_test_split(
            self.songs,
            test_size=test_size,
            random_state=random_state
        )

        print("Train:", self.train_data.shape)
        print("Test :", self.test_data.shape)

    
    def build_user_item_matrix(self):

        self.user_encoder = LabelEncoder()
        self.song_encoder = LabelEncoder()

        user_ids = self.user_encoder.fit_transform(
            self.train_data["user_id"]
        )

        song_ids = self.song_encoder.fit_transform(
            self.train_data["song_id"]
        )

        self.user_item_matrix = csr_matrix(
            (
                self.train_data["play_count"].astype(np.float32),
                (user_ids, song_ids)
            )
        )

        print(self.user_item_matrix.shape)


    def fit_user_model(self):

        self.user_model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=11
        )

        self.user_model.fit(self.user_item_matrix)

    def similar_users(self, user_id):

        user_index = self.user_encoder.transform([user_id])[0]

        distances, indices = self.user_model.kneighbors(
            self.user_item_matrix[user_index],
            n_neighbors=11
        )

        users = self.user_encoder.inverse_transform(indices[0])

        return users[1:]

    
    def recommend_for_user(self, user_id, n_recommendations=10):
        """
        Recommend songs to a user using User-Based Collaborative Filtering.
        """

        # Similar users
        similar = self.similar_users(user_id)

        # Songs already listened to by the target user
        listened = set(
            self.songs[
                self.songs["user_id"] == user_id
            ]["song_id"]
        )

        # Songs listened to by similar users
        recommendations = (
            self.songs[
                self.songs["user_id"].isin(similar)
            ]
            .groupby("song_id")["play_count"]
            .sum()
            .reset_index()
            .rename(columns={"play_count": "score"})
        )

        # Remove songs the user already knows
        recommendations = recommendations[
            ~recommendations["song_id"].isin(listened)
        ]

        # Add artist and title
        metadata = (
            self.songs[
                ["song_id", "artist", "title"]
            ]
            .drop_duplicates()
        )

        recommendations = (
            recommendations
            .merge(metadata, on="song_id", how="left")
            .sort_values("score", ascending=False)
            .head(n_recommendations)
            [["artist", "title", "score"]]
            .reset_index(drop=True)
        )

        recommendations.index += 1

        return recommendations