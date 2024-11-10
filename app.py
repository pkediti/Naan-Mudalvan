from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('spotify_songs.csv')

features = ['danceability', 'energy', 'loudness', 'tempo', 'valence', 'track_popularity']
X = df[features]
X = X.dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

def recommend_songs(song_name, num_recommendations=5):
    song_data = df[df['track_name'].str.lower() == song_name.lower()]
    if song_data.empty:
        return "Song not found in the dataset. Please try another song."

    cluster = song_data['cluster'].values[0]
    recommendations = df[df['cluster'] == cluster].sample(num_recommendations)
    return recommendations[['track_name', 'track_artist']].to_dict(orient='records')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        song_name = request.form["song_name"]
        recommendations = recommend_songs(song_name)
        if isinstance(recommendations, str):
            return render_template("index.html", error_message=recommendations)
        return render_template("recommendations.html", recommendations=recommendations)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
