import praw
import pandas as pd
import os
from datetime import datetime


def get_reddit_data(
    sub_redits: list[str] = None, limit: int = 50, output_file: str = None
):
    limit = int(limit)
    reddit_api = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="script::nlp_unilu",  # Can be anything like "script:my_reddit_app:v1.0"
    )
    if sub_redits is None:
        sub_reddits = [
            "Luxembourg",
            "LuxembourgGausen",
            "WomenInLuxembourg",
            "Luxembourg_memes",
            "LuxembourgHouseCrisis",
            "Amazon_Luxembourg",
            "Luxembourg_IT",
            "LuxembourgITSEC",
            "LuxembourgDramaFree",
            "GeeksLifeLuxembourg",
            "LuxembourgCity",
            "MapswithoutLuxembourg",
            "EggNogLuxembourg",
            "PokemonGoLuxembourg",
            "LuxDate",
            "Luxembourgish",
            "LUXEMBOURGCYKABLYAT",
            "datawithoutluxembourg",
            "Technoluxembourg",
        ]

    all_dfs = []

    for sub_reddit in sub_reddits:
        subreddit = reddit_api.subreddit(sub_reddit)
        df = pd.DataFrame(
            [
                [post.title, post.score, post.selftext, post.url]
                for post in subreddit.hot(limit=limit)
            ],
            columns=["Title", "Score", "Text", "URL"],
        )
        df["subreddit"] = sub_reddit
        all_dfs.append(df)

    if output_file is None:
        output_file_name = f"reddit_data_lim{limit}_subreddits_{len(sub_reddits)}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        output_file = os.path.join(
            os.path.dirname(__file__), "reddit", output_file_name
        )

    (df := pd.concat(all_dfs)).to_csv(output_file, index=False)

    return df


def data_downloader(social_media: str):
    if social_media == "reddit":
        return get_reddit_data
    elif social_media == "twitter":
        return NotImplementedError


if __name__ == "__main__":
    pass
