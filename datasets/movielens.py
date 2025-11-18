from tqdm import tqdm
import os
import zipfile
import torch
import pandas as pd
import requests
from torch.utils.data import Dataset, DataLoader


class MovieLensDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


def GetMovielens(numclients: int = 500, seed: int = 0):
    """
    Loads and prepares the MovieLens-1M dataset for federated learning.
    """
    datadir = "./data/ml-1m"
    zippath = "./data/ml-1m.zip"

    if not os.path.exists(datadir):
        os.makedirs("./data", exist_ok=True)
        print("Downloading MovieLens-1M dataset...")
        dataurl = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"

        resp = requests.get(dataurl, stream=True)
        totalsize = int(resp.headers.get("content-length", 0))
        blocksize = 1024

        progbar = tqdm(total=totalsize, unit="iB", unit_scale=True)
        with open(zippath, "wb") as fout:
            for chunk in resp.iter_content(blocksize):
                progbar.update(len(chunk))
                fout.write(chunk)
        progbar.close()

        with zipfile.ZipFile(zippath, "r") as zipref:
            zipref.extractall("./data/")
        os.remove(zippath)
        print("Download complete.")

    ratingdf = pd.read_csv(
        os.path.join(datadir, "ratings.dat"),
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )

    ratingdf.userId = ratingdf.userId - 1
    moviemap = {mid: i for i, mid in enumerate(ratingdf.movieId.unique())}
    ratingdf.movieId = ratingdf.movieId.map(moviemap)

    numusers = ratingdf.userId.max() + 1
    numitems = ratingdf.movieId.max() + 1

    ratingdf["rating"] = 1.0

    alluserids = sorted(ratingdf.userId.unique())
    selusers = alluserids[:numclients]

    clientdata = []
    testdata = {"users": [], "items": [], "ratings": []}

    print(f"Processing {len(selusers)} clients...")
    for userid in tqdm(selusers):
        userdf = ratingdf[ratingdf.userId == userid].sort_values("timestamp")
        if len(userdf) < 20:
            continue

        traindf = userdf.iloc[:-10]
        testdf = userdf.iloc[-10:]

        testdata["users"].extend(testdf.userId.values)
        testdata["items"].extend(testdf.movieId.values)
        testdata["ratings"].extend(testdf.rating.values)

        trainds = MovieLensDataset(
            traindf.userId.values,
            traindf.movieId.values,
            traindf.rating.values,
        )
        loader = DataLoader(trainds, batch_size=128, shuffle=True)
        clientdata.append({"loader": loader, "group_id": 0})

    testds = MovieLensDataset(
        testdata["users"],
        testdata["items"],
        testdata["ratings"],
    )
    testloader = DataLoader(testds, batch_size=1024, shuffle=False)
    return clientdata, testloader, {"num_users": numusers, "num_items": numitems}
