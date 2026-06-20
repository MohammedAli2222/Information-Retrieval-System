import os
import pickle


CACHE_DIR = "cache"


def save_cache(obj, filename):

    os.makedirs(CACHE_DIR, exist_ok=True)

    path = os.path.join(CACHE_DIR, filename)

    with open(path, "wb") as f:

        pickle.dump(obj, f)

    print(f"[CACHE SAVED] {path}")


def load_cache(filename):

    path = os.path.join(CACHE_DIR, filename)

    if os.path.exists(path):

        with open(path, "rb") as f:

            print(f"[CACHE LOADED] {path}")

            return pickle.load(f)

    return None