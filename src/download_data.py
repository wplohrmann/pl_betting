import os
from tqdm import tqdm, trange
import requests


def main(oldest_year: int) -> None:
    print("Removing existing data")
    for f in os.listdir("."):
        if f.startswith("matches_") and f.endswith(".csv"):
            os.remove(f)
    pbar = trange(oldest_year, 2025)
    for year in pbar:
        pbar.set_description(f"Downloading data for year {year}")
        download_data(year)


def download_data(year: int) -> None:
    url = f"https://www.football-data.co.uk/mmz4281/{year - 2000}{year - 1999}/E0.csv"
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"matches_{year}_{year + 1}.csv", "wb") as file:
            file.write(response.content)
    else:
        print(f"Failed to download data for year {year}")

if __name__ == "__main__":
    main(2014)
