import requests


def main(oldest_year: int):
    for year in range(oldest_year, 2024):
        download_data(year)


def download_data(year: int):
    url = f"https://www.football-data.co.uk/mmz4281/{year - 2000}{year - 1999}/E0.csv"
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"matches_{year}_{year + 1}.csv", "wb") as file:
            file.write(response.content)
    else:
        print(f"Failed to download data for year {year}")
