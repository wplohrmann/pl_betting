import requests


def main(oldest_year: int) -> None:
    for year in range(oldest_year, 2025):
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
