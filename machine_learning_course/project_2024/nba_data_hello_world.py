from nba_api.stats.endpoints import playercareerstats

# Nikola Jokić
career = playercareerstats.PlayerCareerStats(player_id='203999')

# pandas data frames (optional: pip install pandas)
print(f'{career.get_data_frames()[0]}')

# json
career.get_json()

# dictionary
career.get_dict()