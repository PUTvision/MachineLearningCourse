from typing import Dict, List

import csv
import json
import logging
import os

import click

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def calculate_score(predicted: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> int:
    """
    Calculate the prediction score based on matching team members and position accuracy.
    """
    score = 0
    # Define the teams and their relationships
    all_nba_teams = ["first all-nba team", "second all-nba team", "third all-nba team"]
    rookie_teams = ["first rookie all-nba team", "second rookie all-nba team"]

    def add_team_score(teams: List[str], score: int) -> int:
        for team in teams:
            predicted_team = set(predicted.get(team, []))
            ground_truth_team = set(ground_truth.get(team, []))

            correct_predictions = len(predicted_team & ground_truth_team)
            score += correct_predictions * 10

            if correct_predictions == 2:
                score += 5
            elif correct_predictions == 3:
                score += 10
            elif correct_predictions == 4:
                score += 20
            elif correct_predictions == 5:
                score += 40
        return score

    def add_wrong_key_points(
            predicted: Dict[str, List[str]], ground_truth: Dict[str, List[str]], teams: List[str], score: int
    ) -> int:
        team_count = len(teams)
        for i, team in enumerate(teams):
            predicted_team = set(predicted.get(team, []))
            ground_truth_team = set(ground_truth.get(team, []))

            for player in predicted_team:
                if player not in ground_truth_team:
                    if player in ground_truth.get(teams[(i - 1) % team_count], []):
                        score += 8
                    elif player in ground_truth.get(teams[(i + 1) % team_count], []):
                        score += 8
                    elif team_count == 3 and player in ground_truth.get(teams[(i - 2) % team_count], []):
                        score += 6
        return score

    score = add_team_score(all_nba_teams, score)
    score = add_team_score(rookie_teams, score)
    score = add_wrong_key_points(predicted, ground_truth, all_nba_teams, score)
    score = add_wrong_key_points(predicted, ground_truth, rookie_teams, score)

    return score


def normalize_name(name: str) -> str:
    """
    Normalize player names for consistent comparison.
    """
    if not isinstance(name, str):
        log.warning(f"Invalid name format: {name}")
        return name
    # Define the mapping of special characters to their normalized form
    special_chars = {
        'ć': 'c',
        'č': 'c'
    }

    for char, replacement in special_chars.items():
        name = name.replace(char, replacement)
    # Special cases
    if name == 'GG Jackson':
        name = 'GG Jackson II'

    return name


def preprocess_data(data: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Normalize player names in each team of the provided data.
    """
    for team in data:
        data[team] = [normalize_name(player) for player in data[team]]
    return data


def load_json_file(filepath: str) -> Dict[str, List[str]]:
    """
    Load and validate JSON data from a file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("JSON root is not an object")
            return preprocess_data(data)
    except (json.JSONDecodeError, ValueError) as e:
        log.error(f"Failed to load {filepath}: {e}")
        return {}
    except Exception as e:
        log.exception(f"Unexpected error loading {filepath}: {e}")
        return {}


@click.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.argument("ground_truth_file_nba", type=click.Path(exists=True))
@click.argument("ground_truth_file_mf", type=click.Path(exists=True))
@click.argument("output_csv", type=click.Path(writable=True))
def main(directory: str, ground_truth_file_nba: str, ground_truth_file_mf: str, output_csv: str) -> None:
    """
    Evaluate predictions in DIRECTORY against ground truth JSONs and save scores to OUTPUT_CSV.
    """
    ground_truth_nba = load_json_file(ground_truth_file_nba)
    ground_truth_mf = load_json_file(ground_truth_file_mf)

    if not ground_truth_nba or not ground_truth_mf:
        log.error("Aborting due to invalid ground truth files.")
        return

    results: List[List[str | int]] = []

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            log.info(f'Processing file: {filename}')
            predicted = load_json_file(filepath)

            if not predicted:
                log.warning(f"Skipping {filename} due to load/format issues.")
                continue

            score_nba = calculate_score(predicted, ground_truth_nba)
            score_mf = calculate_score(predicted, ground_truth_mf)

            log.info(f'{filename} - Score NBA: {score_nba}, Score MF: {score_mf}')
            results.append([filename, score_mf + score_nba])

    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Filename', 'Score'])
            csvwriter.writerows(results)
        log.info(f"Results written to {output_csv}")
    except Exception as e:
        log.error(f"Failed to write CSV: {e}")


if __name__ == "__main__":
    main()