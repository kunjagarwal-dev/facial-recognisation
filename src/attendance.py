import csv
from datetime import datetime
from pathlib import Path
import pandas as pd


def log_attendance(name, log_path="attendance_log.csv"):
    file_exists = Path(log_path).exists()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Timestamp"])
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


def already_logged_today(name, log_path="attendance_log.csv"):
    if not Path(log_path).exists():
        return False

    df = pd.read_csv(log_path)
    if df.empty:
        return False

    today = datetime.now().strftime("%Y-%m-%d")
    df["date"] = df["Timestamp"].str.split(" ").str[0]
    return ((df["Name"] == name) & (df["date"] == today)).any()