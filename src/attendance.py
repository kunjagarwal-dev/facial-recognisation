import csv
from datetime import datetime
from pathlib import Path
import pandas as pd


def log_attendance(name, log_path="attendance_log.csv"):
    path = Path(log_path)
    needs_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(["Name", "Timestamp"])
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


def already_logged_today(name, log_path="attendance_log.csv"):
    path = Path(log_path)
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return False
    if df.empty:
        return False

    today = datetime.now().strftime("%Y-%m-%d")
    df["date"] = df["Timestamp"].str.split(" ").str[0]
    return ((df["Name"] == name) & (df["date"] == today)).any()