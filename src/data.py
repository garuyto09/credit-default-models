"""Dataset loading for the UCI Default of Credit Card Clients study.

Source: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques
for the predictive accuracy of probability of default of credit card clients.
Expert Systems with Applications, 36(2), 2473-2480.
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

30,000 Taiwanese credit card accounts observed in 2005, with six months of repayment
history, billing and payment amounts. The target is default in the following month.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "default_of_credit_card_clients.xls"

TARGET = "default"


def download(force: bool = False) -> Path:
    """Fetch the dataset from the UCI archive. Cached after the first call."""
    if RAW_PATH.exists() and not force:
        return RAW_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {URL} ...")
    with urllib.request.urlopen(URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
        # The zip holds a single .xls
        name = next(n for n in archive.namelist() if n.endswith(".xls"))
        RAW_PATH.write_bytes(archive.read(name))

    print(f"Saved to {RAW_PATH}")
    return RAW_PATH


def load() -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix and the binary default target.

    The published file carries two header rows — a generic X1..X23 line above the
    real column names — so the second row is the one to read as the header.
    """
    path = download()
    df = pd.read_excel(path, header=1)

    df = df.rename(columns={"default payment next month": TARGET})
    df = df.drop(columns=["ID"])

    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


def describe_features() -> dict[str, str]:
    """Plain-language meaning of the less obvious columns."""
    return {
        "LIMIT_BAL": "Credit limit granted (NT$), including family supplementary cards",
        "SEX": "1 = male, 2 = female",
        "EDUCATION": "1 = graduate school, 2 = university, 3 = high school, 4 = other",
        "MARRIAGE": "1 = married, 2 = single, 3 = other",
        "AGE": "Age in years",
        "PAY_0..PAY_6": "Repayment status, most recent month first. -1 = paid duly, "
        "1..9 = months of payment delay",
        "BILL_AMT1..6": "Bill statement amount (NT$), most recent month first",
        "PAY_AMT1..6": "Amount paid (NT$), most recent month first",
        TARGET: "1 if the account defaulted in the following month, else 0",
    }
