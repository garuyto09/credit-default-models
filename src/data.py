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


def _find_header_row(path: Path, max_scan: int = 5) -> int:
    """Locate the row holding the real column names.

    The published workbook carries a generic X1..X23 line alongside the actual
    names, and which one lands on top has varied between copies of the file
    circulating online. Rather than hard-coding an offset that silently produces
    columns called X1..X23, find the row that names a column we know must exist.
    """
    probe = pd.read_excel(path, header=None, nrows=max_scan)
    for i in range(len(probe)):
        if probe.iloc[i].astype(str).str.strip().eq("LIMIT_BAL").any():
            return i
    raise ValueError(
        f"No header row containing LIMIT_BAL in the first {max_scan} rows of {path}"
    )


def load() -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix and the binary default target."""
    path = download()
    df = pd.read_excel(path, header=_find_header_row(path))
    df.columns = [str(c).strip() for c in df.columns]

    df = df.rename(columns={"default payment next month": TARGET})
    df = df.drop(columns=["ID"], errors="ignore")

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
