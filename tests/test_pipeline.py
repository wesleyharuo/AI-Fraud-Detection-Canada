import json
import subprocess
import sys
import pathlib


def test_train_runs():
    # Ensure training runs and artifacts are produced
    root = pathlib.Path(__file__).resolve().parents[1]
    cfg = root / "config.yaml"
    subprocess.run(
        [sys.executable, str(root / "src" / "train.py"), "--config", str(cfg)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (root / "model" / "artifacts.joblib").exists()


def test_infer_cli():
    root = pathlib.Path(__file__).resolve().parents[1]
    # make sure model exists
    if not (root / "model" / "artifacts.joblib").exists():
        subprocess.run(
            [
                sys.executable,
                str(root / "src" / "train.py"),
                "--config",
                str(root / "config.yaml"),
            ],
            check=True,
        )
    ret = subprocess.run(
        [
            sys.executable,
            str(root / "src" / "infer.py"),
            "--amount",
            "300",
            "--merchant_category",
            "grocery",
            "--channel",
            "pos",
            "--country",
            "CA",
            "--customer_age",
            "30",
            "--customer_tenure_months",
            "12",
            "--is_night",
            "0",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(ret.stdout)
    assert "fraud_probability" in out and "flagged" in out
