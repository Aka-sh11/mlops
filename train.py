import argparse
import os
import json
from datetime import datetime
import tempfile
import joblib
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def download_from_gcs(bucket_name, blob_path, local_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)


def upload_to_gcs(bucket_name, local_path, dest_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_path)
    blob.upload_from_filename(local_path)


def main(args):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    artifact_prefix = f"week_2/output/artifacts/{ts}"
    os.makedirs("tmp", exist_ok=True)

    local_csv = os.path.join("tmp", "iris.csv")
    print(
        f"Downloading gs://{args.data_bucket}/{args.data_path} -> {local_csv}")
    download_from_gcs(args.data_bucket, args.data_path, local_csv)

    df = pd.read_csv(local_csv)
    # try to auto-detect label column
    if 'species' in df.columns:
        label_col = 'species'
    else:
        label_col = df.columns[-1]
    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "timestamp": ts,
        "accuracy": float(acc),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }

    # save model locally then upload
    local_model = os.path.join("tmp", "model.joblib")
    joblib.dump(pipeline, local_model)
    upload_to_gcs(args.output_bucket, local_model,
                  f"{artifact_prefix}/model.joblib")
    print(
        f"Uploaded model to gs://{args.output_bucket}/{artifact_prefix}/model.joblib")

    # save metrics
    local_metrics = os.path.join("tmp", "metrics.json")
    with open(local_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    upload_to_gcs(args.output_bucket, local_metrics,
                  f"{artifact_prefix}/metrics.json")

    # save predictions
    preds_df = X_test.copy()
    preds_df['true'] = y_test.values
    preds_df['pred'] = y_pred
    local_preds = os.path.join("tmp", "preds.csv")
    preds_df.to_csv(local_preds, index=False)
    upload_to_gcs(args.output_bucket, local_preds,
                  f"{artifact_prefix}/preds.csv")

    print("Done. Artifacts available at gs://{}/{}/".format(args.output_bucket, artifact_prefix))
    print("Metrics:", metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-bucket", required=True)
    p.add_argument("--data-path", required=True,
                   help="path inside bucket to CSV (e.g. data//raw/iris.csv)")
    p.add_argument("--output-bucket", required=True)
    args = p.parse_args()
    main(args)
