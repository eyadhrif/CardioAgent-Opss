"""Train/validation tuning; evaluate test only after selecting the champion."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from audio_engine.contract import VERSION, features
from training.data import write_manifest

NAMES = ("decision_tree", "svm", "cnn", "resnet50", "transformer_cnn")


def metrics(y, probability):
    predicted = np.asarray(probability) >= .5
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, predicted)),
        "recall_abnormal": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision_abnormal": float(precision_score(y, predicted, zero_division=0)),
        "f1_abnormal": float(f1_score(y, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def deep_model(name, learning_rate=1e-3):
    import tensorflow as tf
    k = tf.keras
    inputs = k.Input((96, 96, 1), name="features")
    backbone = None
    if name == "resnet50":
        # Grayscale channels are identical, so RGB/BGR reversal is unnecessary.
        # Keep ImageNet affine preprocessing inside a traced layer, not captured eager tensors.
        x = k.layers.Concatenate()([inputs, inputs, inputs])
        x = k.layers.Rescaling(255., offset=[151.061, 138.221, 131.32], name="imagenet_preprocess")(x)
        backbone = k.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
        backbone.trainable = False
        x = backbone(x, training=False)
        x = k.layers.GlobalAveragePooling2D()(x)
    else:
        x = inputs
        for channels in (16, 32, 64):
            x = k.layers.Conv2D(channels, 3, padding="same", activation="relu")(x)
            x = k.layers.MaxPooling2D()(x)
        if name == "transformer_cnn":
            # Each time step attends to the full temporal context after local convolution.
            x = k.layers.Permute((2, 1, 3))(x)
            x = k.layers.Reshape((12, 12 * 64))(x)
            x = k.layers.Dense(64)(x)
            attention = k.layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
            x = k.layers.LayerNormalization()(k.layers.Add()([x, attention]))
            feedforward = k.layers.Dense(64)(k.layers.Dense(128, activation="relu")(x))
            x = k.layers.LayerNormalization()(k.layers.Add()([x, feedforward]))
            x = k.layers.GlobalAveragePooling1D()(x)
        else:
            x = k.layers.GlobalAveragePooling2D()(x)
    x = k.layers.Dropout(.25)(x)
    outputs = k.layers.Dense(1, activation="sigmoid", name="probability_abnormal")(x)
    model = k.Model(inputs, outputs)
    model.compile(optimizer=k.optimizers.Adam(learning_rate), loss="binary_crossentropy",
        metrics=[k.metrics.Recall(name="recall")])
    return model, backbone


def export_model(model, kind, destination, sample):
    import onnxruntime as ort
    if kind == "mfcc":
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        classifier = model.steps[-1][1] if hasattr(model, "steps") else model
        converted = convert_sklearn(model, initial_types=[("features", FloatTensorType([None, 84]))],
            options={id(classifier): {"zipmap": False}}, target_opset=17)
        destination.write_bytes(converted.SerializeToString())
        expected = model.predict_proba(sample)[:, 1]
    else:
        import tensorflow as tf
        import tf2onnx
        function = tf.function(lambda x: model(x, training=False), input_signature=[tf.TensorSpec([None, 96, 96, 1], tf.float32, name="features")])
        tf2onnx.convert.from_function(function, input_signature=function.input_signature,
            opset=17, output_path=str(destination))
        expected = model.predict(sample, verbose=0).ravel()
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: sample.astype(np.float32)})
    observed = np.asarray(outputs[1])[:, 1] if kind == "mfcc" else np.asarray(outputs[0]).ravel()
    error = float(np.max(np.abs(expected - observed)))
    if error > 1e-4:
        destination.unlink(missing_ok=True)
        raise ValueError(f"ONNX parity failed: {error}")
    return error


def run(args):
    import mlflow
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    vault = Path(args.vault)
    manifest_path = vault / "manifests" / "audio.json"
    if args.rebuild_manifest or not manifest_path.exists():
        records = write_manifest(Path(args.dataset), manifest_path, args.seed)
    else:
        records = json.loads(manifest_path.read_text())["records"]
    if args.max_files:
        # Deterministic subset within EACH source/class/split; never recompute split on the subset.
        selected = []
        for source in sorted({r["source"] for r in records}):
            for partition in ("train", "validation", "test"):
                for label in (0, 1):
                    selected.extend([r for r in records if r["source"] == source and r["split"] == partition and r["label"] == label][:args.max_files])
        records = selected
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("cardioagent-audio-benchmark")
    cache = {}
    def load(kind):
        if kind not in cache:
            cache[kind] = np.stack([features(Path(r["path"]).read_bytes(), kind) for r in records])
        return cache[kind]
    y = np.array([r["label"] for r in records])
    train, val, test = [np.array([r["split"] == s for r in records]) for s in ("train", "validation", "test")]
    for mask in (train, val, test):
        if set(y[mask]) != {0, 1}:
            raise ValueError("Both classes required in every split")
    reports = []
    models = {}
    with mlflow.start_run(run_name="five-model-benchmark" if len(args.models) == 5 else "partial-benchmark") as parent:
        run_path = vault / "runs" / parent.info.run_id
        run_path.mkdir(parents=True, exist_ok=True)
        mlflow.log_params({"seed": args.seed, "feature_contract": VERSION, "subset": bool(args.max_files),
            "samples": len(records), "selection": "validation recall, then specificity", "models": ",".join(args.models)})
        used_manifest = run_path / "manifest.json"
        used_manifest.write_text(json.dumps(records, indent=2))
        mlflow.log_artifact(str(used_manifest))
        for name in args.models:
            kind = "mfcc" if name in {"decision_tree", "svm"} else "mel"
            X = load(kind)
            configurations = ([{"max_depth": d, "min_samples_leaf": 3} for d in (3, 6, 12, None)] if name == "decision_tree" else
                [{"C": c, "gamma": g} for c in (.1, 1., 10.) for g in ("scale", .01)] if name == "svm" else
                [{"learning_rate": rate} for rate in (1e-3, 1e-4)])
            best = None
            with mlflow.start_run(run_name=name, nested=True) as child:
                for config in configurations[:args.trials]:
                    with mlflow.start_run(run_name="tuning", nested=True):
                        started = time.perf_counter()
                        if kind == "mfcc":
                            model = (DecisionTreeClassifier(**config, class_weight="balanced", random_state=args.seed)
                                if name == "decision_tree" else make_pipeline(StandardScaler(), SVC(**config,
                                    class_weight="balanced", probability=True, random_state=args.seed)))
                            model.fit(X[train], y[train])
                            probability = model.predict_proba(X[val])[:, 1]
                        else:
                            import tensorflow as tf
                            tf.keras.utils.set_random_seed(args.seed)
                            model, backbone = deep_model(name, **config)
                            class LossLogger(tf.keras.callbacks.Callback):
                                def on_epoch_end(self, epoch, logs=None):
                                    mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)
                            callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True), LossLogger()]
                            counts = np.bincount(y[train], minlength=2)
                            weights = {i: float(sum(counts) / (2 * counts[i])) for i in (0, 1)}
                            model.fit(X[train], y[train], validation_data=(X[val], y[val]), epochs=args.epochs,
                                batch_size=args.batch_size, class_weight=weights, callbacks=callbacks, verbose=2)
                            if backbone is not None:
                                # Fine-tune only top residual stage; frozen batch norm, lower learning rate.
                                backbone.trainable = True
                                for layer in backbone.layers:
                                    layer.trainable = layer.name.startswith("conv5_") and not isinstance(layer, tf.keras.layers.BatchNormalization)
                                model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"] / 100),
                                    loss="binary_crossentropy", metrics=[tf.keras.metrics.Recall(name="recall")])
                                model.fit(X[train], y[train], validation_data=(X[val], y[val]), epochs=args.epochs,
                                    batch_size=args.batch_size, class_weight=weights, callbacks=callbacks, verbose=2)
                            probability = model.predict(X[val], verbose=0).ravel()
                        score = metrics(y[val], probability)
                        mlflow.log_params(config)
                        mlflow.log_metrics({f"validation_{k}": v for k, v in score.items()})
                        mlflow.log_metric("fit_seconds", time.perf_counter() - started)
                        rank = (score["specificity"] >= args.min_specificity, score["recall_abnormal"], score["specificity"], score["auprc"])
                        if best is None or rank > best[0]:
                            best = (rank, model, config, score)
                _, model, config, score = best
                path = run_path / f"{name}.onnx"
                try:
                    import resource
                    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                except ImportError:  # Windows has no `resource`; record unknown, never a guess.
                    peak_rss_mb = None
                parity = export_model(model, kind, path, X[val][:min(16, sum(val))])
                metadata = {"name": name, "feature_kind": kind, "feature_contract": VERSION,
                    "labels": ["normal", "abnormal"], "threshold": .5, "run_id": child.info.run_id,
                    "parent_run_id": parent.info.run_id, "validation": score, "parameters": config,
                    "minimum_specificity": args.min_specificity,
                    "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "subset": bool(args.max_files), "onnx_parity_max_abs": parity,
                    "onnx_bytes": path.stat().st_size,
                    "process_peak_rss_mb": peak_rss_mb}
                path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
                mlflow.log_artifact(str(path))
                mlflow.log_artifact(str(path.with_suffix(".json")))
                mlflow.log_metrics({f"validation_{k}": v for k, v in score.items()})
                reports.append(metadata)
                models[name] = (model, X)
        eligible = [r for r in reports if r["validation"]["specificity"] >= args.min_specificity]
        if not eligible:
            raise ValueError("No model meets the validation specificity gate; no champion promoted")
        winner = max(eligible, key=lambda r: (r["validation"]["recall_abnormal"], r["validation"]["specificity"], r["validation"]["auprc"]))
        # Test reported for every model only AFTER all tuning/champion selection is fixed.
        for report in reports:
            model, X = models[report["name"]]
            started = time.perf_counter()
            probability = model.predict_proba(X[test])[:, 1] if report["feature_kind"] == "mfcc" else model.predict(X[test], verbose=0).ravel()
            report["test"] = metrics(y[test], probability)
            report["test_seconds_per_recording"] = (time.perf_counter() - started) / sum(test)
        result = {"run_id": parent.info.run_id, "champion": winner["name"], "complete_five_model_benchmark": set(args.models) == set(NAMES),
            "subset": bool(args.max_files), "models": reports}
        report_path = run_path / "benchmark.json"
        import onnx
        champion_onnx = onnx.load(str(run_path / f"{winner['name']}.onnx"))
        try:  # MLflow 3 names the logged model with `name`.
            registered = mlflow.onnx.log_model(
                champion_onnx, name="champion", registered_model_name="CardioAgentAudio")
        except TypeError:  # MLflow 2 uses `artifact_path` for the same location.
            registered = mlflow.onnx.log_model(
                champion_onnx, artifact_path="champion", registered_model_name="CardioAgentAudio")
        client = mlflow.MlflowClient()
        version = registered.registered_model_version
        result["registered_model"] = {"name": "CardioAgentAudio", "version": version}
        report_path.write_text(json.dumps(result, indent=2))
        mlflow.log_artifact(str(report_path))
        mlflow.log_metrics({f"champion_test_{k}": v for k, v in winner["test"].items()})
        if args.activate:
            activate(run_path / f"{winner['name']}.onnx", vault)
            if version:
                client.set_registered_model_alias("CardioAgentAudio", "champion", str(version))
        print(json.dumps(result, indent=2))
        return result


def verify_artifact(path: Path):
    import onnxruntime as ort
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["feature_contract"] != VERSION or metadata["onnx_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Artifact hash/feature contract mismatch")
    if not 0 <= metadata["onnx_parity_max_abs"] <= 1e-4:
        raise ValueError("Artifact has not passed ONNX probability parity")
    if metadata["validation"]["specificity"] < metadata["minimum_specificity"]:
        raise ValueError("Artifact fails the validation specificity gate")
    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return metadata


def activate(path: Path, vault: Path):
    verify_artifact(path)
    target = vault / "models"
    target.mkdir(parents=True, exist_ok=True)
    (target / "registry.json").unlink(missing_ok=True)
    # Stage both artifacts, then replace; consumers reject a mismatched pair during a concurrent restart.
    shutil.copy2(path, target / "cardio_model.onnx.tmp")
    shutil.copy2(path.with_suffix(".json"), target / "cardio_model.json.tmp")
    os.replace(target / "cardio_model.onnx.tmp", target / "cardio_model.onnx")
    os.replace(target / "cardio_model.json.tmp", target / "cardio_model.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets")
    parser.add_argument("--vault", default="vault")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///vault/mlflow.db"))
    parser.add_argument("--models", nargs="+", choices=NAMES, default=list(NAMES))
    parser.add_argument("--max-files", type=int, default=0, help="Per source/class/split smoke subset; 0 uses all data")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--min-specificity", type=float, default=.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.epochs < 1 or args.max_files < 0 or args.batch_size < 1:
        parser.error("Trials, epochs and batch size must be positive; max-files nonnegative")
    if not 0 < args.min_specificity <= 1:
        parser.error("Minimum specificity must be in (0, 1]")
    run(args)

if __name__ == "__main__":
    main()
