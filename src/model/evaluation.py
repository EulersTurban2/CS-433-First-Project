import numpy as np

from src.metrics import accuracy, auroc, f1_score, precision, recall


def project(X, w, kind):
    raw = X @ w
    if kind == "prob":
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
    return raw


def metric_row(y, scores, threshold):
    pred = np.where(scores >= threshold, 1, -1)
    y01 = (y == 1).astype(int)
    return {
        "f1": float(f1_score(y, pred)),
        "precision": float(precision(y, pred)),
        "recall": float(recall(y, pred)),
        "accuracy": float(accuracy(y, pred)),
        "auroc": float(auroc(y01, scores)),
    }


def choose_threshold(y, scores, kind):
    y_pos = np.asarray(y) == 1
    if kind == "prob":
        default = 0.5
        grid = np.linspace(0.0, 1.0, 101)
    else:
        default = 0.0
        grid = np.quantile(scores, np.linspace(0.02, 0.98, 50))
        grid = np.unique(np.concatenate([[scores.min() - 1.0, 0.0, scores.max() + 1.0], grid]))
    positive = scores[:, None] >= grid[None, :]
    tp = np.sum(positive & y_pos[:, None], axis=0).astype(float)
    fp = np.sum(positive & ~y_pos[:, None], axis=0).astype(float)
    fn = np.sum(~positive & y_pos[:, None], axis=0).astype(float)
    prec = np.divide(tp, tp + fp, out=np.zeros(len(grid)), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(len(grid)), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(grid)), where=(prec + rec) > 0)
    return float(grid[int(np.argmax(f1))]), default


def f1_mean_std(record):
    vals = np.array([np.nan if fold is None else fold["f1"] for fold in record["folds"]], dtype=float)
    if not np.all(np.isfinite(vals)):
        return np.nan, np.nan
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return float(vals.mean()), std


def paired_delta(candidate, reference):
    diffs = []
    for left, right in zip(candidate["folds"], reference["folds"]):
        if left is None or right is None:
            return np.nan, np.nan
        diffs.append(left["f1"] - right["f1"])
    diffs = np.asarray(diffs, dtype=float)
    std = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    return float(diffs.mean()), std


def show(records):
    print(f"{'model':<28} {'F1':>7} {'std':>7} {'prec':>7} {'recall':>7} {'acc':>7}")
    for record in records:
        mean, std = f1_mean_std(record)
        folds = [fold for fold in record["folds"] if fold is not None]
        prec = np.mean([fold["precision"] for fold in folds]) if folds else np.nan
        rec = np.mean([fold["recall"] for fold in folds]) if folds else np.nan
        acc = np.mean([fold["accuracy"] for fold in folds]) if folds else np.nan

        def fmt(value):
            return "   nan" if not np.isfinite(value) else f"{value:7.3f}"

        print(f"{record['name']:<28} {fmt(mean)} {fmt(std)} {fmt(prec)} {fmt(rec)} {fmt(acc)}")


def fmt(value, signed=False):
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def averages(record):
    f1, spread = f1_mean_std(record)
    folds = [fold for fold in record["folds"] if fold is not None]

    def mean_of(key):
        return float(np.mean([fold[key] for fold in folds])) if folds else np.nan

    return f1, spread, mean_of("precision"), mean_of("recall"), mean_of("accuracy"), mean_of("auroc")


def metric_table(records):
    lines = [
        "| model | F1 | std | precision | recall | accuracy | AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append("| " + record["name"] + " | " + " | ".join(fmt(v) for v in averages(record)) + " |")
    return "\n".join(lines)


def ablation_row(name, delta, spread, f1, against):
    return f"| {name} | {fmt(delta, signed=True)} | {fmt(spread)} | {fmt(f1)} | {against} |"
