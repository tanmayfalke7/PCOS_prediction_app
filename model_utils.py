# model_utils.py
import os
import joblib
import json
import numpy as np
import pandas as pd

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "pcos_model")
PIPE_NAME_CANDIDATES = ["pcos_pipeline.joblib", "pcos_pipeline.pkl", "pcos_model.joblib", "pcos_pipeline"]
META_NAME_CANDIDATES = ["pcos_meta.joblib", "pcos_meta.pkl", "pcos_meta.json", "meta.json"]

def find_file(folder, candidates):
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None

def load_pipeline_and_meta(model_dir=DEFAULT_MODEL_DIR):
    """
    Loads pipeline and metadata (if available).
    Returns: pipeline, meta_dict
    meta_dict may contain keys: 'feature_list', 'numeric_cols', 'categorical_cols', 'target_col'
    """
    # Find pipeline file
    pipe_path = find_file(model_dir, PIPE_NAME_CANDIDATES)
    if pipe_path is None:
        raise FileNotFoundError(f"No pipeline file found in {model_dir}. Looked for {PIPE_NAME_CANDIDATES}")
    pipeline = joblib.load(pipe_path)

    # Find meta file
    meta_path = find_file(model_dir, META_NAME_CANDIDATES)
    meta = {}
    if meta_path:
        if meta_path.endswith(".json"):
            with open(meta_path, "r") as f:
                meta = json.load(f)
        else:
            meta = joblib.load(meta_path)

    # Try to infer feature list if not present in meta
    if 'feature_list' not in meta:
        # Try to get from training DataFrame column info (sklearn stores feature_names_in_ on estimators sometimes)
        try:
            if hasattr(pipeline, "named_steps") and 'preprocessor' in pipeline.named_steps:
                pre = pipeline.named_steps['preprocessor']
                if hasattr(pre, "feature_names_in_"):
                    meta['feature_list'] = list(pre.feature_names_in_)
                else:
                    # try estimator feature_names_in_ fallback
                    if hasattr(pipeline, "feature_names_in_"):
                        meta['feature_list'] = list(pipeline.feature_names_in_)
        except Exception:
            meta['feature_list'] = None

    return pipeline, meta

def prepare_row_dict(input_dict, feature_list):
    """
    Given a dict of input values (friendly keys), produce a DataFrame row aligned to feature_list.
    Missing features left as np.nan (ColumnTransformer will impute).
    """
    if feature_list is None:
        # if feature list unknown, simply create dataframe from input_dict
        return pd.DataFrame([input_dict])
    # Ensure keys in feature_list order
    row = {c: input_dict.get(c, np.nan) for c in feature_list}
    return pd.DataFrame([row])

def predict_single(pipeline, meta, input_dict):
    """
    Predict for a single sample (dictionary). Returns dict with prediction & probability (if available).
    """
    feature_list = meta.get('feature_list')
    row = prepare_row_dict(input_dict, feature_list)
    pred = pipeline.predict(row)[0]
    prob = None
    if hasattr(pipeline.named_steps['clf'], 'predict_proba'):
        prob = pipeline.predict_proba(row)[:,1][0]
    return {"prediction": int(pred), "probability": float(prob) if prob is not None else None, "input_row": row}

def batch_predict(pipeline, meta, df):
    """
    df: pandas DataFrame (may have subset of columns). Returns DataFrame with added 'prediction' and 'probability' columns.
    """
    feature_list = meta.get('feature_list')
    if feature_list is not None:
        # ensure all needed columns exist
        for c in feature_list:
            if c not in df.columns:
                df[c] = np.nan
        df = df[feature_list]
    # run prediction
    preds = pipeline.predict(df)
    probs = pipeline.predict_proba(df)[:,1] if hasattr(pipeline.named_steps['clf'], 'predict_proba') else np.array([None]*len(df))
    out = df.copy()
    out['prediction'] = preds.astype(int)
    out['probability'] = [float(x) if x is not None else None for x in probs]
    return out
