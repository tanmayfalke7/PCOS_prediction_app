# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, accuracy_score
from model_utils import load_pipeline_and_meta, predict_single, batch_predict

st.set_page_config(page_title="PCOS Prediction App", layout="wide")

st.title("PCOS Prediction App — Streamlit")
st.markdown("Upload sample(s) or fill the sidebar form to predict PCOS using your trained model.")

# Load model (show status)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "pcos_model")
try:
    pipeline, meta = load_pipeline_and_meta(MODEL_DIR)
    st.sidebar.success("Model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

feature_list = meta.get('feature_list')
if feature_list is None:
    st.sidebar.warning("Feature list not available in meta. For safest results, include 'feature_list' in meta when saving model.")
else:
    st.sidebar.info(f"Model expects {len(feature_list)} features.")

# --- Sidebar: Single record input
st.sidebar.header("Single record prediction")

# If feature_list known, show inputs dynamically; otherwise a free JSON input
if feature_list:
    input_values = {}
    with st.sidebar.form("single_form"):
        st.write("Fill values (leave blank for missing / imputed)")
        # Show first 15 features grouped, but allow user to expand
        for c in feature_list:
            # Use numeric input for numeric-looking names, fallback to text
            # We attempt numeric conversion where possible
            val = st.text_input(label=c, key=f"input_{c}")
            input_values[c] = float(val) if val.strip() != "" and val.replace('.','',1).isdigit() else None
        submitted = st.form_submit_button("Predict single")
    if submitted:
        # clean None -> np.nan
        input_clean = {k: (v if v is not None else np.nan) for k,v in input_values.items()}
        res = predict_single(pipeline, meta, input_clean)
        st.markdown("**Prediction**")
        st.write(res['prediction'])
        st.markdown("**Probability (positive class)**")
        st.write(res['probability'])
        st.markdown("Input row used (after alignment):")
        st.dataframe(res['input_row'])
else:
    st.sidebar.write("Feature list not available. Use batch file upload below or provide a JSON dict.")
    json_input = st.sidebar.text_area("Paste JSON for a single input (dict feature->value)", height=200)
    if st.sidebar.button("Predict from JSON"):
        try:
            input_dict = eval(json_input)  # user-provided; in production use json.loads with validation
            res = predict_single(pipeline, meta, input_dict)
            st.write("Prediction:", res['prediction'])
            st.write("Probability:", res['probability'])
        except Exception as e:
            st.error(f"Failed to parse or predict: {e}")

# --- Main area: Batch upload & evaluation
st.header("Batch upload & Model evaluation")

uploaded = st.file_uploader("Upload CSV of samples (include columns matching training features). If you upload a labeled test CSV, include target column name in meta or specify it below.", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Uploaded DataFrame (first rows):")
    st.dataframe(df.head())
    st.sidebar.success(f"Uploaded {len(df)} rows.")
    do_preview = st.checkbox("Preview prepared input (aligned to model features)", value=True)
    # Prepare aligned DF for prediction (model_utils will add missing columns)
    preds_df = batch_predict(pipeline, meta, df.copy())
    st.subheader("Predictions (first 10 rows)")
    st.dataframe(preds_df.head(10))

    # Allow download of predictions
    csv_out = preds_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download predictions CSV", data=csv_out, file_name="pcos_predictions.csv", mime="text/csv")

    # If user indicates they have truth labels, evaluate
    st.markdown("---")
    st.subheader("Evaluate model on uploaded labelled data")
    target_col = st.text_input("If your CSV has the true label column name, enter it here (e.g., 'PCOS (Y/N)')", value=meta.get('target_col', ''))
    evaluate = st.button("Evaluate")
    if evaluate:
        if target_col not in df.columns:
            st.error(f"Target column '{target_col}' not found in uploaded CSV.")
        else:
            y_true = df[target_col].copy()
            # transform truth to 0/1 if needed:
            if y_true.dtype == object:
                y_true = y_true.astype(str).str.strip().str.lower().apply(lambda v: 1 if 'pcos' in v or 'yes' in v or v=='1' or 'true' in v else 0)
            y_pred = preds_df['prediction']
            # metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            st.metric("Accuracy", f"{acc:.4f}")
            st.metric("F1 score", f"{f1:.4f}")
            st.write("Classification report:")
            st.text(classification_report(y_true, y_pred))
            # Confusion matrix plot
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # ROC curve if probabilities available
            if 'probability' in preds_df.columns and preds_df['probability'].notnull().all():
                fpr, tpr, _ = roc_curve(y_true, preds_df['probability'])
                roc_auc = auc(fpr, tpr)
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                ax2.plot([0,1],[0,1], linestyle='--')
                ax2.set_xlabel("FPR")
                ax2.set_ylabel("TPR")
                ax2.set_title("ROC Curve")
                ax2.legend()
                st.pyplot(fig2)
            else:
                st.info("Model does not provide probabilities; ROC/AUC cannot be computed.")

# --- Sidebar: Quick actions and show feature importances
st.sidebar.markdown("---")
st.sidebar.header("Model info & diagnostics")
if st.sidebar.button("Show model summary"):
    st.sidebar.write("Model type:", type(pipeline.named_steps['clf']).__name__)
    st.sidebar.write("Meta keys:", list(meta.keys()))
    st.sidebar.write("Feature count (meta):", len(meta.get('feature_list', [])) if meta.get('feature_list') else "unknown")

# Feature importances
st.sidebar.markdown("### Feature importances (if available)")
try:
    clf = pipeline.named_steps['clf']
    if hasattr(clf, "feature_importances_"):
        # need to reconstruct preprocessor feature names
        num_names = meta.get('numeric_cols', [])
        cat_names = []
        if 'categorical_cols' in meta and meta['categorical_cols']:
            # try to read categories from onehot inside pipeline
            pre = pipeline.named_steps['preprocessor']
            if 'cat' in pre.named_transformers_:
                ohe = pre.named_transformers_['cat'].named_steps.get('onehot', None)
                if ohe is not None and hasattr(ohe, "get_feature_names_out"):
                    cat_names = list(ohe.get_feature_names_out(meta.get('categorical_cols', [])))
        feature_names = (meta.get('feature_list') or (num_names + cat_names))
        importances = clf.feature_importances_
        # align lengths check
        if len(importances) == len(feature_names):
            imp_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            st.sidebar.write(imp_df.head(20))
        else:
            st.sidebar.write("Feature importances length does not match feature names. Showing top 10 from raw importances.")
            st.sidebar.write(sorted(importances, reverse=True)[:10])
    else:
        st.sidebar.write("Model has no feature_importances_ (e.g., LogisticRegression).")
except Exception as e:
    st.sidebar.write("Error while getting feature importances:", e)

st.markdown("---")
st.markdown("Built with ❤️ — upload a labeled test CSV to measure real performance.")
