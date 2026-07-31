"""
Live demo — AI Fraud Detection for Canadian Financial Systems
Interactive Gradio front-end for the CatBoost fraud-scoring model in
this repository. Built to run on Hugging Face Spaces (free tier).
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import gradio as gr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fraudkit.model import load_model, predict_proba, prepare_features  # noqa: E402
from fraudkit.features import CATEGORICAL  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "model", "artifacts.joblib")
CONFIG_PATH = os.path.join(ROOT, "config.yaml")

FEATURE_LABELS = {
    "amount": "Valor da transação",
    "customer_age": "Idade do cliente",
    "customer_tenure_months": "Tempo de relacionamento (meses)",
    "is_night": "Transação noturna",
    "merchant_category_freq": "Popularidade da categoria",
    "channel_freq": "Popularidade do canal",
    "country_freq": "Popularidade do país",
    "merchant_category": "Categoria do comerciante",
    "channel": "Canal",
    "country": "País",
}

MERCHANT_CATEGORIES = [
    "grocery",
    "ecommerce",
    "fuel",
    "travel",
    "electronics",
    "restaurant",
]
CHANNELS = ["pos", "web", "mobile", "atm"]
COUNTRIES = ["CA", "US", "MX", "GB", "FR", "IN"]


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        sys.path.insert(0, ROOT)
        from src.train import main as train_main

        train_main(CONFIG_PATH)
    return load_model(MODEL_PATH)


BUNDLE = _ensure_model()
CAT_FEATURE_INDICES = [BUNDLE.feature_cols.index(c) for c in CATEGORICAL]


def _verdict(proba: float, threshold: float):
    if proba >= threshold:
        return ("BLOQUEADA", "#dc2626", "#fef2f2", "🔴")
    if proba >= threshold * 0.5:
        return ("REVISÃO MANUAL", "#d97706", "#fffbeb", "🟡")
    return ("APROVADA", "#16a34a", "#f0fdf4", "🟢")


def _explain(bundle, row_df):
    from catboost import Pool

    X = prepare_features(bundle, row_df)
    pool = Pool(X, cat_features=CAT_FEATURE_INDICES)
    shap_vals = bundle.model.get_feature_importance(pool, type="ShapValues")[0]
    contribs = shap_vals[:-1]
    order = np.argsort(np.abs(contribs))[::-1]
    top = []
    for i in order[:4]:
        name = bundle.feature_cols[i]
        top.append((FEATURE_LABELS.get(name, name), float(contribs[i])))
    return top


def score_transaction(
    amount,
    merchant_category,
    channel,
    country,
    customer_age,
    customer_tenure_months,
    is_night,
):
    start = time.time()
    row = {
        "amount": float(amount),
        "merchant_category": merchant_category,
        "channel": channel,
        "country": country,
        "customer_age": int(customer_age),
        "customer_tenure_months": int(customer_tenure_months),
        "is_night": 1 if is_night == "Noite (00h–06h)" else 0,
    }
    df = pd.DataFrame([row])
    proba = float(predict_proba(BUNDLE, df)[0])
    latency_ms = (time.time() - start) * 1000

    label, color, bg, dot = _verdict(proba, BUNDLE.threshold)
    top_factors = _explain(BUNDLE, df)

    verdict_html = f"""
    <div style="border:1px solid {color}33;background:{bg};border-radius:14px;padding:20px 24px;text-align:center;">
      <div style="font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#6b7280;font-weight:600;">Decisão do modelo</div>
      <div style="font-size:30px;font-weight:800;color:{color};margin-top:4px;">{dot} {label}</div>
      <div style="font-size:15px;color:#374151;margin-top:6px;">Score de risco: <b>{proba*100:.1f}%</b> &nbsp;·&nbsp; limiar de bloqueio: {BUNDLE.threshold*100:.0f}%</div>
    </div>
    """

    pct = min(max(proba * 100, 0), 100)
    gauge_html = f"""
    <div style="margin-top:16px;">
      <div style="display:flex;justify-content:space-between;font-size:12px;color:#6b7280;margin-bottom:4px;">
        <span>0%</span><span>Probabilidade de fraude</span><span>100%</span>
      </div>
      <div style="height:14px;border-radius:8px;background:linear-gradient(90deg,#16a34a 0%,#d97706 50%,#dc2626 100%);position:relative;">
        <div style="position:absolute;top:-6px;left:{pct}%;transform:translateX(-50%);width:2px;height:26px;background:#111827;"></div>
      </div>
    </div>
    """

    rows = ""
    for label_txt, val in top_factors:
        risk_up = val > 0
        color_f = "#dc2626" if risk_up else "#16a34a"
        arrow = "▲ aumenta o risco" if risk_up else "▼ reduz o risco"
        width = min(abs(val) / (max(abs(v) for _, v in top_factors) or 1) * 100, 100)
        rows += f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;font-size:13px;color:#374151;">
            <span>{label_txt}</span><span style="color:{color_f};font-weight:600;">{arrow}</span>
          </div>
          <div style="height:8px;border-radius:4px;background:#e5e7eb;margin-top:3px;">
            <div style="height:8px;border-radius:4px;background:{color_f};width:{width}%;"></div>
          </div>
        </div>
        """
    factors_html = f"""
    <div style="margin-top:18px;">
      <div style="font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#6b7280;font-weight:600;margin-bottom:10px;">Principais fatores (SHAP)</div>
      {rows}
    </div>
    """

    latency_html = f"""<div style="margin-top:14px;font-size:12px;color:#9ca3af;text-align:right;">⚡ Inferência em {latency_ms:.0f} ms</div>"""

    raw_json = {
        "fraud_probability": round(proba, 6),
        "threshold": BUNDLE.threshold,
        "flagged": bool(proba >= BUNDLE.threshold),
        "decision": label,
        "top_factors": [
            {"feature": n, "shap_value": round(v, 4)} for n, v in top_factors
        ],
        "latency_ms": round(latency_ms, 1),
    }

    return verdict_html, gauge_html, factors_html, latency_html, raw_json


PRESETS = {
    "🟢 Compra do dia a dia": (65.0, "grocery", "pos", "CA", 42, 48, "Dia (06h–24h)"),
    "🟡 Caso limítrofe": (380.0, "ecommerce", "web", "US", 29, 14, "Dia (06h–24h)"),
    "🔴 Padrão de alto risco": (
        2850.0,
        "electronics",
        "mobile",
        "MX",
        23,
        1,
        "Noite (00h–06h)",
    ),
}

CUSTOM_CSS = """
#header {text-align:center; padding: 8px 0 4px 0;}
#header h1 {margin-bottom:4px;}
.gradio-container {max-width: 1100px !important; margin: auto;}
footer {display:none !important;}
"""

with gr.Blocks(
    title="AI Fraud Detection · Canada",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=CUSTOM_CSS,
) as demo:
    gr.HTML(
        """
        <div id="header">
          <h1>🇨🇦 AI Fraud Detection — Demonstração ao vivo</h1>
          <p style="color:#6b7280;max-width:720px;margin:0 auto;">
            Motor de score de fraude em tempo real (CatBoost + SHAP) treinado para o mercado financeiro canadense.
            Preencha os dados de uma transação (ou escolha um cenário pronto) e veja o modelo decidir em milissegundos.
          </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("**Cenários prontos**")
            with gr.Row():
                preset_buttons = [gr.Button(name, size="sm") for name in PRESETS]

            amount = gr.Slider(
                5, 5000, value=65, step=5, label="Valor da transação (CAD)"
            )
            merchant_category = gr.Dropdown(
                MERCHANT_CATEGORIES, value="grocery", label="Categoria do comerciante"
            )
            channel = gr.Dropdown(CHANNELS, value="pos", label="Canal")
            country = gr.Dropdown(COUNTRIES, value="CA", label="País da transação")
            customer_age = gr.Slider(18, 90, value=42, step=1, label="Idade do cliente")
            customer_tenure_months = gr.Slider(
                0, 240, value=48, step=1, label="Tempo de relacionamento (meses)"
            )
            is_night = gr.Radio(
                ["Dia (06h–24h)", "Noite (00h–06h)"],
                value="Dia (06h–24h)",
                label="Horário",
            )

            run_btn = gr.Button("🔎 Analisar transação", variant="primary", size="lg")

        with gr.Column(scale=5):
            verdict_out = gr.HTML()
            gauge_out = gr.HTML()
            factors_out = gr.HTML()
            latency_out = gr.HTML()
            with gr.Accordion(
                "Ver resposta JSON (como a API /score responde)", open=False
            ):
                json_out = gr.JSON()

    inputs = [
        amount,
        merchant_category,
        channel,
        country,
        customer_age,
        customer_tenure_months,
        is_night,
    ]
    outputs = [verdict_out, gauge_out, factors_out, latency_out, json_out]

    run_btn.click(fn=score_transaction, inputs=inputs, outputs=outputs)

    def _make_preset_fn(values):
        def _fn():
            return values

        return _fn

    for btn, name in zip(preset_buttons, PRESETS):
        btn.click(fn=_make_preset_fn(PRESETS[name]), outputs=inputs).then(
            fn=score_transaction, inputs=inputs, outputs=outputs
        )

    demo.load(fn=score_transaction, inputs=inputs, outputs=outputs)

    gr.Markdown(
        """
        ---
        <div style="font-size:12px;color:#9ca3af;text-align:center;">
        Modelo treinado em dados sintéticos para fins de demonstração · Código aberto (MIT) ·
        <a href="https://github.com/wesleyharuo/AI-Fraud-Detection-Canada" target="_blank">ver repositório no GitHub</a>
        </div>
        """
    )

if __name__ == "__main__":
    # Optional access control: set DEMO_USERNAME / DEMO_PASSWORD as Space
    # secrets to require a login before the public link works. Leave both
    # unset for an open demo (e.g. local development).
    demo_user = os.environ.get("DEMO_USERNAME")
    demo_pass = os.environ.get("DEMO_PASSWORD")
    auth = (demo_user, demo_pass) if demo_user and demo_pass else None
    auth_message = (
        "Acesso restrito — peça as credenciais a quem compartilhou este link."
        if auth
        else None
    )

    demo.launch(auth=auth, auth_message=auth_message)
