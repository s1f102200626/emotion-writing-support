import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

st.set_page_config(page_title="Emotion Writing Support", page_icon="💬", layout="wide")

MODEL_PATH = "Ryoseihasegawa/wrime-emotion-model"
EMOTION_NAMES = ["喜び", "悲しみ", "期待", "驚き", "怒り", "恐れ", "嫌悪", "信頼"]

# ---------- UI ----------
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
#MainMenu, footer {visibility: hidden;}
h1, h2, h3 {letter-spacing: -0.02em;}
.app-kicker {font-size:.82rem; color:#64748b; font-weight:700; letter-spacing:.08em; text-transform:uppercase;}
.app-title {font-size:2.25rem; line-height:1.2; font-weight:750; color:#0f172a; margin:.25rem 0 .35rem;}
.app-subtitle {color:#64748b; font-size:1rem; margin-bottom:1.5rem;}
.step {display:inline-block; padding:.28rem .65rem; border-radius:999px; background:#eaf2ff; color:#2457a7; font-size:.78rem; font-weight:700; margin-bottom:.35rem;}
.section-title {font-size:1.35rem; font-weight:750; color:#0f172a; margin:.1rem 0 .25rem;}
.section-desc {font-size:.92rem; color:#64748b; margin-bottom:1rem;}
.card {border:1px solid #e2e8f0; border-radius:14px; padding:1rem 1.05rem; background:#fff; margin-bottom:.75rem;}
.score-name {font-size:.82rem; color:#64748b;}
.score-value {font-size:1.55rem; font-weight:750; color:#0f172a;}
.score-level {font-size:.76rem; color:#64748b;}
.result-card {border:1px solid #dbe4ef; border-radius:14px; padding:1rem 1.1rem; background:#fff; min-height:220px;}
.small-note {font-size:.78rem; color:#64748b;}
div[data-testid="stMetric"] {border:1px solid #e2e8f0; border-radius:12px; padding:.7rem .85rem; background:white;}
div[data-testid="stButton"] > button {border-radius:9px; font-weight:650; min-height:2.65rem;}
div[data-testid="stTextArea"] textarea, div[data-testid="stSelectbox"] > div > div {border-radius:10px;}
hr {border-color:#edf1f5;}
</style>
""", unsafe_allow_html=True)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"], base_url="https://api.openai.iniad.org/api/v1")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()


def analyze_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0].detach().cpu().numpy()
    return 1 / (1 + np.exp(-logits))


def level_label(score):
    if score < 0.30:
        return "低い"
    if score < 0.60:
        return "中程度"
    return "高い"


def draw_emotion_chart(scores, title="感情強度"):
    fig = go.Figure(go.Bar(x=EMOTION_NAMES, y=scores, text=[f"{v:.2f}" for v in scores], textposition="outside"))
    fig.update_layout(title=title, height=350, margin=dict(l=25, r=20, t=55, b=30), yaxis=dict(range=[0, 1.05], title="感情強度"), xaxis=dict(title=""), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def draw_target_chart(before, target, after):
    fig = go.Figure()
    fig.add_bar(name="変換前", x=EMOTION_NAMES, y=before)
    fig.add_bar(name="目標", x=EMOTION_NAMES, y=target)
    fig.add_bar(name="結果", x=EMOTION_NAMES, y=after)
    fig.update_layout(barmode="group", height=390, margin=dict(l=25, r=20, t=45, b=30), yaxis=dict(range=[0, 1.05], title="感情強度"), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)


def call_gpt(prompt, system="あなたは日本語の文章コミュニケーションを改善するアシスタントです。"):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def context_text(audience, channel, preserve):
    keep = preserve.strip() if preserve.strip() else "特になし"
    return f"""相手: {audience}\n使用場面: {channel}\n必ず保持する内容: {keep}"""


def rewrite_preset(text, mode, audience, channel, preserve):
    modes = {
        "neutral": "特定の感情を強めず、落ち着いて自然に伝わるニュートラルな表現",
        "gentle": "相手が不快になりにくい、柔らかく配慮のある表現",
        "business": "簡潔で自然な敬語を使ったビジネス向け表現",
        "workplace": "威圧的・攻撃的・人格否定的な表現を避け、業務上必要な内容を明確に伝える職場向け表現",
    }
    prompt = f"""
次の文章を「{modes[mode]}」へ書き換えてください。

元の文章:
{text}

利用条件:
{context_text(audience, channel, preserve)}

条件:
- 元の意味・事実・依頼内容をできるだけ維持する
- 「必ず保持する内容」がある場合、日時・金額・固有名詞・要求事項を変更しない
- 相手と使用場面に自然な文体にする
- 元の文体や構成は必要なら変更し、相手と使用場面に合う文章へ再構成する
- ニュートラル指定の場合、怒り・喜びなどの感情を不必要に強調せず、過度な謝罪・称賛・煽りを避ける。ただし必要な事実や意図は削らない
- 新しい事実を勝手に追加しない
- 書き換え後の文章だけを返す
"""
    return call_gpt(prompt)


def make_direction_text(current_scores, target_scores, original_scores):
    lines = []
    for name, current, target, original in zip(EMOTION_NAMES, current_scores, target_scores, original_scores):
        change = float(target - original)
        remaining = float(target - current)
        if change > 0.05:
            lines.append(f"- {name}: 元 {original:.2f} / 現在 {current:.2f} / 目標 {target:.2f}。さらに {max(0, remaining):.2f} 上げる方向を重視")
        elif change < -0.05:
            lines.append(f"- {name}: 元 {original:.2f} / 現在 {current:.2f} / 目標 {target:.2f}。さらに {max(0, -remaining):.2f} 下げる方向を重視")
        else:
            lines.append(f"- {name}: 明示的な変更指定なし（現在 {current:.2f}）。無理に固定する必要はない")
    return "\n".join(lines)


def emotion_errors(scores, target_scores, original_scores):
    """変更幅の大きい感情を優先。参考値として8感情の通常MAEも保持する。"""
    changes = np.abs(target_scores - original_scores)
    weights = 1.0 + 5.0 * changes
    errors = np.abs(scores - target_scores)
    weighted = float(np.average(errors, weights=weights))
    mae = float(np.mean(errors))
    changed = changes > 0.05
    focus_error = float(np.mean(errors[changed])) if np.any(changed) else mae
    return weighted, mae, focus_error


def candidate_prompt(original_text, current_text, current_scores, target_scores, original_scores, audience, channel, preserve, priority):
    priorities = {
        "自然さ重視": "感情目標を明確に反映しつつ、実際に送っても違和感のない自然さを特に重視する。",
        "感情目標重視": "目標に指定された感情が読者に伝わることを最も重視し、語調・構成を大胆に変更する。",
        "意味保持重視": "重要な事実・依頼・結論を特に丁寧に保持しながら、感情表現は積極的に変更する。",
    }
    current_text_scores = "\n".join(f"- {n}: {s:.2f}" for n, s in zip(EMOTION_NAMES, current_scores))
    target_text_scores = "\n".join(f"- {n}: {s:.2f}" for n, s in zip(EMOTION_NAMES, target_scores))
    return f"""
あなたの最優先の仕事は、指定された感情が文章全体から明確に伝わるよう日本語文章を書き換えることです。
単なる敬語化・語尾変更だけで終わらせず、語彙、語調、文章構成、言い回し、文の長さを必要に応じて大胆に変更してください。

優先順位:
1. ユーザーが変更した感情を目標に近づける（変更幅が大きい感情を特に重視）
2. 元の文章の重要な事実、人物、日時、数量、依頼、結論を保持する
3. 相手・場面に合った自然で読みやすい文章にする

今回の候補方針:
{priorities[priority]}

元の文章:
{original_text}

現在の候補（前回の書き換え結果）:
{current_text}

現在の感情分析モデルの推定値（0～1、確率ではない）:
{current_text_scores}

目標値:
{target_text_scores}

感情ごとの調整指示:
{make_direction_text(current_scores, target_scores, original_scores)}

利用条件:
{context_text(audience, channel, preserve)}

感情表現の目安:
- 喜び: 嬉しさ、満足、明るい言葉
- 悲しみ: 寂しさ、残念さ、落胆
- 期待: 楽しみ、希望、前向きな見通し
- 驚き: 意外さ、予想外だった反応
- 怒り: 不満、強い問題意識、改善要求
- 恐れ: 不安、心配、慎重さ
- 嫌悪: 不快感、拒否感、距離を置く言い方
- 信頼: 安心感、敬意、協力する姿勢

重要な制約:
- 元文にない感情的な言い回しや主観的な受け止め方は、自然な範囲で追加してよい。
- ただし新しい出来事、証拠、約束、相手の意図、脅迫、人格攻撃は創作しない。
- 「必ず保持する内容」、事実関係、重要な依頼・結論は削除・変更しない。
- 元の文体や構成を保つ必要はない。感情が変わるように再構成する。
- 変更指定のない感情を無理に元の数値へ固定しなくてよい。
- 現在の候補で目標に届いていない感情は、前回より明確に表現を変える。
- 数値や感情分析について文章内で説明せず、完成した文章だけを出力する。
"""


def optimize_candidate(original_text, target_scores, audience, channel, preserve, priority, max_attempts=5, tolerance=0.10):
    current_text = original_text
    original_scores = analyze_emotion(original_text)
    best_text = None
    best_scores = None
    best_error = float("inf")
    best_mae = float("inf")
    best_focus_error = float("inf")
    history = []

    for attempt in range(1, max_attempts + 1):
        current_scores = analyze_emotion(current_text)
        prompt = candidate_prompt(original_text, current_text, current_scores, target_scores, original_scores, audience, channel, preserve, priority)
        candidate = call_gpt(prompt, "あなたは指定された感情表現を積極的に実現する日本語文章の書き換えアシスタントです。重要な事実は保持しつつ、語彙・語調・構成を大胆に調整してください。")
        scores = analyze_emotion(candidate)
        error, mae, focus_error = emotion_errors(scores, target_scores, original_scores)
        history.append({"attempt": attempt, "text": candidate, "scores": scores.copy(), "error": error, "mae": mae, "focus_error": focus_error})
        if error < best_error:
            best_text, best_scores = candidate, scores.copy()
            best_error, best_mae, best_focus_error = error, mae, focus_error
        if error <= tolerance:
            break
        current_text = candidate

    return {"label": priority, "text": best_text, "scores": best_scores, "error": best_error, "mae": best_mae, "focus_error": best_focus_error, "history": history}


def save_history(original, result, mode, audience, channel, target=None, rating=None):
    item = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "original": original,
        "result": result,
        "mode": mode,
        "audience": audience,
        "channel": channel,
        "target": target.tolist() if isinstance(target, np.ndarray) else target,
        "rating": rating,
    }
    st.session_state.setdefault("history", []).insert(0, item)
    st.session_state["history"] = st.session_state["history"][:20]


def clear_results():
    for key in ["result_text", "result_scores", "result_mode", "candidates", "selected_candidate", "custom_target", "selected_priority", "edited_scores"]:
        st.session_state.pop(key, None)


# ---------- Header ----------
st.markdown('<div class="app-kicker">Emotion Writing Support</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">感情を確認しながら、伝わり方を整える</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">文章の8感情を分析し、相手・場面・目標感情に合わせた表現を提案します。</div>', unsafe_allow_html=True)

# ---------- Quick guide ----------
st.markdown("### このアプリの使い方")
st.markdown("""
1. **文章を入力・分析**：書き換えたい文章を入力し、「文章を分析」を押します。8つの感情の推定値が表示されます。
2. **変換方法を選択**：相手や使用場面を選びます。数値を操作しない場合は「ニュートラル」などの**かんたん変換**を押してください。細かく調整したい場合は、8感情のスライダーを動かして「3つの候補を生成」を押します。
3. **結果を確認**：生成された文章と変換後の感情分析結果を確認し、必要に応じて文章を編集できます。
""")
with st.expander("感情スコアと変換方法について"):
    st.markdown("""
- 感情スコアは**WRIMEで学習したモデルによる推定値（0〜1）**です。感情の正解や確率を示すものではありません。
- **ニュートラル**は感情値を指定せず、自然で落ち着いた表現に調整します。8感情を一律に0にする機能ではありません。
- 生成結果はAIによる提案です。重要な事実や相手への伝わり方は、送信前にご自身で確認してください。
""")
st.divider()

# ---------- STEP 1 ----------
st.markdown('<span class="step">STEP 1</span><div class="section-title">文章を分析</div><div class="section-desc">まず、書き換えたい文章の現在の感情傾向を確認します。</div>', unsafe_allow_html=True)
text = st.text_area("文章", height=150, placeholder="例：この前お願いした資料なんだけど、まだ完成してないの？…", key="input_text")

if st.button("文章を分析", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("文章を入力してください。")
    else:
        with st.spinner("分析しています…"):
            st.session_state["original_text"] = text.strip()
            st.session_state["original_scores"] = analyze_emotion(text.strip())
            clear_results()
            for i, score in enumerate(st.session_state["original_scores"]):
                st.session_state[f"target_{i}"] = float(round(float(score) / 0.05) * 0.05)

if "original_scores" in st.session_state:
    original_scores = st.session_state["original_scores"]
    original_text = st.session_state["original_text"]

    st.markdown("### 分析結果")
    cols = st.columns(4)
    for i, (name, score) in enumerate(zip(EMOTION_NAMES, original_scores)):
        with cols[i % 4]:
            st.metric(name, f"{score:.2f}", help=f"目安: {level_label(float(score))}")
            st.caption(level_label(float(score)))
    with st.expander("グラフで確認", expanded=True):
        draw_emotion_chart(original_scores, "入力文章の感情強度")

    st.divider()

    # ---------- STEP 2 ----------
    st.markdown('<span class="step">STEP 2</span><div class="section-title">変換条件を設定</div><div class="section-desc">誰に、どこで送る文章かを指定してから、変換方法を選びます。</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        audience = st.selectbox("相手", ["指定なし", "友人", "恋人・パートナー", "家族", "上司", "同僚", "部下", "取引先", "顧客", "その他"])
    with c2:
        channel = st.selectbox("使用場面", ["指定なし", "LINE・DM", "メール", "ビジネスチャット", "SNS投稿", "対面で伝える文章", "その他"])

    preserve = st.text_area("必ず残したい内容（任意）", height=90, placeholder="例：今日中に提出してほしい / 遅れる場合は事前に連絡してほしい")
    st.caption("日時・金額・依頼内容など、AIに変更してほしくない情報を書いてください。")

    st.markdown("#### かんたん変換")
    p1, p2, p3, p4 = st.columns(4)
    preset = None
    with p1:
        if st.button("ニュートラル", use_container_width=True): preset = ("neutral", "ニュートラル")
    with p2:
        if st.button("優しい表現", use_container_width=True): preset = ("gentle", "優しい表現")
    with p3:
        if st.button("ビジネス向け", use_container_width=True): preset = ("business", "ビジネス向け")
    with p4:
        if st.button("職場向け表現", use_container_width=True): preset = ("workplace", "職場向け表現")
    st.caption("ニュートラルは感情の数値を設定せず、相手・使用場面に合った落ち着いた文章へ変換します。")

    if preset:
        with st.spinner("文章を整えています…"):
            result = rewrite_preset(original_text, preset[0], audience, channel, preserve)
            scores = analyze_emotion(result)
        st.session_state["result_text"] = result
        st.session_state["result_scores"] = scores
        st.session_state["result_mode"] = preset[1]
        st.session_state.pop("candidates", None)
        st.session_state.pop("custom_target", None)
        save_history(original_text, result, preset[1], audience, channel)

    st.markdown("#### 感情を細かく調整")
    st.caption("スライダーは現在値から始まります。大きく動かした感情ほど、生成と候補選択で重視します。")

    target_scores = []
    for row in range(4):
        cols = st.columns(2)
        for j in range(2):
            i = row * 2 + j
            name = EMOTION_NAMES[i]
            current = float(original_scores[i])
            with cols[j]:
                st.markdown(f"**{name}**　現在 {current:.2f} → 目標")
                target = st.slider(name, 0.0, 1.0, float(round(current, 2)), 0.05, key=f"target_{i}", label_visibility="collapsed")
                diff = target - current
                if diff > 0.025:
                    st.caption(f"上げる　+{diff:.2f}")
                elif diff < -0.025:
                    st.caption(f"下げる　{diff:.2f}")
                else:
                    st.caption("ほぼ維持")
                target_scores.append(target)
    target_scores = np.array(target_scores, dtype=np.float32)

    with st.expander("詳細設定"):
        max_attempts = st.slider("1候補あたりの最大調整回数", 1, 5, 5)
        tolerance = st.slider("終了判定の重み付き誤差", 0.05, 0.25, 0.10, 0.01)
        st.caption("変更幅の大きい感情を優先した誤差がこの値以下になった場合、反復を終了します。")

    if st.button("3つの候補を生成", type="primary", use_container_width=True):
        with st.spinner("3つの方針で文章を生成し、感情を再分析しています…"):
            candidates = []
            for priority in ["自然さ重視", "感情目標重視", "意味保持重視"]:
                candidates.append(optimize_candidate(original_text, target_scores, audience, channel, preserve, priority, max_attempts, tolerance))
        st.session_state["candidates"] = candidates
        st.session_state["custom_target"] = target_scores.copy()
        st.session_state.pop("result_text", None)

# ---------- STEP 3 / candidates ----------
if st.session_state.get("candidates"):
    st.divider()
    st.markdown('<span class="step">STEP 3</span><div class="section-title">生成結果を選択</div><div class="section-desc">方針の異なる3候補から、実際に使いたい文章を選べます。</div>', unsafe_allow_html=True)
    candidates = st.session_state["candidates"]
    target = st.session_state["custom_target"]
    cols = st.columns(3)
    for i, cand in enumerate(candidates):
        with cols[i]:
            st.markdown(f"#### {cand['label']}")
            st.write(cand["text"])
            st.metric("変更幅を重視した誤差", f"{cand['error']:.3f}")
            st.caption(f"8感情の通常MAE: {cand['mae']:.3f} / 変更した感情のMAE: {cand['focus_error']:.3f}")
            if st.button("この文章を選択", key=f"choose_{i}", use_container_width=True):
                st.session_state["result_text"] = cand["text"]
                st.session_state["result_scores"] = cand["scores"]
                st.session_state["result_mode"] = cand["label"]
                st.session_state["selected_priority"] = cand["label"]
                save_history(st.session_state["original_text"], cand["text"], cand["label"], audience, channel, target)
                st.rerun()
            with st.expander("調整履歴"):
                for h in cand["history"]:
                    st.markdown(f"**試行 {h['attempt']} / 重み付き誤差 {h['error']:.3f} / 通常MAE {h['mae']:.3f}**")
                    st.write(h["text"])

# ---------- final result ----------
if st.session_state.get("result_text"):
    st.divider()
    st.markdown('<span class="step">RESULT</span><div class="section-title">最終結果</div><div class="section-desc">必要なら文章を自分で修正し、もう一度感情を確認できます。</div>', unsafe_allow_html=True)

    result_text = st.session_state["result_text"]
    result_scores = st.session_state["result_scores"]
    st.markdown(f"**{st.session_state.get('result_mode', '変換結果')}**")
    edited = st.text_area("文章を編集", value=result_text, height=170, key="final_editor")

    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("編集した文章を再分析", use_container_width=True):
            with st.spinner("再分析しています…"):
                edited_scores = analyze_emotion(edited)
            st.session_state["result_text"] = edited
            st.session_state["result_scores"] = edited_scores
            st.session_state["edited_scores"] = edited_scores
            st.rerun()
    with b2:
        st.download_button("文章をテキストで保存", data=edited, file_name="rewritten_text.txt", mime="text/plain", use_container_width=True)

    result_scores = st.session_state["result_scores"]
    st.markdown("#### 感情の確認")
    cols = st.columns(4)
    for i, (name, score) in enumerate(zip(EMOTION_NAMES, result_scores)):
        with cols[i % 4]:
            before = float(st.session_state["original_scores"][i])
            st.metric(name, f"{score:.2f}", delta=f"{float(score)-before:+.2f}")

    if "custom_target" in st.session_state:
        target = st.session_state["custom_target"]
        draw_target_chart(st.session_state["original_scores"], target, result_scores)
        error = float(np.mean(np.abs(result_scores - target)))
        closeness = max(0.0, min(100.0, (1.0 - error) * 100.0))
        c1, c2 = st.columns(2)
        c1.metric("目標との平均誤差", f"{error:.3f}")
        c2.metric("目標への近さ（参考値）", f"{closeness:.1f}%")
        st.caption("「目標への近さ」は8感情の平均絶対誤差から算出した参考表示で、心理学的な一致率や正解率ではありません。")
    else:
        fig = go.Figure()
        fig.add_bar(name="変換前", x=EMOTION_NAMES, y=st.session_state["original_scores"])
        fig.add_bar(name="変換後", x=EMOTION_NAMES, y=result_scores)
        fig.update_layout(barmode="group", height=380, yaxis=dict(range=[0,1.05], title="感情強度"), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### この結果を評価")
    r1, r2, r3 = st.columns(3)
    with r1: natural = st.slider("自然さ", 1, 5, 3, key="rate_natural")
    with r2: meaning = st.slider("意味の保たれ方", 1, 5, 3, key="rate_meaning")
    with r3: useful = st.slider("使いたさ", 1, 5, 3, key="rate_useful")
    comment = st.text_input("コメント（任意）", key="rate_comment")
    if st.button("評価を保存"):
        st.session_state.setdefault("ratings", []).append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "text": st.session_state["result_text"],
            "natural": natural, "meaning": meaning, "useful": useful, "comment": comment
        })
        st.success("このセッション内に評価を保存しました。")

# ---------- history ----------
st.divider()
with st.expander("変換履歴"):
    history = st.session_state.get("history", [])
    if not history:
        st.caption("まだ履歴はありません。")
    else:
        for idx, item in enumerate(history, 1):
            st.markdown(f"**{idx}. {item['mode']}**　{item['time']}")
            st.caption(f"相手: {item['audience']} / 場面: {item['channel']}")
            st.write("変換前：", item["original"])
            st.write("変換後：", item["result"])
            st.markdown("---")

st.caption("感情強度はモデルによる推定値です。人の感情や意図を断定するものではありません。")
