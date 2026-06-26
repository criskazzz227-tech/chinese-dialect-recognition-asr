from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import pandas as pd
import streamlit as st

from src.database import (
    add_history,
    authenticate_user,
    create_user,
    delete_history,
    get_history,
    init_db,
)
from src.logging_config import configure_logging
from src.pipeline import CLASSIFIER_OPTIONS, run_pipeline


st.set_page_config(
    page_title="方言智识",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #15201f;
        --muted: #667371;
        --line: #dce4e2;
        --brand: #0f766e;
        --brand-dark: #115e59;
        --accent: #d97706;
        --surface: #ffffff;
        --soft: #f4f8f7;
    }
    .stApp {
        background:
            linear-gradient(180deg, #edf6f4 0, #f8faf9 240px, #f8faf9 100%);
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stSidebar"] {
        background: #102a28;
        border-right: 0;
    }
    [data-testid="stSidebar"] * {
        color: #edf7f5;
    }
    [data-testid="stSidebar"] .stButton button {
        background: transparent;
        border-color: #496663;
        color: #edf7f5;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 28px;
    }
    .brand-mark {
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        border-radius: 8px;
        background: #14b8a6;
        color: #082f2c;
        font-weight: 800;
        font-size: 18px;
    }
    .brand-name {
        font-size: 19px;
        font-weight: 750;
        color: #f4fffd;
    }
    .brand-sub {
        color: #9ec0bc;
        font-size: 12px;
    }
    .hero {
        padding: 10px 0 24px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 24px;
    }
    .hero-kicker {
        color: var(--brand);
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: 6px 0 8px;
        font-size: 34px;
        line-height: 1.2;
        letter-spacing: 0;
    }
    .hero p {
        max-width: 720px;
        margin: 0;
        color: var(--muted);
        font-size: 15px;
    }
    .panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(21, 32, 31, 0.05);
    }
    .login-shell {
        max-width: 470px;
        margin: 6vh auto 0;
    }
    .login-title {
        text-align: center;
        margin-bottom: 24px;
    }
    .login-title h1 {
        margin: 12px 0 6px;
        font-size: 30px;
        letter-spacing: 0;
    }
    .login-title p {
        margin: 0;
        color: var(--muted);
    }
    div[data-testid="stMetric"] {
        background: var(--soft);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }
    .status-strip {
        border-left: 4px solid var(--brand);
        background: #ecfdf9;
        padding: 13px 15px;
        margin: 12px 0 20px;
        color: #134e4a;
    }
    .status-strip.warn {
        border-left-color: var(--accent);
        background: #fffbeb;
        color: #78350f;
    }
    .section-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        margin: 4px 0 8px;
    }
    .stButton > button[kind="primary"] {
        background: var(--brand);
        border-color: var(--brand);
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--brand-dark);
        border-color: var(--brand-dark);
    }
    @media (max-width: 700px) {
        .block-container { padding-top: 1rem; }
        .hero h1 { font-size: 27px; }
        .panel { padding: 16px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_brand():
    st.markdown(
        """
        <div class="brand-lockup">
          <div class="brand-mark">音</div>
          <div>
            <div class="brand-name">方言智识</div>
            <div class="brand-sub">Dialect Intelligence</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_auth():
    _, center, _ = st.columns([1, 1.25, 1])
    with center:
        st.markdown(
            """
            <div class="login-title">
              <div style="font-size:36px">🎙️</div>
              <h1>方言智识</h1>
              <p>中文地域方言识别与语音转写平台</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        login_tab, register_tab = st.tabs(["登录", "注册"])
        with login_tab:
            with st.form("login_form"):
                username = st.text_input("用户名", placeholder="请输入用户名")
                password = st.text_input(
                    "密码",
                    type="password",
                    placeholder="请输入密码",
                )
                submitted = st.form_submit_button(
                    "登录系统",
                    type="primary",
                    width="stretch",
                )
            if submitted:
                user = authenticate_user(username, password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("用户名或密码错误。")

        with register_tab:
            with st.form("register_form"):
                new_username = st.text_input(
                    "设置用户名",
                    placeholder="3 到 24 个字符",
                )
                new_password = st.text_input(
                    "设置密码",
                    type="password",
                    placeholder="至少 6 个字符",
                )
                confirm_password = st.text_input(
                    "确认密码",
                    type="password",
                    placeholder="再次输入密码",
                )
                registered = st.form_submit_button(
                    "创建账户",
                    width="stretch",
                )
            if registered:
                if new_password != confirm_password:
                    st.error("两次输入的密码不一致。")
                else:
                    success, message = create_user(new_username, new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)


def render_sidebar():
    with st.sidebar:
        render_brand()
        st.caption("当前账户")
        st.markdown(f"**{st.session_state.user['username']}**")
        st.divider()
        page = st.radio(
            "功能导航",
            ["识别工作台", "历史记录", "系统说明"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("支持方言")
        st.write("上海话 · 长沙话 · 郑州话")
        st.write("天津话 · 南昌话")
        st.write("")
        if st.button("退出登录", width="stretch"):
            st.session_state.pop("user", None)
            st.session_state.pop("last_result", None)
            st.rerun()
    return page


def render_result(result, elapsed):
    st.markdown("### 识别结果")
    dialect_col, confidence_col, duration_col, time_col = st.columns(4)
    dialect_col.metric("预测方言", result["dialect_name"])
    confidence_col.metric("置信度", f'{result["confidence"]:.1%}')
    duration_col.metric("音频时长", f'{result["duration_seconds"]:.1f} 秒')
    time_col.metric("处理耗时", f"{elapsed:.2f} 秒")

    if result["is_reliable"]:
        st.markdown(
            (
                '<div class="status-strip"><strong>结果可信</strong><br>'
                f'模型预测为 {result["dialect_name"]}，'
                f'置信度 {result["confidence"]:.1%}。</div>'
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            (
                '<div class="status-strip warn"><strong>低置信度提示</strong><br>'
                f'最高置信度 {result["confidence"]:.1%}，低于'
                f' {result["confidence_threshold"]:.0%} 的可靠阈值。'
                "建议使用更清晰、时长更充分的单人语音重试。</div>"
            ),
            unsafe_allow_html=True,
        )

    chart_col, table_col = st.columns([1.5, 1])
    top_df = pd.DataFrame(
        [
            {"方言": item["dialect_name"], "置信度": item["confidence"]}
            for item in result["top_predictions"]
        ]
    )
    with chart_col:
        st.markdown('<div class="section-label">概率分布</div>', unsafe_allow_html=True)
        st.bar_chart(
            top_df,
            x="方言",
            y="置信度",
            horizontal=True,
            height=245,
        )
    with table_col:
        st.markdown('<div class="section-label">Top-3 排名</div>', unsafe_allow_html=True)
        rank_df = top_df.copy()
        rank_df.insert(0, "排名", range(1, len(rank_df) + 1))
        rank_df["置信度"] = rank_df["置信度"].map(lambda value: f"{value:.1%}")
        st.dataframe(rank_df, hide_index=True, width="stretch")
        st.caption(f'分类使用 {result["segments"]} 个音频片段')

    if result["asr_text"]:
        st.markdown("### 语音内容")
        transcript_col, translation_col = st.columns(2)
        transcript_col.text_area(
            "中文转写",
            result["asr_text"],
            height=150,
        )
        translation_col.text_area(
            "英文翻译",
            result["translated_text"],
            height=150,
        )


def render_workspace():
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Recognition Workspace</div>
          <h1>中文地域方言识别</h1>
          <p>上传一段单人语音，系统将分析声学特征并给出方言类别、
          置信度排名与可选的 Whisper 语音转写。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_col, info_col = st.columns([2, 1])
    with control_col:
        st.markdown("### 上传与配置")
        audio_file = st.file_uploader(
            "上传音频文件",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            help="支持 WAV、MP3、M4A、FLAC 和 OGG 格式",
        )
        if audio_file is not None:
            st.audio(audio_file)

        option_col, model_col = st.columns(2)
        with option_col:
            classifier = st.selectbox(
                "方言分类模型",
                options=list(CLASSIFIER_OPTIONS),
                format_func=lambda key: CLASSIFIER_OPTIONS[key],
                index=1,
            )
        with model_col:
            run_asr = st.toggle("同时进行语音转写", value=True)
            whisper_model = st.selectbox(
                "转写模型",
                ["tiny", "base", "small"],
                index=1,
                disabled=not run_asr,
            )
        predict = st.button(
            "开始识别",
            type="primary",
            width="stretch",
            disabled=audio_file is None,
        )

    with info_col:
        st.markdown("### 使用建议")
        st.info(
            "建议上传 3 到 20 秒、背景噪声较少的单人语音。"
            "过短、多人混说或普通话占比较高的音频可能降低可信度。"
        )
        st.markdown("**当前模型**")
        st.caption(
            "CNN 基线与 Whisper 迁移特征双模型 · 5 类方言 · "
            "滑动窗口融合"
        )

    if predict and audio_file is not None:
        suffix = "." + audio_file.name.rsplit(".", 1)[-1].lower()
        tmp_path = None
        try:
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.getbuffer())
                tmp_path = tmp.name

            started_at = perf_counter()
            with st.spinner("正在分析音频，请稍候..."):
                result = run_pipeline(
                    tmp_path,
                    classifier=classifier,
                    whisper_model=whisper_model,
                    run_asr=run_asr,
                )
            elapsed = perf_counter() - started_at
            add_history(
                st.session_state.user["id"],
                audio_file.name,
                result,
                elapsed,
            )
            st.session_state.last_result = {
                "result": result,
                "elapsed": elapsed,
            }
        except Exception as exc:
            st.error(f"识别失败：{exc}")
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    if "last_result" in st.session_state:
        render_result(
            st.session_state.last_result["result"],
            st.session_state.last_result["elapsed"],
        )


def render_history():
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Personal Records</div>
          <h1>识别历史</h1>
          <p>查看当前账户最近的方言识别记录与转写结果。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    rows = get_history(st.session_state.user["id"])
    if not rows:
        st.info("暂时没有识别记录，先去工作台上传一段音频吧。")
        return

    total_col, reliable_col, average_col = st.columns(3)
    reliable_count = sum(row["is_reliable"] for row in rows)
    average_confidence = sum(row["confidence"] for row in rows) / len(rows)
    total_col.metric("识别次数", len(rows))
    reliable_col.metric("可信结果", reliable_count)
    average_col.metric("平均置信度", f"{average_confidence:.1%}")

    table = pd.DataFrame(
        [
            {
                "时间": row["created_at"].replace("T", " "),
                "文件": row["filename"],
                "预测方言": row["dialect"],
                "置信度": f'{row["confidence"]:.1%}',
                "状态": "可信" if row["is_reliable"] else "低置信度",
                "模型": CLASSIFIER_OPTIONS.get(
                    row["classifier"],
                    row["classifier"],
                ),
                "音频时长": f'{row["duration_seconds"]:.1f} 秒',
                "处理耗时": f'{row["processing_seconds"]:.2f} 秒',
            }
            for row in rows
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch", height=410)

    transcripts = [row for row in rows if row["asr_text"]]
    if transcripts:
        with st.expander("查看最近的语音转写"):
            for row in transcripts[:5]:
                st.markdown(f"**{row['filename']} · {row['dialect']}**")
                st.write(row["asr_text"])
                st.divider()

    with st.expander("记录管理"):
        st.warning("清空后无法恢复。")
        if st.button("清空全部历史记录"):
            delete_history(st.session_state.user["id"])
            st.session_state.pop("last_result", None)
            st.rerun()


def render_about():
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">System Overview</div>
          <h1>系统说明</h1>
          <p>从音频预处理、方言分类到语音转写的完整实验型应用。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    architecture_col, security_col = st.columns(2)
    with architecture_col:
        st.markdown("### 识别流程")
        st.markdown(
            """
            1. 音频重采样与静音裁剪  
            2. 提取 Log-Mel 声学特征  
            3. 1D CNN 完成五分类预测  
            4. 长音频滑动窗口概率融合  
            5. 可选 Whisper 语音转写
            """
        )
    with security_col:
        st.markdown("### 数据与账户")
        st.markdown(
            """
            - 用户与历史记录保存在本地 SQLite  
            - 密码使用 PBKDF2-SHA256 加盐哈希  
            - 上传音频仅用于本次推理，完成后删除临时文件  
            - 不在数据库中保存原始音频
            """
        )


configure_logging()
init_db()
if "user" not in st.session_state:
    render_auth()
    st.stop()

selected_page = render_sidebar()
if selected_page == "识别工作台":
    render_workspace()
elif selected_page == "历史记录":
    render_history()
else:
    render_about()
