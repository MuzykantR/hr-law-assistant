"""Web interface: uv run streamlit run app.py"""

import streamlit as st

from rag.assistant import build_assistant

st.set_page_config(page_title="Ассистент по Трудовому кодексу", page_icon="⚖️")
st.title("Ассистент по Трудовому кодексу РФ")
st.caption("Отвечает только по тексту ТК РФ и ссылается на статью и часть. Не является юридической консультацией.")

mode = st.sidebar.radio("Поиск", ["dense", "hybrid", "bm25"], help="dense лучше всех по оценке (results/retrieval_eval.md); hybrid = BM25 + эмбеддинги через RRF")
st.sidebar.markdown(
    "**Примеры вопросов**\n"
    "- Сколько дней отпуска мне положено?\n"
    "- Можно ли разделить отпуск на части?\n"
    "- За сколько надо предупредить об увольнении?\n"
    "- Как оплачивается работа в выходной?"
)


@st.cache_resource(show_spinner="Загружаю кодекс и индекс…")
def get_assistant(mode: str):
    return build_assistant(mode)


assistant = get_assistant(mode)

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    st.chat_message(role).markdown(content)

if question := st.chat_input("Задайте вопрос о трудовых правах"):
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Ищу в кодексе…"):
            answer = assistant.ask(question)
        if answer.query != question:
            st.caption(f"Исправлена раскладка: «{answer.query}»")
        st.markdown(answer.text)
        for c in answer.sources:
            with st.expander(f"[{c.citation}] {c.title}"):
                st.text(c.text)
    st.session_state.history += [("user", question), ("assistant", answer.text)]
