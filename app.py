"""Interface de chat (Streamlit) do assistente da TechNova."""
import os

import streamlit as st

# No Streamlit Cloud as chaves vêm de st.secrets; localmente, do arquivo .env
try:
    for chave in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
        if chave in st.secrets:
            os.environ[chave] = str(st.secrets[chave])
except Exception:
    pass

import agente  # noqa: E402  (importado depois de configurar as variáveis)

st.set_page_config(page_title="TechNova | Assistente de atendimento")


@st.cache_data
def pedidos_exemplo():
    rows = agente.con().execute(
        """SELECT order_status, order_id FROM orders
           WHERE order_status IN ('delivered', 'shipped', 'canceled')
           GROUP BY order_status"""
    ).fetchall()
    return [(agente.STATUS_PT[s], oid) for s, oid in rows]


with st.sidebar:
    st.header("TechNova")
    st.write(
        "Assistente de atendimento com IA para uma loja online fictícia. "
        "Consulta pedidos reais da base pública Olist (SQL), responde sobre políticas da loja (RAG) "
        "e resume avaliações de clientes."
    )
    st.subheader("Experimente")
    st.write("Pedidos de exemplo:")
    for status, oid in pedidos_exemplo():
        st.code(oid, language=None)
        st.caption(status)
    st.write("Perguntas: prazo de troca, garantia de placa de vídeo, frete grátis, opinião sobre produtos de informática.")
    if st.button("Limpar conversa"):
        st.session_state.historico = []
        st.rerun()

st.title("Assistente de atendimento")

if "historico" not in st.session_state:
    st.session_state.historico = []

for m in st.session_state.historico:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if pergunta := st.chat_input("Digite sua dúvida"):
    st.session_state.historico.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        log = []
        with st.spinner("Consultando..."):
            try:
                resposta = agente.responder(st.session_state.historico, log=log)
            except Exception as e:
                resposta = f"Erro ao chamar o modelo: {e}"
        st.markdown(resposta)
        if log:
            with st.expander("Ferramentas usadas"):
                st.json(log)

    st.session_state.historico.append({"role": "assistant", "content": resposta})
