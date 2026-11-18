# E-commerce AI Assistant

AI customer-service assistant for a fictional online tech store (TechNova). An LLM agent decides which tools to call to answer customers with grounded, verifiable information:

| Tool | What it does | Technique |
|---|---|---|
| `consultar_pedido` | Order status and delivery dates | SQL over real e-commerce data |
| `itens_do_pedido` | Items, prices and shipping of an order | SQL |
| `buscar_politica` | Returns, warranty, shipping, payment, cancellation | RAG with multilingual embeddings |
| `resumo_avaliacoes` | Average rating and real customer comments by category | SQL aggregation |

**Live demo:** LINK_DO_APP

![demo](demo.gif)

## Architecture

```
User ──> Streamlit chat ──> LLM (tool calling) ──┬──> SQL tools ──> SQLite (Olist sample)
                                                  ├──> RAG ──> store policies (embeddings)
                                                  └──> review aggregation
```

- Provider-agnostic: uses the OpenAI-compatible API, so the model can be swapped by changing two environment variables.
- Guardrails: order IDs are validated before hitting the database, tool errors are returned to the model instead of crashing, a step limit prevents loops, and the system prompt forbids inventing order data or policy details.
- Transparency: the chat shows which tools were called for each answer.

## Evaluation

`avaliar.py` runs 20 test questions (orders, policies, reviews and out-of-scope / prompt-injection attempts) and measures:

| Metric | Result |
|---|---|
| Tool routing accuracy | XX% |
| Factual accuracy (expected info present in answer) | XX% |
| Hallucinations found in manual review | X of 20 |
| Average latency | X.X s |

Model: `MODELO_USADO`. Full results in `resultados/avaliacao.csv`.

## Run locally

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash); on Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # add your API key
python preparar_dados.py           # builds data/olist_amostra.db
streamlit run app.py
```

## Next steps toward production

- Conversation memory and handoff to a human agent with full context
- Vector database instead of in-memory embeddings as the knowledge base grows
- Observability (traces, cost and latency per conversation) and continuous evaluation on real tickets
- Stricter input/output guardrails and PII masking

## Data

Order data: [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (sample of 5,000 orders). Store policies are fictional and written for this project.
