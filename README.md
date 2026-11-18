# E-commerce AI Assistant

AI customer-service assistant for a fictional online tech store (TechNova). An LLM agent decides which tools to call to answer customers with grounded, verifiable information instead of guessing.

| Tool | What it does | Technique |
|---|---|---|
| `consultar_pedido` | Order status and delivery dates | SQL over real e-commerce data |
| `itens_do_pedido` | Items, prices and shipping of an order | SQL |
| `buscar_politica` | Returns, warranty, shipping, payment, cancellation | RAG with multilingual embeddings |
| `resumo_avaliacoes` | Average rating and real customer comments by category | SQL aggregation |

**Live demo:** https://technova-assistant.streamlit.app

![demo](demo.gif)

## Architecture

```
User ──> Streamlit chat ──> LLM (tool calling) ──┬──> SQL tools ──> SQLite (Olist sample)
                                                  ├──> RAG ──> store policies (embeddings)
                                                  └──> review aggregation
```

- **Provider-agnostic:** uses the OpenAI-compatible API, so the model can be swapped through environment variables.
- **Fault tolerance:** retries with backoff on rate limits and overloaded servers, and automatic fallback across a list of models. Built after the provider returned 503 errors during development.
- **Guardrails:** order IDs are validated before hitting the database, tool errors are returned to the model instead of crashing the chat, a step limit prevents loops, and the system prompt forbids inventing order data or policy details.
- **Transparency:** the chat shows which tools were called for each answer.

## Evaluation

`avaliar.py` runs 30 test questions: orders, policies, reviews, out-of-scope requests, and 10 hard cases (uppercase or truncated order IDs, two questions in one message, edge cases outside the policy rules, an angry customer, prompt injection and a request for personal data).

| Metric | Result |
|---|---|
| Tool routing accuracy | 100% (30/30) |
| Factual accuracy | 100% (21/21 checkable cases) |
| Average latency | ~15 s (during provider overload) |

**What the evaluation caught:** the first run scored 95% factual accuracy. The failing case was "how many installments can I pay in?": retrieval missed the payment section because the customer's wording ("parcelar") didn't match the document heading. The agent did not hallucinate an answer; it said it couldn't find the information and offered a human agent, as instructed. Rewriting the section heading and text fixed retrieval, raising factual accuracy to 100%.

Model: `gemini-3.8-flash` with fallback to other Gemini Flash models. Full results in `resultados/avaliacao.csv`.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
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
