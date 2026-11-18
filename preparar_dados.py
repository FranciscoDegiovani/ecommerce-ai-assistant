"""Baixa a base pública da Olist e gera uma amostra em SQLite para o assistente.

Uso:
    python preparar_dados.py                      # baixa do Kaggle via kagglehub
    python preparar_dados.py --origem pasta_csv   # usa CSVs já baixados
"""
import argparse
import os
import sqlite3

import pandas as pd

ARQUIVOS = {
    "olist_customers_dataset.csv": "customers",
    "olist_orders_dataset.csv": "orders",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "payments",
    "olist_order_reviews_dataset.csv": "reviews",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "product_category_name_translation.csv": "category_translation",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", help="pasta com os CSVs da Olist (opcional)")
    ap.add_argument("--destino", default="data/olist_amostra.db")
    ap.add_argument("--n-pedidos", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    origem = args.origem
    if not origem:
        import kagglehub
        origem = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    print(f"Lendo CSVs de: {origem}")

    d = {t: pd.read_csv(os.path.join(origem, f), encoding="utf-8-sig") for f, t in ARQUIVOS.items()}

    # amostra: 90% pedidos entregues + 10% outros status (enviado, cancelado...) para a demo
    orders = d["orders"]
    entregues = orders[orders.order_status == "delivered"]
    outros = orders[orders.order_status != "delivered"]
    n_ent = min(int(args.n_pedidos * 0.9), len(entregues))
    n_out = min(args.n_pedidos - n_ent, len(outros))
    amostra = pd.concat([
        entregues.sample(n=n_ent, random_state=args.seed),
        outros.sample(n=n_out, random_state=args.seed),
    ])

    ids = set(amostra.order_id)
    itens = d["order_items"][d["order_items"].order_id.isin(ids)]
    tabelas = {
        "orders": amostra,
        "customers": d["customers"][d["customers"].customer_id.isin(amostra.customer_id)],
        "order_items": itens,
        "payments": d["payments"][d["payments"].order_id.isin(ids)],
        "reviews": d["reviews"][d["reviews"].order_id.isin(ids)],
        "products": d["products"][d["products"].product_id.isin(itens.product_id)],
        "sellers": d["sellers"][d["sellers"].seller_id.isin(itens.seller_id)],
        "category_translation": d["category_translation"],
    }

    os.makedirs(os.path.dirname(args.destino) or ".", exist_ok=True)
    if os.path.exists(args.destino):
        os.remove(args.destino)
    con = sqlite3.connect(args.destino)
    for nome, df in tabelas.items():
        df.to_sql(nome, con, index=False)
        print(f"{nome:22s} {len(df):>7,} linhas")
    for tabela in ["orders", "order_items", "payments", "reviews"]:
        con.execute(f"CREATE INDEX idx_{tabela}_order ON {tabela}(order_id)")
    con.execute("CREATE INDEX idx_items_product ON order_items(product_id)")
    con.commit()
    con.close()
    print(f"\nBanco salvo em {args.destino} ({os.path.getsize(args.destino) / 1e6:.1f} MB)")
    print(amostra.order_status.value_counts().to_string())


if __name__ == "__main__":
    main()
