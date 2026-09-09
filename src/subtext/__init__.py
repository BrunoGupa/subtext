"""Subtext — hybrid retrieval (vector + SQL) over a film dialogue corpus in ClickHouse.

The pieces, in the order a question moves through them:

* :mod:`subtext.ingest`           — load a corpus into ``lines``
* :mod:`subtext.chunking`         — chunk it (``line`` or ``window``), a swept parameter
* :mod:`subtext.embeddings`       — Gemini embedding vectors
* :mod:`subtext.retrieval`        — vector search, and vector search feeding a GROUP BY
* :mod:`subtext.schema_retrieval` — the schema slice that grounds text-to-SQL
* :mod:`subtext.agent`            — the ADK agent that plans, queries and self-validates
* :mod:`subtext.evaluation`       — recall@k, faithfulness, and the sweep
"""

__version__ = "0.1.0"
