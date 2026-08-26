"""Reel Query — hybrid retrieval (vector + SQL) over a film dialogue corpus in ClickHouse.

The pieces, in the order a question moves through them:

* :mod:`reel_query.ingest`           — load a corpus into ``lines``
* :mod:`reel_query.chunking`         — chunk it (``line`` or ``window``), a swept parameter
* :mod:`reel_query.embeddings`       — local sentence-transformer vectors
* :mod:`reel_query.retrieval`        — vector search, and vector search feeding a GROUP BY
* :mod:`reel_query.schema_retrieval` — the schema slice that grounds text-to-SQL
* :mod:`reel_query.agent`            — the ADK agent that plans, queries and self-validates
* :mod:`reel_query.evaluation`       — recall@k, faithfulness, and the sweep
"""

__version__ = "0.1.0"
