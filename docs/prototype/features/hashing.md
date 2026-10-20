# Product Hashing

Product IDs are mapped to a fixed number of buckets before they reach the model. This avoids a
large embedding row for every catalog item.

The implementation hashes the complete namespaced CanonicalEvent item ID with BLAKE2b and reserves
bucket zero for padding. It never uses Python hash, and category or brand inputs remain separate.

ProductHashConfig requires three explicit values:

- bucket count;
- seed;
- residual embedding dimension.

There are no production defaults yet. The team should choose those values during architecture
review after checking product count, collisions, model size, and federated communication cost.

The hash is a compact representation mechanism, not encryption or a privacy guarantee. ONNX models
consume the resulting int64 buckets; string hashing remains preprocessing.

Run:

    uv run --locked pytest -q tests/features
    uv run --locked ruff check ppsi/features tests/features
