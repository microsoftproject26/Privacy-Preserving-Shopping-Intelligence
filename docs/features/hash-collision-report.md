# Product Hash Collision Report

This report measures the existing `blake2b-64-v1` product hash on the committed catalogue. It is
decision input for the architecture checkpoint, not a final architecture choice.

## Reproduce

```text
uv run --locked python -m ppsi.features.collision_report --catalog fixtures/reference/item_catalog_v1.proposed.parquet --output docs/evidence/s2-se-03/product_hash_collisions.v1.json --seed 13 --bucket-counts 50000 262145 524289 1048577
```

The command is deterministic: the JSON has no timestamp and records the input path relative to the
repository, so two machines produce byte-identical output. It records the catalogue SHA-256,
algorithm, seed, candidate counts, and aggregate results; it never writes product IDs.

## Why these candidates

Bucket `0` is reserved for padding. `50000` is the value the current model code uses, and the other
three leave exactly `2^18`, `2^19`, and `2^20` usable buckets. For the catalogue's 113,765 unique
products those three are roughly 2.3x, 4.6x, and 9.2x as many usable buckets as products, while
`50000` is about 0.44x, fewer buckets than there are products.

## Results

`collision_rate` is `excess_collisions / unique_product_count`: products beyond the first occupant
of each used bucket, divided by all unique products.

| Total buckets | Occupied | Collided buckets | Excess collisions | Collision rate | Max load |
|---:|---:|---:|---:|---:|---:|
| 50,000 | 44,934 | 33,223 | 68,831 | 60.5028% | 11 |
| 262,145 | 92,335 | 18,538 | 21,430 | 18.8371% | 6 |
| 524,289 | 102,202 | 10,765 | 11,563 | 10.1639% | 5 |
| 1,048,577 | 107,683 | 5,880 | 6,082 | 5.3461% | 4 |

Doubling usable capacity lowers both the collision rate and the maximum observed bucket load.

## What the 50,000 row means

At 50,000 buckets there are fewer buckets than products, so collisions are not an accident of the
hash; they are forced. Three products in five share their identity with another product, and one
bucket carries eleven distinct products. Every product sharing a bucket is indistinguishable to the
model, which is a real constraint on how much product-level signal the residual can carry.

That is a statement about capacity, not a verdict. A small bucket count is a legitimate choice when
product identity is meant to be a coarse auxiliary signal and category, brand, and price bands carry
the specific information. The architecture task should combine this evidence with embedding size,
measured accuracy, and communication cost before fixing the final value.
