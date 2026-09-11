# Flower Upload and Download Byte Measurement

How many bytes of model payload actually cross the Flower boundary, measured separately for each
direction.

## What is counted

- `download`: server to client.
- `upload`: client to server.

The figure is the length of the serialized arrays Flower transmits, read at the two points in the
strategy that see every transmission of a round. It is **not** a parameter count multiplied by a
dtype width.

The difference is not cosmetic. A `Linear(16, 8)` module holds 136 float32 parameters, so an
estimate would report 544 bytes. The measured payload is 800 bytes, because each array carries its
own serialization header. On this model the estimate is 32% low.

## What is not counted

This is the model payload boundary, not total network traffic. gRPC framing, headers, config
records, metric records, and acknowledgements are outside it. A payload figure must never be
reported as a wire figure.

A client whose reply never arrives is not recorded at all. It did not transmit zero bytes; it did
not transmit, and a zero-byte upload row would be read as a measurement rather than an absence.

## Reproduce

```text
uv run --locked python scripts/federated/fl_byte_measurement.py --output docs/evidence/s2-se-05/communication_bytes.v1.json
```

Two clients, two rounds, seed 13. Repeating the command leaves the JSON byte-identical.

## Result

| Round | Download | Upload | Download transmissions | Upload transmissions |
|---:|---:|---:|---:|---:|
| 1 | 1,600 B | 1,600 B | 2 | 2 |
| 2 | 1,600 B | 1,600 B | 2 | 2 |
| **Run** | **3,200 B** | **3,200 B** | **4** | **4** |

Download and upload are equal here only because every selected client returns a payload of the same
shape every round. They are accumulated separately and will diverge as soon as client selection,
partial participation, or an asymmetric payload appears.

## Privacy

`CommunicationRecord` carries a round, an opaque client ID, a direction, and a byte count. It has no
field that can hold payload contents, and constructing one with an extra field fails. Client IDs use
the existing `client_id_from_user` contract; a raw numeric user identifier is refused rather than
stored.
