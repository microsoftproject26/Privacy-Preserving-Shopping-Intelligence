"""v4data09 (owner: Data engineer, DATA-09) - the gated RETRO_TEST manifests of the single TEST look (PREG1_AMENDMENT_06D
§2; EVALUATION_CONTRACT_V4 §10; MANIFEST_SPEC_V4 M07/M08, global_rules.test_gate) over the LOCKED DATA-04 builder
(subclassed through v4data06.D6Builder, never edited). Nothing here runs, reads a TEST row or builds a manifest unless
d9_gate.check_test_gate returns a GateRecord (ALL_METHODS_LOCK.json + a MANIFEST_BUILD TEST_EXPOSURE_REGISTER entry).
"""
