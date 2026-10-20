# RESTRICTED — superseded CENSUS-02 run 1 (quarantined 2026-09-26T19:43Z by the Operations Director)

**Why:** this run's `PRECISION_PRIOR.json` contains v1-model OUTCOME LEVELS: the mean RR of the v1 GRU and SASRec models
on VALIDATION_CONFIRM, restricted to stratum C and to candidate cohorts, and their mean GRU − SASRec difference. The
task packet did not request these values. They are information about how the two architectures compare on the new
population, and they must not influence the G0c or G2 decisions.

**Rule:**
- The Operations Director, Fable and every decision-making reviewer must NOT open this folder until
  ARCHITECTURE_LOCK (G2) is written. Only the CENSUS-02 worker has seen it (disclosed in its handoff).
- The folder is preserved unchanged. It is not deleted, because it is historical evidence.
- Every count in this run equals run 2, the run of record, which withholds the levels.
- After G2 it may be released as historical prior evidence.

**File sha256 at quarantine:**
{
 "CANDIDATE_N_PROJECTIONS.json": "8a54539eec417234b2415a3e54f55c6d9d8c2703ddba5756b93ec01c65a1c0b4",
 "CATALOGUE_OPTION_B.json": "26bbb9936b5053a5679cf1e543c121dec56184654c05e520537616b1d18e355f",
 "CENSUS02_RECEIPT.json": "cdd659764601e454c85f7ec8082211c649eea70d4ecd765734a0473229660593",
 "FL_LOCAL_WORK.json": "9f20575c0e7a75808e3b5439f458c02bdc907852801a5ed72d60b891c2eb82c0",
 "PRECISION_PRIOR.json": "6e74713362b22f1af64fd8334de47886672ba930d8e16d5d52a7a76f565530c3",
 "STRATA_CENSUS.json": "9a53fb33e0b2d27768847282a8a56b37a7a760db289f0ad53db088183e4c1643"
}
