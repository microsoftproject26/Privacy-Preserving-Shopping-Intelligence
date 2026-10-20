"""X budget fix (POST_HOC_EXPLORATORY, 2026-10-08): v4int.launch_fl.execute hard-codes the 6.0-EFE endpoint (half-EFE unit
12) twice after training: the 'ended without its 6.0 evaluation' check and the CONTROLLED_BUDGET rows. For a run whose
budget is not 6 EFE (launch_fl_x2 / launch_fl_x3 --x-efe 12 / 24) both must use the run's OWN endpoint unit
2 x cfg.endpoint_efe. apply(F) recompiles execute from its own source with exactly these two replacements (each must
match exactly once, else refused) in the module namespace; for a 6-EFE run the unit is 12, so behaviour is unchanged.
The checkpoint id keeps the file name endpoint_6.0 (fl.checkpoint ENDPOINT, the weights at the run's endpoint)."""
import inspect

OLD_CHECK = "    if not evals or evals[-1][0] != 12:\n"
NEW_CHECK = ("    _XU = int(round(2 * float(cfg.endpoint_efe)))          # X fix: the run's own endpoint unit\n"
             "    if not evals or evals[-1][0] != _XU:\n")
OLD_CB = '("CONTROLLED_BUDGET", 12, f"{run_id}:endpoint_6.0")'
NEW_CB = '("CONTROLLED_BUDGET", _XU, f"{run_id}:endpoint_6.0")'
RECORD = {"patch": "x_execute_fix", "replacements": [[OLD_CHECK.strip(), NEW_CHECK.strip()], [OLD_CB, NEW_CB]],
          "label": "POST_HOC_EXPLORATORY"}


def apply(F):
    src = inspect.getsource(F.execute)
    if src.count(OLD_CHECK) != 1 or src.count(OLD_CB) != 1:
        raise SystemExit("x_execute_fix: launch_fl.execute is not the expected source (refused)")
    src = src.replace(OLD_CHECK, NEW_CHECK).replace(OLD_CB, NEW_CB)
    exec(compile(src, F.__file__, "exec"), F.__dict__)
    return RECORD
