"""Parse all new work files using the copied project's actual interpreter."""
import ast
import json
import sys
import ppsi_paths

review = ppsi_paths.DELIVERY_REVIEW_ROOT.resolve()   # the DELIVERY_REVIEW folder
files = []
errors = []
for root in [review / 'ppsi_v4', review / 'code']:
    for p in root.rglob('*.py'):
        if '__pycache__' in p.parts:
            continue
        files.append(p.relative_to(review).as_posix())
        try:
            ast.parse(p.read_bytes(), filename=str(p))
        except SyntaxError as ex:
            errors.append(dict(path=str(p), error=str(ex)))
result = dict(interpreter=sys.version, files=len(files), errors=errors, success=not errors)
(review / 'evidence/FINAL_PYTHON_SYNTAX.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
raise SystemExit(0 if result['success'] else 1)
