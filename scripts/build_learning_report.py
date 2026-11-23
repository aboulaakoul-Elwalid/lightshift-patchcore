"""Build the unified learning report (single self-contained HTML).

report/parts/*.html are concatenated in filename order; tokens are expanded:
  {{CHART:name}}  -> inline SVG from report_charts.CHARTS[name]()
  {{TABLE:name}}  -> HTML table from report_charts.TABLES[name]()
  {{V:key}}       -> scalar from report_charts.VALUES()[key]
Run: .venv/bin/python scripts/build_learning_report.py   (needs the results/ folders)
Out: results/report/lightshift-learning-report.html
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_charts as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "report" / "lightshift-learning-report.html"

body = "\n".join(p.read_text() for p in sorted((REPO / "report" / "parts").glob("*.html")))
vals = rc.VALUES()


def sub(m):
    kind, name = m.group(1), m.group(2)
    if kind == "CHART":
        return rc.CHARTS[name]()
    if kind == "TABLE":
        return rc.TABLES[name]()
    if kind == "V":
        return str(vals[name])
    raise KeyError(kind)


body = re.sub(r"\{\{(CHART|TABLE|V):([\w\-\.]+)\}\}", sub, body)
left = re.findall(r"\{\{[^}]*\}\}", body)
assert not left, left[:5]
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>LightShift-PatchCore: the learning report</title>
<style>{(REPO / 'report' / 'style.css').read_text()}</style></head>
<body><main>{body}</main></body></html>"""
OUT.write_text(html)
print(f"wrote {OUT} ({len(html)/1024:.0f} KB)")
