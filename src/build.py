"""Build the pages: each template with the line /*PRESETS*/ replaced by presets.js
(and /*CA_PRESETS*/, where present, by ca_presets.js), and gecko.html with /*GECKO*/ replaced by gecko.json.

    python3 src/build.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
presets = open(os.path.join(HERE, "presets.js")).read()
ca_presets = open(os.path.join(HERE, "ca_presets.js")).read()
for name in ["fluoddity", "fluoddity-ca", "fluoddity-ca-next"]:
    t = open(os.path.join(HERE, name + ".tmpl.html")).read()
    open(os.path.join(ROOT, name + ".html"), "w").write(t.replace("/*PRESETS*/", presets).replace("/*CA_PRESETS*/", ca_presets))
    print("wrote", name + ".html")
# the trained geckos: the template with /*GECKO*/ replaced by every model JSON (train/export_gecko_web.py) and
# the pick, ?model=KEY in the URL (default: the first). The page's model menu lists them.
MODELS = [("settle", "gecko-settle.json", "settles back (dsF3)"), ("robust", "gecko.json", "robust (dynR)")]
MODELS = [m for m in MODELS if os.path.exists(os.path.join(ROOT, m[1]))]
js = "const GECKO_MODELS = {" + ",".join(f'"{k}":' + open(os.path.join(ROOT, fn)).read() for k, fn, _ in MODELS) + "};\n"
js += "const GECKO_LABELS = " + repr([[k, lab] for k, _, lab in MODELS]).replace("'", '"') + ";\n"
js += "const GECKO_KEY = (k => k in GECKO_MODELS ? k : GECKO_LABELS[0][0])(new URLSearchParams(location.search).get(\"model\"));\n"
js += "const GECKO = GECKO_MODELS[GECKO_KEY];"
t = open(os.path.join(HERE, "gecko.tmpl.html")).read()
open(os.path.join(ROOT, "gecko.html"), "w").write(t.replace("/*GECKO*/", js))
print("wrote gecko.html with", [k for k, _, _ in MODELS])
