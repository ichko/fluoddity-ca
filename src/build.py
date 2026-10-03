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
# the trained gecko: its template with /*GECKO*/ replaced by gecko.json (written by train/export_gecko_web.py)
gecko = open(os.path.join(ROOT, "gecko.json")).read()
t = open(os.path.join(HERE, "gecko.tmpl.html")).read()
open(os.path.join(ROOT, "gecko.html"), "w").write(t.replace("/*GECKO*/", "const GECKO = " + gecko + ";"))
print("wrote gecko.html")
