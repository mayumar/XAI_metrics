# uv run prueba/main.py -e train -d ARAMIS20
# uv run prueba/main.py -e train -d AHU21
uv run prueba/main.py -e lime -d ARAMIS20
uv run prueba/main.py -e lime -d AHU21
uv run prueba/main.py -e shap_local -d ARAMIS20
uv run prueba/main.py -e shap_local -d AHU21
# uv run prueba/main.py -e eval -d ARAMIS20
# uv run prueba/main.py -e eval -d AHU21