# uv run prueba/main.py -e train -d ARAMIS20
# uv run prueba/main.py -e train -d AHU21
# uv run prueba/main.py -e batch -d AHU21 -n 3
# uv run prueba/main.py -e batch -d ARAMIS20 -n 3
# uv run prueba/main.py -e lime
uv run prueba/main.py -e maple
uv run prueba/main.py -e shap_local
# uv run prueba/main.py -e breakdown
# uv run prueba/main.py -e eval