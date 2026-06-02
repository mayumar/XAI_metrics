# XAI-metrics

Librería para evaluar explicaciones de modelos de machine learning mediante
métricas de explicabilidad. El paquete permite comparar métodos XAI atributivos
como LIME, SHAP o BreakDown sobre un mismo modelo y conjunto de datos, generando resultados individuales y reportes agregados en formato CSV y JSON.

## Características principales

- Ejecución centralizada de métricas XAI mediante `run_evaluation`.
- Configuración declarativa mediante archivos YAML.
- Descubrimiento automático de combinaciones de datasets, modelos y métodos XAI.
- Soporte para métricas de complejidad, fidelidad, robustez, sensibilidad y
  faithfulness.
- Generación de reportes por dataset, modelo y método de explicación.
- Registro automático de métricas implementadas en el paquete.

## Instalación

Clona el repositorio e instala el paquete en modo editable:

```bash
git clone https://github.com/mayumar/XAI_metrics.git
cd XAI_metrics
pip install -e .
```

El proyecto requiere Python `>=3.10`.

## Uso rápido

La forma principal de usar la librería es llamar a `run_evaluation` indicando un
archivo de configuración. El siguiente ejemplo carga la configuración, construye
los contextos de evaluación, ejecuta las métricas y guarda los reportes:

```python
from xai_metrics.runner import run_evaluation

results = run_evaluation(
    config="xai_metrics/config.yaml",
    report_output_dir="results/reports",
)
```

El objeto `results` contiene:

- `contexts`: resultados individuales para cada combinación evaluada.
- `reports`: tablas agregadas por dataset y modelo.
- `report_paths`: rutas de los reportes CSV y JSON generados.

Si tus modelos necesitan un cargador personalizado o algunas métricas requieren
funciones de explicación en tiempo de ejecución, puedes pasarlas como argumentos:

```python
from xai_metrics.runner import run_evaluation
from utils import load_model
from xai_methods.lime import make_lime_explain_func
from xai_methods.shap import make_shap_local_explain_func
from xai_methods.break_down import make_breakdown_explain_func

explain_funcs = {
    "lime": make_lime_explain_func(X_train_norm),
    "shap": make_shap_local_explain_func(X_train_norm),
    "breakdown": make_breakdown_explain_func(X_train_norm),
}

results = run_evaluation(
    config="xai_metrics/config.yaml",
    model_loader=load_model,
    explain_funcs=explain_funcs,
    report_output_dir="results/reports",
)
```

## Configuración

La evaluación se define en un archivo YAML con dos bloques principales:
`context` y `metrics`.

```yaml
context:
  device: "cpu"
  datasets_dir: "prueba/data"
  models_dir: "prueba/results/models"
  attributions_dir: "prueba/results/attributions"

metrics:
  - name: Complexity
    params:
      normalise: true

  - name: Sparseness
    params:
      normalise: true

  - name: Faithfulness
    params:
      base_strategy: mean

  - name: LocalLipschitzEstimate
    params:
      nr_samples: 200
      abs: false
      normalise: true
      perturb_mean: 0.0
      perturb_std: 0.1
```

Con `datasets_dir`, `models_dir` y `attributions_dir`, la librería busca
automáticamente los contextos disponibles. Cada contexto combina:

- un dataset,
- un modelo entrenado,
- un conjunto de test,
- sus etiquetas,
- un archivo de atribuciones generado por un método XAI.

También se puede configurar un único contexto indicando rutas directas:

```yaml
context:
  dataset_name: "hydraulic"
  model_name: "IForest"
  xai_method_name: "LIME"
  device: "cpu"
  model_path: "prueba/results/models/hydraulic/IForest/hydraulic_IForest_seed_0.pkl"
  X_test_path: "prueba/data/hydraulic/X_test_train_norm.csv"
  y_test_path: "prueba/data/hydraulic/y_test_train.csv"
  attributions_path: "prueba/results/attributions/hydraulic/IForest/LIME/hydraulic_IForest_lime_attributions.csv"
```

## Estructura esperada de los datos

Para el descubrimiento automático, se espera una organización como la siguiente:

```text
prueba/
├── data_dir/
│   └── dataset1_name/
│       ├── X_test.csv
│       └── y_test.csv
├── models_dir/
│   └── dataset1_name/
│       └── ML_model1_name/
│           └── model.pkl
└── attributions_dir/
    └── dataset1_name/
        └── ML_model1_name/
            ├── XAI_method1_name/
            │   └── attributions.csv
            └── XAI_method2_name/
                └── attributions.csv
```

Los archivos de entrada deben cumplir estas condiciones:

- `X_test`: CSV con las observaciones en filas y las variables en columnas.
- `y_test`: CSV con las etiquetas, indexado igual que `X_test`.
- `attributions`: CSV donde cada fila corresponde a una observación explicada y
  cada columna a una variable.
- El índice de las atribuciones debe existir en el índice de `X_test`.
- El modelo cargado debe ser compatible con las métricas ejecutadas. Por defecto,
  la librería puede cargar modelos `.pkl`, `.pickle`, `.joblib`, `.jl`, `.pt` y
  `.pth`.

## Métricas disponibles

La librería incluye, entre otras, las siguientes métricas:

- `Complexity`
- `Sparseness`
- `Consistency`
- `Faithfulness`
- `FaithfulnessEstimate`
- `Monotonicity`
- `MonotonicityCorrelation`
- `MonotonicityMetric`
- `SensitivityN`
- `Sufficiency`
- `Completeness`
- `NonSensitivity`
- `LocalLipschitzEstimate`
- `MaxSensitivity`
- `RelativeInputStability`
- `RelativeOutputStability`
- `AvgSensitivity`

Las métricas se registran automáticamente mediante el decorador
`@register_metric`. Para ejecutar solo algunas métricas, usa `selected_metrics`:

```python
results = run_evaluation(
    config="xai_metrics/config.yaml",
    selected_metrics=["Complexity", "Sparseness", "Faithfulness"],
)
```

## Salida

Cuando `report_output_dir` no es `None`, la librería guarda un CSV y un JSON por
cada combinación de dataset y modelo:

```text
results/reports/
├── hydraulic_ECOD_xai_metrics_report.csv
└── hydraulic_ECOD_xai_metrics_report.json
```

Un reporte CSV tiene una estructura como esta:

```text
metric,BreakDown,LIME,SHAP
Completeness,,0.0,0.0
Complexity,1.7844,1.7611,1.7762
Consistency,,1.0,1.0
Faithfulness,0.6270,-0.8129,-0.1154
LocalLipschitzEstimate,,24.6297,4.8229
Monotonicity,,1.0,1.0
Sparseness,0.0558,0.0770,0.0832
Sufficiency,,0.75,1.0
```

Las filas corresponden a métricas y las columnas a métodos XAI. Esto permite
comparar de forma directa la calidad de distintas explicaciones para un mismo
modelo.

## Crear una métrica nueva

Una métrica debe heredar de `BaseMetric`, definir un nombre y registrar la clase:

```python
from xai_metrics.base import BaseMetric, register_metric

@register_metric
class MyMetric(BaseMetric):
    NAME = "MyMetric"

    def run(self):
        X_test = self.context.X_test
        attributions = self.context.attributions
        return float(attributions.mean())
```

Después de añadirla dentro del paquete `xai_metrics.metrics`, el sistema de
autodescubrimiento podrá registrarla y se podrá usar en el `config.yaml`:

```yaml
metrics:
  - name: MyMetric
    params: {}
```

## Ejecutar tests

```bash
pytest
```

## Documentación

El repositorio incluye configuración de Sphinx en `docs/`. Para generar la
documentación HTML:

```bash
cd docs
make html
```

## Licencia

Este proyecto se distribuye bajo licencia MIT.
