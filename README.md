# Principal-Agent Problem with Regime Switching

Este repositorio contiene la implementación numérica de un problema principal-agente aplicado a la ejecución óptima de una orden financiera bajo cambios de régimen en el mercado.

El objetivo general es estudiar cómo un cliente, o principal, puede diseñar un contrato para inducir la estrategia de ejecución de un agente. Dado un contrato, el agente resuelve un problema de control óptimo. Posteriormente, el cliente compara distintos contratos y selecciona aquel que minimiza su criterio de costo ajustado por riesgo.

El código está organizado para separar claramente dos niveles del problema:

1. **Problema del agente:** dada una función contractual (c(t)), se resuelve la ecuación HJB del agente mediante iteración de políticas y se simulan trayectorias bajo la estrategia óptima obtenida.

2. **Problema del cliente:** se parametriza una familia de contratos mediante coeficientes (\beta), se resuelve el problema del agente para cada contrato candidato y se evalúa el objetivo del cliente.

---

## Estructura del repositorio

El repositorio contiene tres archivos principales de Python:

```text
principal-agent-regime-switching/
│
├── hjb_policy_iteration.py
├── client_optimal_contract.py
├── run_contracts.py
└── README.md
```

---

## 1. `hjb_policy_iteration.py`

Este archivo contiene las funciones necesarias para resolver y simular el problema del agente.

El módulo recibe como entrada una función contractual (c(t)). A partir de ella:

1. construye la discretización de la ecuación HJB;
2. resuelve la HJB mediante iteración de políticas;
3. obtiene la estrategia óptima discreta (q^*);
4. simula trayectorias bajo dicha estrategia;
5. permite graficar las trayectorias simuladas.

### Funciones principales

#### `solve_hjb(grids, params, contract_func)`

Resuelve la ecuación HJB del agente para una función contractual dada.

Argumentos principales:

* `grids`: tupla con las mallas de estado `(I_grid, P_grid, Q_grid)`.
* `params`: diccionario con los parámetros del modelo.
* `contract_func`: función de ponderación contractual (c(t)).

Regresa un diccionario que contiene:

* `all_thetas`: valores aproximados de la función transformada.
* `all_qs`: estrategias óptimas discretizadas.
* `tau_grid`: malla temporal en tiempo restante.
* `t_grid`: malla temporal en tiempo real.
* `c_t_array`: valores del contrato sobre la malla temporal.
* `histories`: información de convergencia de la iteración de políticas.

#### `simulate_one_path(result, grids, params, contract_func, ...)`

Simula una trayectoria hacia adelante usando la estrategia óptima calculada con `solve_hjb`.

Regresa un diccionario con las trayectorias simuladas de:

* `t`: tiempo;
* `q`: estrategia de ejecución;
* `I`: impacto en el precio;
* `P`: precio;
* `Q`: inventario restante;
* `regime`: régimen de mercado;
* `X`: riqueza acumulada;
* `clip_count`: número de veces que la trayectoria sale de la malla usada para aproximar la política.

#### `plot_simulated_paths(paths, title=...)`

Genera una figura de resumen con las trayectorias simuladas. La figura contiene:

1. estrategia óptima (q^*);
2. inventario restante (Q_t);
3. impacto (I_t);
4. régimen de mercado;
5. precio (P_t);
6. riqueza acumulada.

---

## 2. `client_optimal_contract.py`

Este archivo contiene las funciones para resolver el problema del cliente.

La lógica del archivo es:

```text
beta
  -> contrato c_beta(t)
  -> resolver HJB del agente
  -> simular trayectorias bajo la estrategia óptima del agente
  -> calcular C0*(c_beta) y J(c_beta)
  -> escoger el beta con menor valor objetivo
```

El problema del cliente se aproxima mediante una búsqueda exhaustiva sobre una malla finita de coeficientes (\beta). Por lo tanto, el contrato encontrado es óptimo dentro de la malla considerada, no necesariamente dentro de toda la clase continua de contratos admisibles.

### Contratos Bernstein

Los contratos se parametrizan mediante polinomios de Bernstein. Si

[
\beta = (\beta_0,\dots,\beta_d),
]

entonces se construye una función contractual (c_\beta(t)) de grado (d). En el código, esta construcción se realiza con:

```python
contrato_bernstein(beta, T=1.0)
```

La función requiere coeficientes no negativos.

### Funciones principales

#### `contrato_bernstein(beta, T=1.0)`

Construye la función (c_\beta(t)) asociada a un vector de coeficientes beta.

#### `normalizar_beta_integral(beta, T=1.0, integral_objetivo=1.0)`

Reescala un vector beta para que la integral del contrato sea igual a un valor objetivo.

Esta función es útil cuando se desea comparar contratos con la misma escala total.

#### `malla_beta_simplex(grado=2, niveles=3, T=1.0, integral_objetivo=1.0)`

Genera candidatos beta no negativos con integral fija.

Esta opción permite comparar formas contractuales manteniendo constante la escala total del contrato.

#### `malla_beta_caja(beta_min, beta_max, niveles)`

Genera una malla de candidatos beta usando una caja distinta para cada coeficiente.

Ejemplo:

```python
beta_candidates = malla_beta_caja(
    beta_min=[0.0, 0.0, 0.0],
    beta_max=[2.0, 2.0, 0.5],
    niveles=[7, 7, 4],
)
```

Este ejemplo genera:

```text
beta_0 en [0.0, 2.0] con 7 puntos
beta_1 en [0.0, 2.0] con 7 puntos
beta_2 en [0.0, 0.5] con 4 puntos
```

En total se evalúan (7 \times 7 \times 4 = 196) - 1 contratos, ya que se omite el contrato nulo.

#### `evaluar_contrato(contrato, grids, parametros_agente, parametros_cliente, beta=None)`

Evalúa un contrato dado desde el punto de vista del cliente.

Los pasos son:

1. resolver la HJB del agente inducida por el contrato;
2. simular trayectorias bajo la estrategia óptima;
3. calcular (C_0^*), (J) y otras métricas auxiliares.

#### `evaluar_beta(beta, grids, parametros_agente, parametros_cliente)`

Construye el contrato Bernstein asociado a beta y lo evalúa con `evaluar_contrato`.

#### `buscar_mejor_contrato(beta_candidates, grids, parametros_agente, parametros_cliente, imprimir=True)`

Evalúa una lista de candidatos beta y selecciona el que minimiza el objetivo del cliente.

Regresa:

* `tabla`: un `DataFrame` con las métricas de todos los candidatos;
* `mejor`: un diccionario con el mejor contrato, su beta, la solución HJB, las trayectorias simuladas y las métricas asociadas.

---

## 3. `run_contracts.py`

Este archivo permite correr el problema del agente para una familia fija de contratos especificados manualmente.

A diferencia de `client_optimal_contract.py`, aquí no se busca un beta óptimo. En su lugar, el usuario define una familia de contratos específicos y el código:

1. resuelve el problema del agente para cada contrato;
2. simula trayectorias bajo la estrategia óptima;
3. guarda los resultados en caché;
4. genera gráficas comparativas.

Este archivo es útil para analizar contratos concretos, comparar sus estrategias inducidas y reutilizar soluciones HJB ya calculadas.

### Caché

El archivo usa un sistema de caché mediante archivos `.pkl`.

Cuando se corre el script por primera vez, se resuelve la HJB para cada contrato y se guarda el resultado. En ejecuciones posteriores, si el archivo `.pkl` ya existe y `force_recompute=False`, el código carga los resultados guardados y no recalcula la HJB.

Esto permite modificar estilos, títulos o gráficas sin repetir el cálculo numérico más costoso.

### Definir una familia de contratos

Para cambiar la familia de contratos, se deben modificar:

```python
FAMILY_NAME = "Nombre de la familia"
```

las funciones:

```python
def contract_1(t):
    ...

def contract_2(t):
    ...
```

y el diccionario:

```python
def get_contracts():
    return {
        "Contrato 1": contract_1,
        "Contrato 2": contract_2,
        ...
    }
```

Las llaves del diccionario son los nombres que aparecerán en consola, en las gráficas y en el caché.

### Funciones principales

#### `run_contracts(...)`

Resuelve la HJB y simula trayectorias para todos los contratos definidos en la familia.

Para hacer comparaciones más limpias, se usan las mismas semillas en todos los contratos. Es decir, la simulación `s` usa la misma semilla para cada contrato.

#### `resimulate_records_with_common_seeds(...)`

Rehace únicamente las simulaciones usando las estrategias óptimas ya guardadas.

Esta función no llama a `solve_hjb`, por lo que no recalcula la HJB.

#### `plot_resumen_contrato_2x2(...)`

Genera una figura 2x2 para un contrato específico, superponiendo varias simulaciones.

La figura contiene:

1. función de ponderación (c(t));
2. estrategia de ejecución del agente (q_t^*);
3. impacto en el precio (I_t);
4. cambios de régimen.

#### `plot_resumenes_2x2_por_contrato(...)`

Genera una figura 2x2 para cada contrato de la familia.

#### `plot_contract_shapes(...)`

Grafica la forma de los contratos (c(t)).

#### `plot_q_strategies(...)`

Grafica las estrategias óptimas simuladas (q^*(t)).

#### `plot_price_impact(...)`

Grafica el impacto en el precio generado por el agente.

#### `plot_full_simulations_per_contract(...)`

Usa la función de graficación del módulo HJB para generar una figura completa por contrato.

#### `main(...)`

Función principal del archivo. Se encarga de:

1. definir o recibir la familia de contratos;
2. cargar resultados desde caché si existen;
3. resolver la HJB si es necesario;
4. simular trayectorias;
5. guardar resultados;
6. generar gráficas.

---

## Instalación

Este repositorio requiere Python y las siguientes librerías:

```text
numpy
scipy
pandas
matplotlib
```
---

## Uso básico

### 1. Clonar el repositorio

```bash
git clone https://github.com/valdo343/principal-agent-regime-switching.git
cd principal-agent-regime-switching
```

### 2. Correr el problema del cliente

Esto evalúa una malla de candidatos beta, resuelve el problema del agente para cada candidato y guarda una tabla con resultados.

El archivo generado es:

```text
resultados_grid_cliente_sencillo.csv
```

### 3. Correr contratos específicos

Este script genera resultados y figuras en la carpeta:

```text
figuras_contratos/
```

Además, guarda un archivo `.pkl` con los resultados de la HJB. Esto permite reutilizar las estrategias óptimas ya calculadas sin volver a resolver la HJB.

---

## Cómo modificar los experimentos

### Cambiar la malla de estados

En `run_contracts.py`, la malla se define en:

```python
def make_grids(n_points: int = 10):
    I_grid = np.linspace(0.0, 5.0, n_points)
    P_grid = np.linspace(75.0, 85.0, n_points)
    Q_grid = np.linspace(0.0, 1.0, n_points)
    return I_grid, P_grid, Q_grid
```

Aumentar `n_points` mejora la resolución de la malla, pero también aumenta el costo computacional.

### Cambiar los parámetros del agente

Los parámetros principales se definen en:

```python
def make_params():
    return {
        "T": 1.0,
        "dtau": 0.05,
        "gamma": 0.015,
        "eta": 1.5,
        "sigma": 1.0,
        "rho": [2.0, 0.8],
        "kappa": [0.8, 2.5],
        "lambda": [2.0, 3.0],
        "A": 115.0,
        "q_max": 10.0,
        "n_q_candidates": 100,
        "tol": 1e-6,
        "max_iter": 20,
        "print_m_matrix_warnings": False,
    }
```

donde:

* `T`: horizonte temporal;
* `dtau`: paso temporal para resolver la HJB;
* `gamma`: aversión al riesgo del agente;
* `eta`: penalización por intensidad de ejecución;
* `sigma`: volatilidad del precio;
* `rho`, `kappa`, `lambda`: parámetros dependientes del régimen;
* `A`: penalización terminal por inventario remanente;
* `q_max`: velocidad máxima permitida;
* `n_q_candidates`: número de valores discretos considerados para el control (q);
* `tol`: tolerancia de convergencia;
* `max_iter`: máximo número de iteraciones de política.

### Recalcular o usar caché

Por defecto, si ya existe un archivo `.pkl`, el código puede reutilizar los resultados guardados.

Para forzar el recálculo de la HJB, se puede llamar:

```python
records = main(force_recompute=True)
```

Para cargar la HJB guardada y solo modificar las gráficas:

```python
records = main(force_recompute=False)
```

---

## Salidas del código

El código puede producir:

1. archivos `.pkl` con resultados de la HJB y trayectorias simuladas;
2. figuras `.png` con contratos, estrategias, impacto en precio y régimen;
3. archivos `.csv` con resultados de búsqueda en el problema del cliente.

La carpeta principal de salida para `run_contracts.py` es:

```text
figuras_contratos/
```

---

## Interpretación general del flujo numérico

El flujo computacional del repositorio puede resumirse como:

```text
Contrato c(t)
    ↓
Problema del agente
    ↓
HJB + iteración de políticas
    ↓
Estrategia óptima q*(t,I,P,Q,régimen)
    ↓
Simulación de trayectorias
    ↓
Evaluación o comparación de contratos
```

En el problema del cliente, este flujo se repite para muchos contratos parametrizados por beta. En `run_contracts.py`, el flujo se aplica únicamente a los contratos definidos manualmente por el usuario.

---

## Autor

**Oswaldo Bueno Rivera**

Centro de Investigación en Matemáticas, A.C. (CIMAT)

Este repositorio forma parte del trabajo desarrollado en mi tesis de maestría. Su propósito es reunir los códigos utilizados para la implementación numérica, simulación y análisis del modelo principal-agente estudiado.

