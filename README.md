# AAGE · Práctica 2 (Parte I): aprendizaje federado desde cero

Implementación en PyTorch, sin frameworks de FL, del bucle completo de aprendizaje federado sobre
Fashion-MNIST: reparto de los datos entre 10 clientes (IID y no-IID), entrenamiento local,
agregación con FedAvg y FedProx, y medida del coste de comunicación.

El código base incluye la carga de datos, la configuración, el registro de resultados y las
gráficas. Las partes propias del aprendizaje federado (modelos, reparto de datos, entrenamiento
local, agregación, la ronda federada y la medida de la comunicación) quedan por implementar.

## Instalación

Requiere Python 3.10 o superior; basta con la CPU. Los paquetes necesarios son `torch`,
`torchvision` (descarga Fashion-MNIST), `matplotlib`, `pyyaml` y `pytest`, y están en
`requirements.txt`. En Linux sin GPU se recomienda instalar antes la versión de torch para CPU,
que ocupa mucho menos:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

```bash
# Opción A: conda (el entorno con conda, los paquetes con pip)
conda create -n aage python=3.11
conda activate aage
pip install -r requirements.txt

# Opción B: venv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

En **Google Colab**: subir la carpeta, situarse en ella con `%cd` y ejecutar `!pip install pyyaml`
(el resto ya viene instalado). Después, `!python run.py`.

Fashion-MNIST se descarga automáticamente en `data/` la primera vez (~30 MB).

## Ejecutar un experimento

```bash
python run.py                                         # configuración por defecto
python run.py --config configs/examples/iid.yaml      # un ejemplo
python run.py --set partition.alpha=1.0 client.local_epochs=5 server.fraction=1.0
```

Toda ejecución parte de [configs/default.yaml](configs/default.yaml), donde están comentadas
todas las opciones. `--config` y `--set` solo indican lo que cambia, lo que facilita los barridos:

```bash
for a in 0.1 1.0 10; do python run.py --set partition.alpha=$a; done
python plot.py runs/* -o runs/alpha.png
```

Referencias de comparación:

| Referencia | Comando |
|---|---|
| Centralizado (techo): un único cliente con todos los datos | `python run.py --config configs/examples/centralized.yaml` |
| Solo local (suelo): cada cliente entrena por su cuenta | `python run.py --config configs/examples/local_only.yaml` |

## Resultados

Cada ejecución crea `runs/<nombre>/` (nombre automático a partir de la configuración, o
`--set output.name=...`). Si la carpeta ya existe, se sobrescribe.

- `config.json`: configuración completa.
- `metrics.csv`: una fila por ronda: `round, test_acc, test_loss, train_loss, bytes_up, bytes_down, bytes_cum, time_s`.
  El modelo global se evalúa siempre sobre el test completo de Fashion-MNIST.
- `partition.png`: número de ejemplos de cada clase en cada cliente.

`python plot.py runs/a runs/b ...` dibuja la accuracy y la pérdida por ronda y la accuracy frente a
los MB transmitidos.

## Estructura

```
run.py              punto de entrada: lee la config y monta el experimento
plot.py             gráficas a partir de los metrics.csv
configs/            default.yaml + ejemplos
fl/
├── data.py         carga de datos y particionado
├── models.py       MLP, CNN, evaluate(), count_params()
├── server.py       Server: la ronda federada; LocalOnly: referencia sin comunicación
├── clients/        entrenamiento local: un archivo por tipo de cliente
│   ├── base.py         Client (FedAvg)
│   └── fedprox.py      FedProxClient
├── aggregators/    agregación en el servidor: un archivo por método
│   ├── base.py         Aggregator (interfaz)
│   └── fedavg.py       FedAvg
├── codecs/         codificación de lo que viaja por la red
│   └── base.py         Codec (interfaz; por defecto, sin compresión) y state_dict_nbytes
└── utils.py        configuración, semillas y registro
tests/public/       tests públicos (uno por hueco)
```

### Huecos

Los huecos lanzan `NotImplementedError("TODO: ...")` y su docstring describe el comportamiento
esperado. Orden recomendado:

| # | Archivo | Qué |
|---|---|---|
| 1 | `fl/data.py` | `iid`, `dirichlet`, `dirichlet_shards` |
| 2 | `fl/models.py` | `MLP` y `CNN` (diseño libre, ≤ 100 000 parámetros) |
| 3 | `fl/codecs/base.py` | `state_dict_nbytes` |
| 4 | `fl/server.py` | `sample_clients` y `Server.run_round` |
| 5 | `fl/aggregators/fedavg.py` | `FedAvg.aggregate` |
| 6 | `fl/clients/base.py` | `Client.local_train` |
| 7 | `fl/clients/fedprox.py` | `FedProxClient.loss` |
| 8 | `fl/server.py` | `LocalOnly.run_round` |

Con los seis primeros, `python run.py` ya ejecuta un experimento completo.

### Métodos nuevos

Los métodos que no tienen hueco, como la **cuantización de la subida**, siguen la misma
convención:

- Cada método es una clase en su propio archivo, dentro de la carpeta correspondiente, y hereda de
  la interfaz de esa carpeta (`Client`, `Aggregator` o `Codec`).
- Se registra con un nombre en el `__init__.py` de la carpeta (`build_clients`, `build_aggregator`
  o `build_codec`) para poder elegirlo desde la configuración, p. ej. `--set server.codec=int8`.
- Sus parámetros, si los tiene, se añaden a `configs/default.yaml`.

La cuantización debe estar disponible como `server.codec: int8`; un test público lo comprueba.

**No se deben cambiar las firmas de las funciones con hueco** ni el formato de `metrics.csv`: los
tests y la corrección dependen de ellos.

## Tests

```bash
pytest                      # todos
pytest tests/public/test_data.py -v
```

Los tests públicos no cubren todos los casos; la corrección incluye tests adicionales.

## De Keras a PyTorch

<details>
<summary>Guía rápida (pulsar para desplegar)</summary>

Las ideas son las mismas; la diferencia principal es que en PyTorch el bucle de entrenamiento se
escribe explícitamente (lo que permite, por ejemplo, añadir el término de FedProx).

| Keras | PyTorch |
|---|---|
| `keras.Sequential([...])` | `nn.Sequential(...)` |
| `layers.Dense(200, activation="relu")` | `nn.Linear(n_entrada, 200), nn.ReLU()` |
| `layers.Conv2D(16, 3, padding="same")` | `nn.Conv2d(canales_entrada, 16, 3, padding=1)` |
| `layers.MaxPooling2D(2)` / `layers.Flatten()` | `nn.MaxPool2d(2)` / `nn.Flatten()` |
| `loss="sparse_categorical_crossentropy"` + softmax final | `F.cross_entropy(logits, y)` (sin softmax: recibe logits) |
| `optimizer=SGD(learning_rate=0.01)` | `torch.optim.SGD(model.parameters(), lr=0.01)` |
| `model.fit(x, y, batch_size=32, epochs=E)` | el bucle de abajo |
| `model.evaluate(x, y)` | `evaluate(model, dataset)` (en `fl/models.py`) |
| `model.get_weights()` / `set_weights(w)` | `model.state_dict()` / `model.load_state_dict(sd)` |
| `clone_model(m)` + `set_weights(...)` | `copy.deepcopy(model)` |
| `model.count_params()` | `count_params(model)` (en `fl/models.py`) |
| `tf.Tensor`, `.numpy()` | `torch.Tensor`, `.item()` para un escalar |

Equivalente a `model.fit` (las imágenes ya vienen como `(canales, alto, ancho)`):

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)   # mini-lotes barajados
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.train()                                               # modo entrenamiento
for epoch in range(E):
    for x, y in loader:
        optimizer.zero_grad()                # borra los gradientes del paso anterior
        loss = F.cross_entropy(model(x), y)  # pasada hacia delante
        loss.backward()                      # calcula los gradientes
        optimizer.step()                     # actualiza los pesos
```

Otras operaciones útiles: `model.parameters()` recorre los tensores entrenables;
`tensor.detach().clone()` hace una copia que no participa en el cálculo de gradientes;
`(a - b) ** 2` y `.sum()` funcionan como en NumPy.

</details>

## Tiempos de referencia

Configuración por defecto (30 rondas, 5 de 10 clientes por ronda, 1 época local), solo CPU
(sobremesa de 16 hilos): **MLP ≈ 20 s**, **CNN de referencia ≈ 75 s**. En un portátil, entre 2 y
4 veces más. Con GPU disponible, `device: auto` la utiliza.
