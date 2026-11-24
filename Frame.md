# Frame

Revisión de la clase `Frame` en [src/core/entities.py](src/core/entities.py)
y de su integración con [src/core/scanner.py](src/core/scanner.py).
La revisión nueva de `entities.py` se limita a `Frame`; los defectos
anteriores de `Intersection` no se vuelven a analizar aquí.

## Estado comprobado

`Frame` ya define `__eq__` y `__lt__`, por lo que `total_ordering` deja
de impedir la carga de `entities.py`. Se verificaron las comparaciones
derivadas `<=`, `>` y `>=` con puntuaciones numéricas distintas.

Se retiró de las pruebas del escáner el recorte que excluía `Frame`.
Ahora se cargan ambos archivos completos. Los alias temporales siguen
siendo necesarios por las importaciones mixtas ya conocidas.

Resolver el decorador no resuelve los errores de geometría, puntuación
o integración descritos a continuación.

## `__init__(...)`

Guarda las cuatro intersecciones en el orden superior izquierda,
superior derecha, inferior derecha e inferior izquierda. Inicializa
la caché de puntuación en `None` y conserva `image_shape`.

### Comportamientos que conviene definir

No hay validación explícita de tipos, coordenadas, orden ni geometría.
Definir si deben exigirse cuatro esquinas distintas y finitas que formen
un rectángulo, un paralelogramo o cualquier cuadrilátero convexo.

`image_shape` es opcional, pero `score`, `relative_area` y las comparaciones
requieren dimensiones disponibles. Definir si el constructor debe
exigirlas o si estos métodos deben pedirlas cuando aún no existen.

Las esquinas se conservan por referencia. Definir si pueden modificarse
después de crear el marco y cómo afecta eso a la puntuación y al hash.

### Casos regulares verificados

Se conserva el orden de las esquinas y la forma de imagen entregada.

## `score(image_shape=None)`

Suma la puntuación de orientación correspondiente a cada esquina,
ponderada por el área relativa del marco.

### Error confirmado: devuelve `None` en la primera llamada con la forma actual

La condición de retorno de caché exige `self.__score is None` en lugar
de exigir una puntuación ya calculada. Si la forma explícita coincide
con la guardada, devuelve inmediatamente ese `None`.

Para un rectángulo de área `50`, imagen `(100, 100)` y puntuación uno
en cada esquina:

```python
frame.score()            # 0.02
```

En otra instancia equivalente, antes de calcular ninguna puntuación:

```python
frame.score((100, 100))  # None; debería producir la misma puntuación numérica.
```

Prueba: `FrameScoreTests.test_first_score_with_current_explicit_shape_returns_number`.

### Error confirmado: falla al cambiar las dimensiones explícitamente

`score((200, 100))` llama a `relative_area` dentro del bucle de esquinas.
La primera llamada actualiza la forma guardada. La segunda recibe la
misma forma y genera el `ValueError` descrito en `relative_area`.

También falla una llamada explícita con la forma original después de
haber calculado `score()` sin argumentos.

Pruebas: `FrameScoreTests.test_score_with_new_shape_updates_and_completes`
y `test_explicit_shape_after_previous_score_remains_valid`.

### Problema confirmado: la caché no se reutiliza en llamadas sin argumentos

Dos llamadas consecutivas a `score()` con las mismas entradas consultan
dos veces la orientación de cada esquina y recalculan el área relativa.
La puntuación almacenada no evita ese trabajo. Esta observación es un
problema de reutilización, no un resultado numérico incorrecto por sí solo.

Prueba: `FrameScoreTests.test_unchanged_inputs_reuse_score_cache`.

### Comportamientos que conviene definir

Definir qué cambios invalidan la puntuación: dimensiones, coordenadas,
orientaciones o imagen de las intersecciones. La normalización por área
debe resolverse de forma consistente para las cuatro esquinas.

Definir también el significado y rango esperado de la puntuación, y cómo
tratar orientaciones ausentes o no finitas.

### Casos regulares verificados

Sin forma explícita y con dimensiones guardadas, el rectángulo de prueba
obtiene `0.02`. Con orientaciones controladas se verifica que la esquina
de índice `i` utiliza el componente `i` de su vector de orientación.
Esto no garantiza que el vector producido por `Intersection` sea correcto.

## `area()`

### Error confirmado: área incorrecta para rectángulos girados

El método utiliza una diferencia vertical y una diferencia horizontal
como si fueran longitudes de lados, y las multiplica por el seno del
ángulo entre las normales de las líneas. Esas diferencias son proyecciones
sobre los ejes; no son las longitudes reales cuando el marco está girado.

```text
Esquinas: (10, 0), (20, 10), (10, 20), (0, 10)
Área real: 200
Frame.area(): 100
```

El caso también se reproduce con objetos `Intersection` reales obtenidos
del escáner: dos familias ortogonales, separadas `20` unidades cada una
y giradas `0.2` radianes, forman un rectángulo de área `400`.
`Frame.area()` devuelve aproximadamente `384.2121988`.

Pruebas: `FrameGeometryTests.test_rotated_square_area_is_rotation_invariant`
y `FrameScannerIntegrationTests.test_frame_from_real_scanner_intersections_has_correct_area`.

### Error confirmado: desbordamiento con coordenadas `uint8`

La resta y el producto conservan el tipo sin signo de las coordenadas.
Los valores negativos y los productos mayores de `255` se desbordan.

Para `(1, 1), (21, 1), (21, 21), (1, 21)` con componentes `np.uint8`,
el resultado es `144` en lugar de `400`. El mismo rectángulo con enteros
ordinarios produce el área correcta.

Prueba: `FrameGeometryTests.test_unsigned_coordinates_do_not_overflow_differences`.

### Comportamientos que conviene definir

Decidir qué geometrías admite el marco y cómo se normalizan los tipos
numéricos antes de operar. El cálculo actual ni siquiera consulta la
esquina inferior derecha, por lo que no describe el área de un
cuadrilátero general. Independientemente del contrato elegido, girar un
rectángulo no debe cambiar su área.

## `relative_area(image_shape=None)`

### Error confirmado: rechaza dimensiones que ya están configuradas

La condición inicial lanza `ValueError("Image area not set yet")` cuando
el argumento coincide con `self.image_shape` y no es `None`. El mensaje
contradice el estado: precisamente en ese caso la forma ya está guardada.

Con imagen `(100, 100)` y área `50`, `relative_area()` devuelve `0.005`,
pero `relative_area((100, 100))` genera la excepción.

Entregar una forma distinta funciona una vez y actualiza el objeto;
repetir la misma llamada falla. Este cambio de estado provoca también
el fallo de `score` cuando itera por varias esquinas.

Pruebas: `FrameRelativeAreaTests.test_explicit_current_shape_is_valid`
y `test_explicit_new_shape_can_be_reused`.

### Problemas de validación y comportamientos que conviene definir

- Sin dimensiones, la llamada sin argumentos genera `TypeError` al
  indexar `None`.
- Con forma `(0, 100)`, el caso probado divide entre cero y produce
  infinito con una advertencia de NumPy.
- Con forma `(-100, 100)`, devuelve un área relativa negativa.

Definir el formato de `image_shape`, exigir dimensiones válidas y
decidir si cambiar la forma mediante este método debe modificar el
objeto. También debe definirse si un marco exterior a la imagen puede
tener área relativa superior a uno.

## `__eq__`, `__ne__`, `__lt__` y `__hash__`

### Error confirmado: igualdad y hash usan criterios incompatibles

La igualdad compara solo `score()`, mientras el hash utiliza las
coordenadas de las esquinas. Dos marcos trasladados, con las mismas
dimensiones, área y orientaciones, obtienen la misma puntuación pero
hashes diferentes.

```text
primer_marco == marco_trasladado  -> True
hash(primer_marco) == hash(marco_trasladado)  -> False
len({primer_marco, marco_trasladado})  -> 2
```

Esto incumple el requisito de que los objetos considerados iguales
tengan el mismo hash. El comentario de `__hash__` no coincide con el
comportamiento reproducido.

Pruebas: `FrameComparisonTests.test_equal_frames_have_equal_hashes`
y `test_set_deduplicates_frames_that_compare_equal`.

### Problema confirmado: comparaciones con otros tipos

`frame == object()` y `frame.__lt__(object())` generan `AttributeError`
porque llaman a `other.score()` sin comprobar compatibilidad. Definir
los tipos comparables y devolver `NotImplemented` cuando corresponda,
para permitir el comportamiento normal de los operadores de Python.

### Comportamientos que conviene definir

Separar el significado de igualdad del criterio usado para ordenar
candidatos. Decidir si dos marcos son iguales por geometría, identidad
o puntuación, y hacer coherente el hash con esa decisión. Si los datos
que determinan la igualdad pueden cambiar, definir si los objetos
deberían ser utilizables como claves o miembros de conjuntos.

Las comparaciones dependen de `score()`, por lo que también requieren
una forma de imagen válida y datos de orientación disponibles.

### Casos regulares verificados

Con esquinas controladas, formas configuradas y puntuaciones distintas,
la igualdad, desigualdad, ordenación y operaciones generadas por
`total_ordering` funcionan en los casos probados.

## `coordinates()`

Devuelve una tupla con las coordenadas de las cuatro esquinas en el
orden del constructor. La prueba de orden pasa.

Definir si debe devolver valores normalizados o copias independientes.
Actualmente agrega directamente cada `corner.intersection`, que puede
ser mutable dependiendo de la entrada.

## `__repr__()`

Devuelve la cadena constante `"rectangle"`. La prueba pasa y no se
confirmó un error funcional. Definir si esa representación basta para
inspeccionar distintos marcos durante la depuración.

## `summary()`

Solicita el resumen de cada esquina mediante
`summary(one_line=True, println=False)`, agrega sus etiquetas de
orientación y el área, e imprime el texto. Devuelve `None`.

Con esquinas controladas, las etiquetas, llamadas y salida impresa pasan
la prueba. El área mostrada hereda los errores de `area()` y los resúmenes
reales dependen de la conectividad de las intersecciones.

Definir si también debe poder devolver una cadena sin imprimir, y cómo
mostrar esquinas cuyo cálculo no esté disponible.

## Integración y verificación

Las pruebas están en [test/test_frame.py](test/test_frame.py). Los casos
unitarios utilizan esquinas controladas para aislar `Frame`, y los casos
de integración construyen el marco con intersecciones reales del escáner.
No se modificaron `Frame`, `Intersection` ni `scanner` para conseguir
que las pruebas pasaran.

```bash
.venv/bin/python -m unittest test.test_frame -v
.venv/bin/python -m unittest test.test_scanner_diagnostics -v
.venv/bin/python -m unittest discover -s test -v
```

Las 27 pruebas de `Frame` produjeron 11 éxitos, 9 fallos y 7 errores.
Parte de esos casos expresa contratos propuestos, como la validación de
dimensiones y la reutilización de caché; se distinguen de los resultados
numéricos incorrectos y las excepciones con entradas válidas.

Las 32 pruebas específicas de `scanner` mantienen el resultado anterior:
20 éxitos, con `failures=7, errors=10` al contar subpruebas.
Siguen presentes el atributo mal escrito en el caso sin líneas, los
cruces no finitos con verticales exactas, la falta de longitud de muestreo
y los resultados desactualizados. El detalle está en [scanner.md](scanner.md).

La suite completa ejecutó 117 pruebas, con `failures=30, errors=26`,
incluidas subpruebas. Ya no hay errores de carga por una clase `Frame`
incompleta. La suite incluye también los fallos de importación y los
defectos de `Intersection` y geometría documentados previamente.

`scanner.py` ya importa `Frame`, pero todavía no construye objetos de esa
clase. Las pruebas ensamblan esa integración explícitamente y configuran
la longitud de muestreo para comprobar el cálculo real. La finalización
de `Frame` no completa automáticamente el flujo de `scan()`.
