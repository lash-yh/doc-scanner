# scanner

Revisión de los métodos actuales de [src/core/scanner.py](src/core/scanner.py).
El flujo todavía está en desarrollo. Se distinguen los errores reproducidos,
los problemas de integración y los comportamientos que conviene definir.
No se añadieron las etapas que faltan ni se modificó la implementación.

## `__init__(image)`

Guarda la imagen de entrada y exige que tenga dos dimensiones. Conserva
una referencia al array original, sin copiarlo.

### Problemas de validación confirmados

- `None` y una lista de listas producen `AttributeError` al acceder a
  `.shape`, en lugar de una validación explícita de la entrada.
- Los arrays de forma `(0, 20)` y `(20, 0)` se aceptan aunque no contienen
  píxeles.
- Se aceptan imágenes bidimensionales `uint16`, `float32`, `float64` y
  `bool`, pero el preprocesamiento probado falla después con `cv2.error`.

Estas observaciones no significan que deban admitirse listas o todos los
tipos numéricos. Exponen un contrato de entrada que todavía debe definirse.

Pruebas: `ConstructorDiagnostics`.

### Comportamientos que conviene definir

Definir si solo se permiten imágenes NumPy en gris de tipo `uint8`, o si
otros tipos deben convertirse. Especificar el rango de intensidad, las
dimensiones mínimas y la respuesta ante imágenes vacías.

Decidir si se conserva la referencia a la imagen o una copia. Actualmente,
modificar el array entregado también modifica `self.image`, sin invalidar
los resultados calculados anteriormente.

### Casos regulares verificados

Se conserva la referencia de una imagen válida. Las entradas NumPy
escalares, unidimensionales y de tres canales se rechazan con `ValueError`.

## `scan()`

Actualmente ejecuta únicamente `self.preprocess()` y devuelve `None`.

### Estado actual y componentes pendientes

La prueba confirma que no llama a `hough_transform`, `clac_intersections`
ni `calc_connectivity`. Se considera un flujo pendiente de completar,
no un error de cálculo de esos métodos.

La clase `Frame` ya está definida y se importa sin el bloqueo anterior
de `total_ordering`. `scanner.py` ya la importa, pero todavía no construye
objetos `Frame` ni selecciona marcos por puntuación. Completar esa clase
no incorpora automáticamente esas etapas al escáner.

Prueba: `ScanDiagnostics.test_scan_currently_only_runs_preprocess_and_returns_none`.

### Comportamientos que conviene definir

Definir qué etapas debe coordinar y qué resultado público debe entregar:
estado del proceso, candidatos a documento, imagen rectificada u otro
resultado. Especificar también el tratamiento de una imagen sin documento,
la selección entre varios candidatos y la configuración de cada etapa.

## `preprocess(...)`

Aplica filtro de mediana, ecualización, apertura y cierre morfológicos,
máscara de intensidad, Canny y reconstrucción de contornos exteriores.
Después genera una versión dilatada y una versión erosionada de los bordes.

Guarda `blurred`, `hist_equalized`, `filterred`, `edges_img` y
`edges_img_dilated`; devuelve `None`.

### Error confirmado: resultados posteriores desactualizados

Si se detectan líneas, se cambia la imagen y se vuelve a preprocesar,
los bordes se actualizan, pero `self.lines` y `self.intersections` se
conservan con los valores anteriores.

La prueba sustituye el contenido de una imagen con rectángulo por ceros:
los bordes nuevos quedan vacíos, pero permanecen las líneas del rectángulo
y las intersecciones anteriores. El objeto puede mezclar resultados de
imágenes o configuraciones diferentes.

Prueba: `PreprocessDiagnostics.test_preprocessing_again_should_invalidate_downstream_results`.

### Comportamientos que conviene definir

**Tamaño de los filtros.** `kernel_size` controla la apertura y el cierre;
el filtro de mediana siempre utiliza `25`. No se considera ignorado el
parámetro, porque sí se emplea en la morfología. Definir si la mediana debe
tener un parámetro independiente.

**Tipo e intensidad de los bordes.** El lienzo de contornos se crea con
`np.zeros` sin `dtype`, por lo que `edges_img` y `edges_img_dilated` son
`float64`. En una imagen de un canal, el color `[128, 255, 0]` produce
bordes con valor `128`; la imagen probada contiene valores `{0, 128}`.
Hough y la comprobación `> 0` funcionan con ese formato, por lo que esto
no se presenta como un fallo por sí mismo. Definir si se pretende una
máscara booleana, una imagen `uint8` de valores `0/255` o el formato actual.

**Relación entre dilatación y erosión.** Ambas operaciones parten del
contorno reconstruido, pero la dilatación se calcula antes de modificar
`edges_img` con la erosión. Cambiar `erode_ks` no cambia la imagen dilatada.
Definir si se necesitan dos ramas independientes o una secuencia donde
la dilatación utilice el resultado erosionado.

**Parámetros inválidos.** En sondeos sobre una imagen sintética:

| Entrada | Comportamiento observado |
| --- | --- |
| `kernel_size=0`, `dilate_ks=0` o `erode_ks=0` | La llamada termina sin rechazo explícito. |
| `kernel_size=-1` | `ValueError` de NumPy por dimensiones negativas. |
| `kernel_size=1.5` | `TypeError` al construir el kernel. |
| `intensity_lower=255, intensity_upper=0` | La máscara queda vacía. |
| `canny_lower=70, canny_upper=10` | La llamada termina; no valida el orden. |

Definir los tamaños válidos de los kernels y los rangos y relaciones de
los umbrales. Aceptar un valor sin excepción no demuestra que tenga el
significado esperado por quien llama al método.

**Contornos interiores.** Se utiliza `RETR_EXTERNAL`. Definir si deben
ignorarse siempre los contornos contenidos dentro de otros, especialmente
cuando existan varios documentos o un documento sobre un marco visible.

### Casos regulares verificados

El preprocesamiento conserva la imagen de entrada y las dimensiones de
las salidas. La imagen con rectángulo genera bordes finitos; la imagen
vacía genera cero bordes. También pasa la prueba que excluye todos los
píxeles mediante una máscara de intensidad.

## `hough_transform(err, threshold, ks)`

Detecta líneas sobre `edges_img` y las clasifica en `v`, `h` y `o`, según
el ángulo de su normal. Guarda y devuelve el diccionario `self.lines`.
`threshold` multiplica el máximo del acumulador de Hough.

### Error confirmado: conserva intersecciones de líneas anteriores

Una llamada nueva reemplaza `self.lines`, pero no invalida las
intersecciones existentes. La prueba obtiene familias vacías sobre bordes
vacíos y confirma que permanecen las intersecciones previas.

Prueba: `HoughDiagnostics.test_new_hough_results_should_invalidate_old_intersections`.

### Problemas confirmados de parámetros y estado

`ks` no se utiliza. Se probaron `ks=1` y `ks=9`: ambos envían los mismos
argumentos a `hough_line_peaks`, incluidos `min_distance=10` y
`min_angle=50`. No está definido qué debería controlar este parámetro.

Un `threshold=-1` produce `OverflowError` en la detección de picos con las
dependencias instaladas. `threshold=2` terminó sin líneas en el sondeo.
Los valores `err=0` y `err=-1` clasificaron todas las líneas detectadas
como oblicuas.

Invocar el método antes del preprocesamiento produce `AttributeError`
porque no existe `self.edges_img`.

### Comportamientos que conviene definir

Definir el significado de `ks` o retirarlo de la interfaz cuando se decida
el contrato. Validar el rango de `threshold` y la tolerancia angular `err`.

Las comparaciones angulares usan `< err`: un ángulo exactamente en el
límite queda fuera de la familia correspondiente. La prueba con
`phi=-pi/12` y la tolerancia predeterminada lo clasifica como oblicuo.
Definir si los extremos deben incluirse y qué rotaciones del documento
debe soportar el detector.

Decidir si cada método requiere explícitamente la etapa anterior o debe
ejecutarla cuando falte. Si se exige orden, proporcionar una excepción
que explique el requisito, en lugar de depender de atributos ausentes.

### Casos regulares verificados

Una imagen de rectángulo detecta al menos dos líneas verticales y dos
horizontales. Una imagen vacía devuelve las tres familias vacías.
Se verificó la clasificación de ángulos controlados, la conservación de
`rho` y que `threshold=0.3` se convierte en `3` cuando el máximo del
acumulador es `10`.

## `clac_intersections()`

Forma el producto cartesiano entre las familias vertical y horizontal,
calcula cada cruce y crea objetos `Intersection`. Guarda y devuelve
`self.intersections`. Se conserva el nombre público actual, incluido
`clac`, sin renombrarlo durante esta revisión.

### Error confirmado: falta de líneas

Cuando alguna de las dos familias está vacía, asigna
`self.intersections = []`, pero intenta devolver `self.intersetions`.
La llamada termina con `AttributeError`.

Se reprodujo con ambas familias vacías, solo verticales y solo horizontales.
Este caso debe ser normal cuando no se detecta un documento.

Prueba: `IntersectionDiagnostics.test_empty_or_missing_line_family_returns_empty_list`.

### Error confirmado: vertical exacta produce una intersección no finita

Con las líneas `v=[(0, 20)]` y `h=[(pi/2, 10)]`, el cruce esperado es
`(20, 10)`, pero el método crea una intersección `(NaN, NaN)`.

El cálculo convierte primero las líneas polares a puntos usando dos
abscisas. Esa representación falla para una vertical exacta y los valores
no finitos se propagan a las ecuaciones cartesianas. El método no rechaza
el resultado ni utiliza una representación que evite esa singularidad.

Prueba: `IntersectionDiagnostics.test_exact_vertical_line_should_produce_finite_intersection`.

### Comportamientos que conviene definir

Se conservan intersecciones fuera de la imagen: una horizontal `y=100`
produce un cruce con esa coordenada en una imagen de altura `40`.
Definir si deben descartarse, mantenerse como candidatos o admitirse con
un margen configurable. También conviene definir el tratamiento de líneas
paralelas, cruces no finitos y candidatos duplicados.

El método requiere `self.lines`; sin una detección anterior genera
`AttributeError`. Con pares disponibles también necesita `edges_img`
para obtener el ancho. Definir y validar esas precondiciones.

Los nombres locales `lines_h` y `lines_v` están intercambiados respecto
a las familias extraídas. No se confirmó un error de coordenadas causado
por esos nombres: las pruebas de cruces inclinados y metadatos pasan.
Corregir los nombres puede mejorar la lectura sin alterar la matemática.

### Casos regulares verificados

Dos líneas inclinadas de la familia vertical y dos horizontales generan
cuatro cruces con las coordenadas esperadas. Cada objeto conserva sus
líneas vertical y horizontal correctas y las abscisas `(0, ancho)`.

## `calc_connectivity()`

Asigna `edges_img_dilated` como imagen de cada intersección, llama a su
método `connectivity()` y devuelve la misma lista de objetos.

### Error confirmado de integración: falta la longitud de muestreo

Los objetos creados por `clac_intersections` conservan
`along_length=None`. `calc_connectivity` no establece ese valor ni lo
proporciona al llamar a `connectivity`.

Incluso con líneas inclinadas válidas e imagen dilatada disponible, la
secuencia termina en:

```text
TypeError: unsupported operand type(s) for /: 'NoneType' and 'float'
```

Asignar manualmente `along_length=4` a la intersección permite completar
la prueba sobre una imagen de píxeles encendidos. Esto identifica la
configuración ausente, sin resolver los otros defectos de `Intersection`.

Pruebas: `ConnectivityDiagnostics.test_real_intersection_flow_needs_a_sampling_length`
y `test_manually_configured_sampling_length_allows_calculation`.

### Error confirmado de integración: reutilización de conectividad antigua

Asignar una imagen nueva no invalida la caché del objeto `Intersection`.
La prueba calcula conectividad uno, reemplaza la imagen dilatada por
ceros y vuelve a llamar a `calc_connectivity`. La imagen se actualiza,
pero el resultado continúa siendo uno en las cuatro direcciones.

La corrección debe coordinarse con la invalidación de caché descrita en
[Intersection.md](Intersection.md).

Prueba: `ConnectivityDiagnostics.test_new_dilated_image_should_recompute_existing_connectivity`.

### Comportamientos que conviene definir

Definir dónde se configura `along_length`, si depende del tamaño de la
imagen y cómo se transmite `width`. Determinar cuándo se recalculan las
intersecciones existentes al cambiar la imagen o los parámetros.

Sin `self.intersections`, el método genera `AttributeError`. Con una
lista no vacía también requiere `edges_img_dilated`. Definir la respuesta
ante etapas ausentes y ante una intersección inválida dentro de la lista:
detener todo el cálculo, omitir el candidato o registrar un resultado
fallido explícito.

### Casos regulares verificados

Una lista vacía se devuelve sin error. Con objetos controlados se
comprueba que cada uno recibe la imagen dilatada y una llamada a
`connectivity`. También se verificó el cálculo real con la longitud de
muestreo configurada manualmente.

## Verificación y límites del diagnóstico

Las pruebas nuevas están en
[test/test_scanner_diagnostics.py](test/test_scanner_diagnostics.py).
Se utilizan imágenes NumPy sintéticas y geometría determinista. Algunos
casos controlan los picos de Hough o las llamadas a otras etapas para
aislar el comportamiento de un método; otros ejecutan el cálculo real.

La nueva revisión confirma que `Frame` ya no bloquea la importación por
`total_ordering`: existen operaciones de igualdad y orden, y se generan
los métodos derivados. La carga normal desde la raíz sigue afectada por
las importaciones mixtas ya conocidas, un problema independiente.

Se retiró el aislamiento que recortaba la definición de `Frame`. Las
pruebas cargan ahora los archivos completos en módulos privados y solo
resuelven temporalmente los alias de importación. `Frame`, `Intersection`
y los métodos de `scanner` ejecutan su código actual. Esto permite probar
su comportamiento sin afirmar que las importaciones normales estén resueltas.

La integración nueva se prueba en [test/test_frame.py](test/test_frame.py):
se construye un `Frame` a partir de intersecciones reales del escáner.
Con una longitud de muestreo configurada manualmente, la conectividad y
una puntuación numérica pueden calcularse. La misma prueba de geometría
confirma que `Frame.area()` devuelve aproximadamente `384.2122` para
un rectángulo girado cuya área real es `400`.
El diagnóstico de esa clase está en [Frame.md](Frame.md).

Las expectativas de `ValueError` para entradas o etapas inválidas, y de
invalidación de resultados, expresan el comportamiento propuesto para
hacer explícito el contrato. Deben distinguirse de los casos numéricos
incorrectos y las excepciones reproducidas con entradas válidas.

```bash
uv run python -m unittest test.test_scanner_diagnostics -v
```

La comprobación también puede ejecutarse con el intérprete del entorno:

```bash
.venv/bin/python -m unittest test.test_scanner_diagnostics -v
```

La ejecución aislada realizó 32 pruebas: 20 pasaron y 12 detectaron
problemas. El resumen de `unittest` fue `failures=7, errors=10`; estos
contadores incluyen las subpruebas y no representan 17 métodos distintos.
Los errores y fallos permanecen reproducibles; no se marcaron como éxitos
esperados ni se ocultaron mediante cambios en producción.

El descubrimiento completo ya puede cargar los módulos de pruebas
anteriores: desaparecieron los errores de carga causados por `Frame`.
La suite sigue fallando por los casos documentados de `Frame`, del
escáner y de las dependencias previamente revisadas. Los resultados de
la nueva revisión completa se registran en [Frame.md](Frame.md).
