# Memoria y rendimiento

OmniLRS sigue siendo el simulador completo: Lunalab, Lunaryard, LargeScale, deformación, cráteres, rocas, clipmaps, sensores, ROS 2, Yamcs, Zenoh, SDG, física y el motor estelar. Los cambios reducen copias y acotan cachés. No eliminan sistemas.

Isaac Sim no se ejecutó en esta máquina, así que no hay una medición de FPS, VRAM ni del RSS del proceso de Kit. El objetivo de “estable en 6 GiB de RAM de sistema” **no está alcanzado**.

## Límite externo

La [documentación de Isaac Sim 5.0](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html) pide como mínimo **32 GiB de RAM** y **16 GiB de VRAM**. El runtime de Kit, RTX y PhysX vive fuera de este repositorio. Recortar OmniLRS no baja ese piso: en un equipo de 6 u 8 GiB el sistema operativo ya ocupa una parte, y el proceso de Isaac Sim por sí solo supera lo que queda.

Hay que separar tres cifras:

- **RAM del sistema**: instalada, y la que el sistema operativo aún puede entregar (`available`).
- **Working set del proceso**: RAM física cargada a ese proceso en ese instante.
- **Commit del proceso**: memoria reservada, incluida la que puede estar en el archivo de paginación.

Al arrancar, OmniLRS escribe las tres. Una bajada del working set no significa que el equipo tenga más RAM libre.

## Qué consume memoria

Cifras **calculadas** a partir del tamaño de los `np.zeros` / `np.empty` del código, con la configuración por defecto de LargeScale (`num_blocks=4`, bloque de 50 m, 0.025 m/px):

| Reserva | Tamaño |
| --- | --- |
| Ventana del DEM de alta resolución, float32 | 22000² × 4 = **1.803 GiB** |
| La misma ventana con `num_blocks=2` | **0.730 GiB** |
| Bloque de cráteres con padding, float64 | 2800² × 8 = **59.8 MiB** por bloque en vuelo |
| El mismo bloque en float32 | **29.9 MiB** |

El clipmap fino se dimensiona a partir de esa ventana, así que reducir `num_blocks` no deja un agujero: acorta el vecindario fino y el terreno lejano sigue en el clipmap grueso. La resolución en metros por píxel no cambia. El número de bloques no baja por debajo del `block_span` de las rocas, para no muestrear elevación fuera del DEM.

Ocho procesos de cráteres más un proceso de interpolación duplican el intérprete y las splines. En Windows cada uno es un proceso nuevo, no un hilo. El perfil bajo los deja en 2 y 1. El algoritmo de cráteres es el mismo.

`samples_per_pixel_per_frame: 32` en `RayTracedLighting` reserva buffers de RTX propios de un trazador de caminos. En el perfil bajo pasa a 1 muestra por píxel, que es el régimen de tiempo real de RTX. El modo de render no cambia. En un equipo con RAM de sobra se conserva el 32 del yaml.

`multi_gpu: true` con una sola GPU crea un contexto que no aporta un segundo dispositivo. Se apaga cuando `nvidia-smi` ve una GPU, también en el perfil alto.

## Medido en esta máquina

Medido el 22 de octubre de 2026 con Python 3.11, sin Isaac Sim. RAM instalada **31.129 GiB**. En ese momento el sistema solo tenía **0.661 GiB disponibles** y el proceso de la medición ocupaba **25.9 MiB** de working set (commit **757 MiB**). Con el perfil `auto`, este arranque concreto usaría el perfil bajo por la RAM libre, no por los 31 GiB instalados.

Malla del terreno de Lunaryard, 2000×2000 vértices (40 m a 2 cm):

- Constructor anterior, pico de `tracemalloc` medido en una malla de 250×250 y extrapolado por número de celdas: **2119 MiB**. La extrapolación es lineal y el constructor reserva un objeto Python por vértice y por índice.
- Constructor nuevo, working set medido al construir la malla de 2000×2000: **350.2 MiB**. La suma de los arrays es **350.7 MiB**.

Malla de Lunalab, 1000×650:

- Extrapolación del constructor anterior: **344 MiB**.
- Arrays nuevos: **56.9 MiB**.

Desplazamiento del DEM de alta resolución sobre un campo float32 de 1500×1500 (**8.58 MiB**): el working set no subió (**0.0 MiB**). No se reservó la ventana de 1.8 GiB porque no había RAM libre para hacerlo. El desplazamiento de esa ventana usa el mismo bucle y un buffer de una fila.

Los tests de `test/test_memory_optimizations.py` comprueban que la topología de la malla y las cotas desplazadas coinciden con el comportamiento anterior.

## Perfil automático

`OMNILRS_MEMORY_PROFILE` puede ser `auto` (defecto), `low`, `balanced`, `high` u `off`.

`auto` mira la RAM instalada y, si queda poca libre, baja un escalón:

| Condición | Perfil |
| --- | --- |
| Menos de 12 GiB instalados, o menos de 2 GiB libres | low |
| Menos de 24 GiB instalados, o perfil alto con menos de 4 GiB libres | balanced |
| 24 GiB o más, con RAM libre | high |

`off` y `high` no recortan muestras, bloques ni workers. `high` sí apaga multi-GPU si solo hay una GPU.

En el perfil bajo, además: viewport 1280×720 si el yaml no lo fija, hilos de PhysX acotados a 2, colas de los workers más cortas, clipmap grueso en modo híbrido (el DEM de baja resolución no se copia a VRAM) y streaming de texturas más el eco mode de RTX cuando Kit expone esas claves. Las texturas, los materiales y el modo de render siguen siendo los del yaml.

En un equipo con 24 GiB o más y varios GiB libres no se aplica ninguno de esos topes. Las copias que se eliminaron siguen valiendo en ese equipo: la malla del patio, el DEM en float32, el desplazamiento de la ventana sin un segundo buffer y el material MDL, que ahora se compone una sola vez. Los prototipos de roca se comparten solo cuando el archivo, los colisionadores, el material y la etiqueta coinciden. En LargeScale por defecto las dos capas de rocas no coinciden en colisionadores, así que cada una conserva los suyos.

## Arranque

El log `omnilrs.memory` indica el perfil, la RAM de sistema y cada tope aplicado. Después de construir la escena registra working set y commit del proceso junto a la RAM disponible del sistema.

## Fase 2: arranque de Isaac Sim 5.0

Isaac Sim 5.0 en modo standalone ya abre `isaacsim.exp.base.python.kit`. Ese kit es el de Python. `isaacsim.exp.full.kit` es el del editor y carga más extensiones. OmniLRS deja `experience` vacío cuando ese archivo base no está en `EXP_PATH`, que es el mismo resultado que el constructor de `SimulationApp`. Si `EXP_PATH` apunta al kit base y el archivo existe, se pasa esa ruta. La ruta automática nunca elige el kit full. Un `OMNILRS_EXPERIENCE` explícito sí se respeta.

En un equipo con **menos de 12 GiB instalados**, y con el perfil `auto` o `low`, el arranque pasa a headless, oculta la UI y activa `disable_viewport_updates`. Esa bandera, documentada por NVIDIA, evita refrescar el viewport por defecto mientras la sesión ya es headless. Las cámaras y los sensores RTX siguen creando sus propios render products. El bucle de ejecución de ROS 2, Yamcs y SDG sigue en `world.step(render=True)`. Zenoh sigue bombeando `simulation_app.update()`. `OMNILRS_KEEP_WINDOW=1` conserva la ventana en ese equipo. `OMNILRS_MEMORY_PROFILE=high` u `off` también la conservan, aunque el equipo tenga poca RAM instalada. SDG no entra en ese camino: sigue con viewport, con el cierre de Replicator y con los 100 frames renderizados de calentamiento, porque la generación de imágenes depende de ese render.

El calentamiento de ROS 2, Zenoh y Yamcs en ese mismo equipo de menos de 12 GiB instalados hace **98 pasos de física sin render y 2 con render**. Sigue habiendo 100 pasos del solver por cada llamada, con el mismo `physics_dt`. Los 100 pasos anteriores renderizaban el viewport en todos. `OMNILRS_WARMUP_RENDERED` y `OMNILRS_WARMUP_PHYSICS` pisan esas cuentas. En un equipo con 12 GiB o más el calentamiento sigue siendo 100 pasos renderizados.

`limit_cpu_threads` queda en 4 cuando la RAM instalada es menor de 12 GiB, y en 8 cuando el perfil es `low` o `balanced` en una máquina más grande. Eso acota los hilos de carb y de TBB. El solver, la gravedad, el CCD y el paso de PhysX no se tocan. Con una sola GPU, `max_gpu_count` queda en 1. En el perfil `high` de un equipo grande, con dos GPU, no se escribe ninguno de esos topes.

La ventana del sistema, cuando la sesión no es headless y el yaml ya fijó ancho y alto, usa ese mismo tamaño. El valor por defecto de `SimulationApp` es una ventana de 1440×900 junto a un viewport de 1280×720. Igualarlas evita ese segundo tamaño de swapchain. Es un ahorro de VRAM **estimado**, no medido aquí.

Al cerrar, ROS 2, Zenoh y Yamcs ya no importan `omni.replicator.core`. SDG sí, para terminar la escritura de los anotadores.

`OMNILRS_DISABLE_UI_EXTENSIONS=1` añade `--disable` para barras y menús (`omni.kit.window.toolbar`, menús del viewport, `isaacsim.gui.menu` y otras de la misma lista). Está apagado por defecto: no se comprobó, contra un Kit 107.3 en marcha, que el kit base pueda arrancar sin ellas. La lista no incluye `isaacsim.sensors.camera`, `isaacsim.sensors.rtx`, `omni.physx`, `omni.hydra.rtx` ni `omni.usd`. `OMNILRS_DISABLE_VIEWPORT=1` aplica `disable_viewport_updates` en una sesión que ya es headless, también en equipos grandes.

No se desactivan las actualizaciones de pose de Fabric. OmniLRS lee transforms y articulaciones. No se activa `skipMaterialLoading`. No se cambia el pipeline de física a GPU. El modo ROS 2 sigue habilitando `isaacsim.ros2.bridge` y `omni.kit.viewport.actions` después del arranque.

### Qué ya no depende del runtime de Kit

El motor estelar (`src/stellar/stellar_engine.py`) y los modelos de radio, potencia, térmico, OBC y compresión de cámara no importan `omni`, `isaacsim` ni `pxr`. El test de radio comprueba la caída cuadrática. El test térmico avanza un constante de tiempo con el Sol en +Z y compara cada cara con la sigmoide. El motor estelar se ejecutó aquí con `de421.bsp`, la fecha por defecto (2024-05-11 16:30 UTC) y un observador a −89.5° de latitud: altitud solar **−1.335°**, azimut **140.127°**, distancia **1.508×10¹¹ m** (skyfield 1.55). Esas cuentas no pasan por Kit.

El mixin que escribe la luz solar sí usa `pxr`. PhysX, las cámaras RTX, el stage USD y las mallas de clipmap siguen dentro del proceso de Kit. Separarlos en otro proceso duplicaría el stage y la escena física. No hay un runtime alternativo dentro de este repositorio que conserve esas funciones.

### Medido, estimado y requisito de hardware

En esta máquina, el 22 de octubre de 2026, `import isaacsim` termina en `ModuleNotFoundError`. Una lectura de RAM del mismo día da **31.129 GiB instalados** y **0.769 GiB disponibles**. No hay una medición de RSS de Kit, de VRAM ni de FPS. El objetivo de ejecutar el simulador completo con 6 GiB de RAM **sigue sin estar validado**.

| Dato | Tipo |
| --- | --- |
| Working set y commit del proceso de los tests de terreno, fase 1 | Medido, sin Kit |
| RSS de `SimulationApp`, VRAM, FPS, estabilidad en 6 u 8 GiB | No medido |
| Ventana del DEM, bloques de cráteres, tamaño de malla | Calculado a partir de los arrays |
| Viewport headless, hilos de Kit, calentamiento, Replicator al cerrar, kit base frente a kit full | Ahorro estimado. La política está cubierta por tests de decisión, no por una sesión de Kit |
| Modelos de radio, térmico y motor estelar con de421 | Ejecutados aquí, fuera de Kit |
| RAM instalada de esta máquina | Medida: 31.129 GiB. No es un equipo de 6–8 GiB |
| Mínimo de Isaac Sim 5.0 publicado por NVIDIA | 32 GiB de RAM y 16 GiB de VRAM. Es el requisito del runtime, no una cifra medida por OmniLRS |

Lo que este repositorio no puede bajar: el proceso de Kit, RTX, PhysX, la caché de shaders y el kit base, que de todos modos carga sensores y el renderer. Un DEM de cientos de kilómetros a resolución fina no cabe residente en 6 GiB; la ventana de alta resolución ya se acota en el perfil bajo, y el archivo del polo sur no está en este checkout.

## 8 GiB de VRAM con 64 GiB de RAM

Una RTX 2000 Blackwell de portátil con 8 GiB de VRAM y 64 GiB de RAM cae en el perfil de RAM **alto**. La ventana del DEM, los workers de cráteres, los hilos de PhysX y los rebotes del yaml se quedan. El presupuesto nuevo mira la VRAM por separado.

Con `OMNILRS_MEMORY_PROFILE=auto` y la GPU más pequeña en 9 GiB o menos:

- `samples_per_pixel_per_frame` de `RayTracedLighting` pasa a 1. Es el modo de tiempo real de RTX. En trazado de caminos el tope es 8. Los rebotes no se tocan.
- El viewport queda en 1280×720. Las resoluciones de las cámaras del robot no se tocan: Husky sigue en 320×240 y 1280×720, Pragyaan en 640×480 y 1440×1080.
- El clipmap grueso queda en modo híbrido: el DEM de baja resolución vive en los 64 GiB y no se copia a VRAM.
- Se activa el streaming de texturas. El material es el mismo; Kit mantiene residente el mip que cabe en la vista. El eco mode no se activa en este perfil, porque cambia las muestras cuando la cámara está quieta.
- Color y profundidad de cada resolución salen del mismo render product. Antes había un segundo `Camera` del mismo tamaño solo para la profundidad.

`OMNILRS_MEMORY_PROFILE=high` deja las 32 muestras. El clipmap híbrido sigue aplicándose por debajo de 12 GiB de VRAM.

Esto no se ha medido en una Blackwell. La GPU de la máquina donde está abierto el repo es una GeForce RTX 2060 SUPER, también de 8192 MiB, y Kit no se ha lanzado. El ahorro de muestras, de la ventana y del segundo framebuffer de profundidad es **estimado**. El heap de RTX que reserva Isaac Sim 5.0 sigue teniendo un mínimo publicado de 16 GiB de VRAM. Caber el viewport en 8 GiB no demuestra que el proceso de Kit arranque y se quede estable en esa Blackwell.
