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
