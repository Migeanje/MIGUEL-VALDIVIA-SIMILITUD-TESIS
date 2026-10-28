# Guion de la sesión con expertos

> **Estado: borrador.** Los valores que el plan no fija (duración, tamaño de grupo, orden de las
> vistas) son valores por defecto de este borrador y se listan al final. Los textos entre
> comillas angulares («…») son lo que dice el facilitador; adáptelos sin cambiar su sentido.

Guía del facilitador para la sesión en la que el panel de expertos conoce el prototipo y lo
evalúa. Cubre los dos primeros momentos del procedimiento del plan de tesis (§3.8.3): la
presentación del prototipo y de una muestra de sus resultados, y el cuestionario individual,
previa firma del consentimiento informado. El tercer momento, el análisis, está en
[protocolo_datos.md](protocolo_datos.md).

## Resumen

| Aspecto | Valor |
|---|---|
| Panel | De seis a ocho docentes de la UCSM de perfil mixto: dos a tres especialistas en inteligencia artificial, aprendizaje automático o ciencia de datos, y tres a cinco asesores de tesis de distintas carreras de la Facultad (§3.8.1) |
| Formato | Sesiones de 1 a 4 expertos con este mismo guion; dos o tres sesiones cubren el panel |
| Duración | 90 minutos |
| Facilitador | El investigador; opcionalmente, un asistente que toma notas sin nombres |
| Equipo | La laptop local con la imagen Docker del prototipo (O07, T29) y un proyector o pantalla |
| Lugar | [completar] |
| Qué se evalúa | Las tres vistas del prototipo (mapa temático, consulta de redundancia y panel de vacíos) y una muestra de sus resultados ([muestras_anonimizadas.md](muestras_anonimizadas.md)) |

## Antes de la sesión

- [ ] Los jueces validaron el cuestionario (V de Aiken, T30) y la versión final está impresa,
      sin la frase del borrador sobre su validación.
- [ ] Las muestras están generadas y pasaron las verificaciones de
      [muestras_anonimizadas.md](muestras_anonimizadas.md).
- [ ] El prototipo corre en la laptop **sin conexión a internet**; se probaron las tres vistas y
      la consulta de cada caso de la muestra B.
- [ ] Ningún experto del panel es anotador del conjunto etiquetado con trabajo pendiente: la
      sesión muestra similitudes entre tesis, y los anotadores etiquetan sin ver puntajes del
      sistema (O02). Si un docente cumple ambos roles, participa en el panel solo después de
      cerrar el etiquetado, incluida la adjudicación (T21).
- [ ] Se sabe si el umbral de redundancia τ ya está calibrado (T22). Si no lo está, se usa el
      texto "τ pendiente" del bloque 6.
- [ ] Están listos los materiales de la lista final.

## Cronograma

| # | Bloque | Minutos | Inicio | Material |
|---|---|:-:|:-:|---|
| 1 | Bienvenida | 5 | 0:00 | — |
| 2 | Consentimiento informado y códigos | 10 | 0:05 | Formularios, tarjetas de código, sobre A |
| 3 | Qué es el prototipo | 5 | 0:15 | Proyector |
| 4 | Vista del mapa temático | 12 | 0:20 | Muestra C |
| 5 | Vista del panel de vacíos | 12 | 0:32 | Muestra D |
| 6 | Vista de consulta | 12 | 0:44 | Muestras A y B |
| 7 | Uso guiado del prototipo | 10 | 0:56 | Laptop |
| 8 | Cuestionario individual | 20 | 1:06 | Cuestionarios, sobre B |
| 9 | Cierre | 4 | 1:26 | — |
| | **Total** | **90** | | |

Las vistas se presentan en el orden de las dimensiones del cuestionario: el mapa para la
dimensión 1, los vacíos para la 2 y la consulta, con sus recomendaciones, para la 3. El
cuestionario se responde una sola vez, al final, como establece el segundo momento de §3.8.3.

## Guion por bloque

### 1. Bienvenida (5 min)

> «Buenos días. Gracias por su tiempo. Esta sesión forma parte de la validación de un prototipo
> desarrollado en una tesis de Ingeniería de Sistemas. Primero les mostraré el prototipo y una
> muestra de sus resultados; luego, cada uno responderá un cuestionario breve de forma
> individual. No hay respuestas correctas ni incorrectas: los aciertos y los errores que ustedes
> observen son igual de útiles.»

### 2. Consentimiento informado y códigos (10 min)

1. Entregue dos copias del [consentimiento informado](consentimiento_informado.md) a cada
   participante y lea en voz alta las secciones 3 a 6.
2. Responda las preguntas. Cada participante marca su rol (experto del panel), las
   autorizaciones opcionales y firma ambas copias.
3. Cada participante saca al azar una tarjeta de código del mazo (`E01`, `E02`, …). Anote el
   código en ambas copias y en la lista de códigos en papel ([protocolo_datos.md](protocolo_datos.md)).
4. Guarde la copia del investigador y la lista de códigos en el **sobre A**, separado de los
   cuestionarios.

> «Antes de empezar, les pido leer este formulario. Su participación es voluntaria y pueden
> retirarse en cualquier momento, sin dar explicaciones. Sus respuestas se identifican solo con el
> código de la tarjeta que van a sacar; su nombre queda únicamente en este formulario, que guardo
> aparte de los cuestionarios.»

Si alguien no acepta, agradézcale; puede retirarse o quedarse como oyente, sin responder el
cuestionario. No se registra ningún dato suyo.

### 3. Qué es el prototipo (5 min)

> «El prototipo analiza las tesis de cinco carreras de la Facultad depositadas en el repositorio
> institucional desde 2021 hasta el 2 de octubre de 2026, la fecha de corte del estudio. Hace tres
> cosas: agrupa las tesis por afinidad temática, señala áreas poco exploradas y, ante una
> propuesta nueva, muestra los trabajos previos más parecidos y sugiere antecedentes y líneas
> alternativas. Los materiales no muestran nombres de autores, asesores ni jurados. Una similitud
> alta entre dos trabajos no significa plagio ni una repetición confirmada: es una señal para
> revisar.»

Reparta los materiales impresos (muestras A a D), numerados para recogerlos al final.

### 4. Vista del mapa temático (12 min)

Muestre el mapa, abra dos o tres grupos de la muestra C y, en cada uno, sus términos
representativos y algunos de sus documentos.

> «Cada punto es una tesis y cada color, un grupo temático formado automáticamente. Para cada
> grupo, el sistema muestra sus términos más representativos y sus documentos. La muestra
> impresa reúne grupos elegidos al azar, no los mejores. Tómense unos minutos para revisarla.»

[ajustar a la versión final de la vista (T28): cómo se muestran las tesis que no quedan en ningún
grupo]

### 5. Vista del panel de vacíos (12 min)

Muestre el panel de vacíos y recorra la muestra D, un caso de cada tipo.

> «Un vacío es un área que el sistema señala como poco explorada. Usa cuatro reglas: grupos
> pequeños pero coherentes; dos grupos vecinos con casi ninguna tesis que los conecte; un tema
> presente en una carrera y ausente en una carrera afín; y un tema cuya participación disminuyó
> en los últimos años. El sistema los propone como candidatos; la valoración de ustedes es la que
> indica si son oportunidades reales de investigación.»

### 6. Vista de consulta (12 min)

Copie en la vista de consulta el título, el resumen y los objetivos de un caso de la muestra B y
muestre el resultado: los trabajos más similares con su grado de similitud, el grupo temático
asignado, los antecedentes sugeridos y las líneas alternativas. Repita con un segundo caso.
Luego recorra la muestra A, las asignaciones de las tesis de 2025 y 2026.

> «Para esta demostración usamos tesis de 2025 y 2026 como si fueran propuestas nuevas, y las
> comparamos con las tesis de 2021 a 2024. El sistema muestra los trabajos previos más parecidos,
> ordenados por similitud, el grupo temático al que asigna la propuesta, antecedentes sugeridos y
> líneas alternativas. Lo que se escribe en la consulta no se guarda.»

**τ pendiente.** Mientras el umbral de redundancia no esté calibrado (T22), diga:

> «El umbral a partir del cual el sistema considerará redundante una propuesta todavía no está
> calibrado; se calibra con pares de tesis que clasifican docentes. Por eso hoy verán la
> similitud y el orden de los trabajos, pero no un veredicto de "redundante" o "no redundante".»

[completar si τ ya existe: cómo muestra la vista el veredicto y de dónde sale el umbral]

### 7. Uso guiado del prototipo (10 min)

> «Ahora cada uno puede usar el prototipo durante unos minutos. Pueden repetir un caso de la
> hoja de consultas, escribir una propuesta propia o explorar el mapa. El sistema no guarda lo
> que se escribe.»

Cada participante usa la laptop unos dos minutos. Explique los controles solo si se lo piden; no
sugiera qué buscar.

### 8. Cuestionario individual (20 min)

Entregue el cuestionario final. Los participantes conservan las muestras impresas mientras
responden.

> «Les entrego el cuestionario. Anoten arriba el código de su tarjeta, no su nombre. Respondan de
> forma individual y sin conversar entre ustedes hasta que todos lo hayan entregado. Pueden
> consultar los materiales impresos. Si prefieren no responder alguna afirmación, déjenla en
> blanco. Al terminar, coloquen el cuestionario en el sobre B.»

Durante el cuestionario, responda solo con las aclaraciones estándar (más abajo) y anote en la
bitácora cada pregunta recibida. No mire las respuestas al recibirlas.

### 9. Cierre (4 min)

Recoja y cuente los materiales impresos. Si hay tiempo, invite comentarios finales y anótelos
sin nombres.

> «Muchas gracias. Si después desean retirar sus respuestas, pueden escribirme indicando su
> código antes del [completar: fecha límite]. Les pido devolver los materiales impresos: contienen
> información del repositorio que no se distribuye.»

## Qué decir y qué no decir

El facilitador no induce respuestas: presenta lo que el sistema produce y deja la valoración a
cada experto.

| Situación | Decir | No decir |
|---|---|---|
| Presentar un resultado | «Esto es lo que produce el sistema con la configuración actual.» | «Los resultados son muy buenos.» «El sistema casi siempre acierta.» |
| Pedir una opinión | «¿Qué observa en este grupo?» | «¿No le parece que este grupo es coherente?» |
| Un experto señala un error | «Gracias, lo anoto.» Si pregunta cómo funciona el sistema, explíquelo sin defender el resultado. | «Eso se debe a…» para justificar o minimizar el error. |
| Una similitud alta | «Una similitud alta es una señal para revisar, no un veredicto.» | «Estas dos tesis son redundantes.» «Esto es plagio.» |
| Alguien reconoce una tesis | «Los materiales no muestran autores; le pido no comentar de quién es.» | Confirmar o negar la autoría; hablar de autores, asesores o jurados. |
| Un enlace al repositorio | «El enlace lleva a la página pública del repositorio; durante la sesión no lo abrimos.» | Abrir la página del repositorio o el PDF, que muestran a los autores. |
| Expectativas | — | Mencionar los criterios de éxito (por ejemplo, la mediana de 4), lo que la tesis "necesita" o las respuestas de otros participantes. |
| Durante el cuestionario | Silencio y disponibilidad para dudas de procedimiento. | Comentar el avance, apurar, explicar el sentido de un ítem. |

## Aclaraciones estándar

Se dan con estas mismas palabras a todos los participantes.

| Pregunta | Respuesta |
|---|---|
| ¿Qué significa un ítem? | «Respóndalo según como usted lo entienda.» |
| ¿Puedo dejar un ítem en blanco? | «Sí.» |
| ¿Respondo sobre el sistema o sobre la muestra? | «Sobre lo que observó hoy del prototipo y de sus resultados.» |
| ¿Qué es la similitud? | «Un valor que mide cuánto se parecen dos textos por su contenido: a mayor valor, mayor parecido.» |
| Ítem 11, mientras τ no esté calibrado | «Hoy el sistema mostró los trabajos más similares, sin un veredicto. Responda según lo que observó.» [ajustar según la decisión pendiente sobre el ítem 11 y la versión final del cuestionario] |

## Si algo falla

| Situación | Qué hacer |
|---|---|
| El prototipo no arranca o se cae | Continúe con los materiales impresos y las capturas de respaldo de las tres vistas, y anótelo en la bitácora. |
| Un participante se retira durante la sesión | Pregúntele si desea que se usen las respuestas que dio; si no, su cuestionario se descarta sin transcribirlo. Anótelo sin su nombre. |
| Un participante no puede terminar el cuestionario | Se transcriben los ítems respondidos; los demás quedan en blanco. |
| Se acaba el tiempo | Acorte el bloque 7, nunca el bloque 8. |

## Materiales

- [ ] Laptop con la imagen Docker del prototipo probada sin conexión, y su cargador.
- [ ] Proyector o pantalla, con cable y adaptador.
- [ ] Capturas de respaldo de las tres vistas (con las mismas reglas de anonimización que las muestras).
- [ ] Muestras A a D impresas: un juego numerado por participante y uno para el facilitador.
- [ ] Cuestionario final validado: uno por participante y dos de reserva.
- [ ] Consentimiento informado: dos copias por participante.
- [ ] Mazo de tarjetas de código, barajado, sin los códigos ya usados en sesiones anteriores.
- [ ] Lista de códigos en papel ([protocolo_datos.md](protocolo_datos.md)).
- [ ] Sobre A (consentimientos y lista de códigos) y sobre B (cuestionarios).
- [ ] Lapiceros, reloj o cronómetro, y la bitácora impresa.

## Bitácora de la sesión

Una por sesión. No lleva nombres.

| Campo | Registro |
|---|---|
| Sesión | S__ |
| Fecha y hora de inicio y fin | |
| Códigos presentes | |
| Versión del prototipo (commit) y del cuestionario | |
| ¿τ calibrado? | ☐ Sí ☐ No |
| Desviaciones del guion | |
| Preguntas recibidas y aclaraciones dadas | |
| Marcas ambiguas detectadas al transcribir (código e ítem) | |
| Problemas técnicos | |
| Comentarios finales (sin nombres) | |

## Valores por defecto de este borrador

El plan y el plan de tesis no fijan estos detalles; son propuestas por confirmar.

1. **Duración: 90 minutos**, con 20 para el cuestionario de 18 preguntas.
2. **Grupos de 1 a 4 expertos**, para que todos usen el prototipo en una sola laptop (O07).
3. **Orden de las vistas: mapa, vacíos y consulta**, el de las dimensiones 1 a 3 del cuestionario.
4. **El cuestionario se aplica una vez, al final**, y no por bloques, como el segundo momento de §3.8.3.
5. **Un bloque de uso guiado**, porque el ítem 15 pregunta si la interfaz es fácil de usar.
6. **Sin grabaciones de audio ni de video**: solo la bitácora escrita.
7. **Códigos por sorteo de tarjetas**, para que no sigan el orden de llegada ni el perfil.
8. **Materiales impresos que se recogen al final**, porque las tesis no se redistribuyen.
