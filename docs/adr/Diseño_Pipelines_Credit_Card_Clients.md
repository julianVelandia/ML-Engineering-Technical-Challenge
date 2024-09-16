## ADR: Diseño de Pipelines para Credit Card Clients

### Contexto

El proyecto de Credit Card Clients necesita una solución automatizada para entrenar, monitorear y actualizar modelos de
machine learning que predicen comportamientos de clientes de tarjetas de crédito. Se plantea el diseño para que los
datos se actualizan
periódicamente, y el modelo debe ajustarse en base a nuevas entradas. Se ha optado por utilizar **Airflow** para la
orquestación y **RandomForest** como algoritmo principal de clasificación.

### Decisiones

1. **Pipelines**
    - **Decisión**: Los pipelines se separan en "Creación del Modelo", "Monitoreo del Modelo" y una task de "Predicción de Nuevos Datos".
    - **Justificación**:
        - La separación permite un desarrollo modular donde cada pipeline cumple un rol específico sin depender de la
          ejecución simultánea de otros procesos.
        - Los pipelines independientes facilitan la gestión de errores y la escalabilidad de cada proceso según las
          necesidades de negocio.
        - Al separar el pipeline de monitoreo, se garantiza que la monitorización y reentrenamiento del modelo ocurran
          sin interferir en las predicciones diarias de nuevos datos.

2. **RandomForest como Algoritmo**
    - **Decisión**: Se utiliza **RandomForestClassifier** para entrenar el modelo de predicción.
    - **Justificación**:
        - RandomForest es robusto ante el sobreajuste y maneja bien datasets con múltiples variables.
        - Los datos disponibles sobre clientes de tarjetas de crédito contienen variables tanto continuas como
          categóricas, lo que se ajusta bien a RandomForest.
        - Algoritmos alternativos como XGBoost o SVM fueron evaluados, pero RandomForest mostró resultados óptimos en
          términos de precisión y tiempo de entrenamiento en las pruebas iniciales.

3. **Uso de Airflow para Orquestación**
    - **Decisión**: Se implementan 2 pipelines separados en Airflow para ETL, entrenamiento de modelos y monitoreo de
      rendimiento.
    - **Justificación**:
        - Airflow permite la automatización y monitoreo de tareas de larga duración, facilitando la integración de
          pipelines que dependen unos de otros.
        - La capacidad de Airflow para manejar flujos de trabajo complejos y programados lo hace ideal para el
          seguimiento y la actualización continua de modelos.
        - Alternativas como Kubeflow o Luigi fueron consideradas, pero Airflow ofrece mayor flexibilidad y una mejor
          integración con diversas fuentes de datos y tareas periódicas.

### Consecuencias

- **Mantenimiento**: La separación de pipelines facilita la actualización independiente de cada parte del flujo de
  trabajo, minimizando la complejidad de mantenimiento.
- **Precisión**: RandomForest ofrece un buen balance entre rendimiento y precisión sin requerir un ajuste continuo
  complejo.
- **Escalabilidad**: Airflow permite escalar los pipelines a medida que crecen los datos o se requiere aumentar la
  frecuencia de predicciones o monitoreo.
