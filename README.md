## Pipeline de Credit Card Clients

### Descripción
Este proyecto implementa un pipeline de procesamiento de datos y entrenamiento de un modelo de clasificación para predecir clientes de tarjetas de crédito.
Utiliza **Airflow** para orquestar los pipelines y **Docker** para contenerizar los servicios.
El modelo de machine learning es un **RandomForestClassifier** entrenado y ajustado utilizando [datos de clientes de tarjetas de crédito](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

![image](https://github.com/user-attachments/assets/2aa1caa9-32e2-44e7-835b-d4ccfc0a1bc4)
![image](https://github.com/user-attachments/assets/2aa1caa9-32e2-44e7-835b-d4ccfc0a1bc4)


### Instalación y Configuración

1. **Correr el entorno con Docker**:
   ```bash
   docker-compose up --build
   ```

4. **Acceder a Airflow**:
   - Visitar `http://localhost:8080` en tu navegador.
   - Usuario por defecto: `admin`
   - Contraseña por defecto: `admin`

###e Tests

1. **Correr los tests unitarios**:
   ```bash
   pytest
   ```

### Ejecución del Pipeline

1. **Pipeline de creación del modelo**:
   - Dentro de Airflow, habilitar y ejecutar el DAG `pipeline_creacion_modelo` que incluye:
     - Cargar datos
     - Limpieza de datos
     - Entrenar modelo
     - Ajustar hiperparámetros

2. **Pipeline de monitoreo del modelo**:
   - Habilitar y ejecutar el DAG `pipeline_monitoreo_modelo` para:
     - Monitorear el rendimiento del modelo
     - Reentrenar el modelo si es necesario

### Acceso a los Informes
Los gráficos y reportes de interpretabilidad del modelo, se guardan en la carpeta `reports/` dentro del contenedor Docker, y se pueden acceder a través de las rutas definidas en el pipeline.
