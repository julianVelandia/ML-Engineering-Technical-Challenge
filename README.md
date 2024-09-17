## Pipeline de Credit Card Clients

### Descripción
Este proyecto implementa un pipeline de procesamiento de datos y entrenamiento de un modelo de clasificación para predecir clientes de tarjetas de crédito.
Utiliza **Airflow** para orquestar los pipelines y **Docker** para contenerizar los servicios.
El modelo de machine learning es un **RandomForestClassifier** entrenado y ajustado utilizando [datos de clientes de tarjetas de crédito](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

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

Se deben disparar los Pipelines manualmente, ya que el paràmetro schedule està en None

![image](https://github.com/user-attachments/assets/6da007ce-0305-4586-9497-514cffb5c9c7)



1. **Pipeline de creación del modelo**:
   - Dentro de Airflow, habilitar y ejecutar el DAG `pipeline_creacion_modelo` que incluye:
     - Cargar datos
     - Limpieza de datos
     - Entrenar modelo
     - Ajustar hiperparámetros
     - Generar Reporte

![image](https://github.com/user-attachments/assets/4bc1955e-ceab-4d32-aaba-3434d7f1754b)

2. **Pipeline de monitoreo del modelo**:
   - Habilitar y ejecutar el DAG `pipeline_monitoreo_modelo` para:
     - Monitorear el rendimiento del modelo
     - Reentrenar el modelo si es necesario
    
Las mètricas que el modelo selecciona como las mejores son estas: 
![image](https://github.com/user-attachments/assets/73eb0a03-5bbc-4a72-8397-9430c564430c)


3. **Pipeline de EDA (Análisis Exploratorio de Datos)**:
   - Habilitar y ejecutar el DAG `eda_credit_card_clients` para:
     - Cargar los datos
     - Generar un análisis exploratorio de datos (EDA), que incluye estadísticas descriptivas y visualizaciones clave
     - Guardar el análisis y los gráficos generados en un reporte

El reporte del EDA se encuentra en `reports/`
    
![image](https://github.com/user-attachments/assets/13c483f9-6110-409d-992a-24af69717fcb)


4. **Pipeline de predicción con nuevos datos**:
   - Habilitar y ejecutar el DAG `pipeline_predecir_nuevos_datos` para:
     - Cargar nuevos datos de clientes
     - Aplicar el modelo entrenado a los nuevos datos
     - Generar predicciones sobre la probabilidad de incumplimiento de pago de los nuevos clientes
     - Guardar las predicciones generadas en un archivo CSV

### Acceso a los Informes

Los gráficos y reportes de interpretabilidad del modelo, se guardan en la carpeta `reports/` dentro del contenedor Docker, y se pueden acceder a través de las rutas definidas en el pipeline.

![image](https://github.com/user-attachments/assets/19c9ec89-5355-49c0-b095-0b31017553cd)

![image](https://github.com/user-attachments/assets/d699067b-66e9-4ac3-bc35-7a1fd57d7d2c)

![image](https://github.com/user-attachments/assets/c14dd6d0-d5fa-4bec-a4b5-3874568de445)
