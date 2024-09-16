import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from config.constants import DATA_DIR, FEATURES_FILENAME, REPORTS_DIR
from config.constants import EDA_FILE


def generar_eda():
    """
    Genera un análisis exploratorio de datos (EDA) a partir de los datos cargados en el sistema.
    El EDA incluye estadísticas descriptivas y visualizaciones clave, las cuales se guardan en
    formato markdown (.md) y como imágenes (.png) dentro de un directorio específico de reportes.

    Proceso:
        - Carga los datos desde los archivos CSV de características.
        - Genera estadísticas descriptivas y visualizaciones de las variables.
        - Guarda el análisis en un archivo .md y las visualizaciones en formato .png.
    """
    features_path = os.path.join(DATA_DIR, FEATURES_FILENAME)
    df = pd.read_csv(features_path)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(EDA_FILE, 'w', encoding='utf-8') as f:
        f.write("# Análisis Exploratorio de Datos (EDA)\n")
        f.write("## Dataset: Default of Credit Card Clients\n")
        f.write(f"**Número de registros:** {df.shape[0]}\n")
        f.write(f"**Número de columnas:** {df.shape[1]}\n\n")

        f.write("## Primeras Filas del Dataset\n")
        f.write(f"{df.head().to_markdown()}\n\n")

        f.write("## Estadísticas Descriptivas\n")
        f.write(f"{df.describe().to_markdown()}\n\n")

        f.write("## Información de Tipos de Datos\n")
        f.write(f"{df.dtypes.to_markdown()}\n\n")

        f.write("## Valores Nulos\n")
        f.write(f"{df.isnull().sum().to_markdown()}\n\n")

        f.write("## Visualizaciones\n")

        if 'default_payment_next_month' in df.columns:
            plt.figure(figsize=(8, 6))
            sns.countplot(data=df, x='default_payment_next_month', palette='viridis')
            plt.title('Distribución de Default Payment Next Month')
            plot_path = os.path.join(REPORTS_DIR, 'default_payment_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            f.write(f"![Distribución de Default Payment Next Month]({plot_path})\n\n")

        corr = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plot_path = os.path.join(REPORTS_DIR, 'correlation_matrix.png')
        plt.savefig(plot_path)
        plt.close()
        f.write(f"![Matriz de Correlación]({plot_path})\n\n")

        f.write("### Histogramas de Variables Principales\n")
        num_vars = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']
        for var in num_vars:
            if var in df.columns:
                plt.figure(figsize=(8, 6))
                sns.histplot(df[var], kde=True, color='blue')
                plt.title(f'Distribución de {var}')
                plot_path = os.path.join(REPORTS_DIR, f'{var}_distribution.png')
                plt.savefig(plot_path)
                plt.close()
                f.write(f"![Distribución de {var}]({plot_path})\n\n")

