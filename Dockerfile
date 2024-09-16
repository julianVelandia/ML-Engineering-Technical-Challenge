FROM apache/airflow:2.5.1

USER airflow

RUN pip install --no-cache-dir \
    ucimlrepo \
    scikit-learn \
    pandas \
    numpy \
    joblib \
    scipy \
    matplotlib \
    seaborn \
    shap
