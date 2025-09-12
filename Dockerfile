FROM continuumio/miniconda3

WORKDIR /app

COPY app/environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

# SHELL ["conda", "run", "-n", "env_gnn", "/bin/bash", "-c"]

COPY ./app .

EXPOSE 8000

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["conda", "run", "-n", "env_gnn", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]