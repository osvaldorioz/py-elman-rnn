from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import elman_rnn
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Generación de datos de ejemplo
def generate_data(n_samples: int):
    X = np.linspace(0, 2 * np.pi, n_samples)
    y = np.sin(X)
    return X.reshape(-1, 1), y.reshape(-1, 1)

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/elman-rnn")
def calculo(num_samples: int, input_size: int, 
            hidden_size: int, output_size: int, 
            epochs: int, lr: float):
    output_file = 'elman_rnn.png'

    X_train, y_train = generate_data(num_samples)

    # Entrenamiento del modelo
    # input_size=1
    # hidden_size=10
    # output_size=1
    # epochs=500
    # lr=0.01
    output, loss_values = elman_rnn.train_rnn(X_train, y_train, input_size, hidden_size, output_size, epochs, lr)

    # Graficar los resultados
    plt.figure(figsize=(12, 5))

    # Gráfica de dispersión
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, label='Datos reales', color='blue', alpha=0.5)
    plt.scatter(X_train, output, label='Predicciones', color='red', alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Dispersión de datos y predicciones")

    # Gráfica de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(loss_values, label='Pérdida')
    plt.xlabel("Época")
    plt.ylabel("Error MSE")
    plt.legend()
    plt.title("Evolución de la pérdida durante el entrenamiento")

    plt.tight_layout()
    #plt.show()

    plt.savefig(output_file)
    plt.close()
    
    j1 = {
        "Grafica": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/elman-rnn-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)