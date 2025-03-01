El algoritmo **Elman RNN** es una red neuronal recurrente (RNN) simple propuesta por Jeffrey Elman en 1990. Se caracteriza por tener una **capa oculta con retroalimentación**, lo que le permite **recordar información de estados anteriores** y capturar dependencias temporales en secuencias de datos.

### **Estructura del Elman RNN**
1. **Capa de entrada**: Recibe los datos de entrada \( x_t \).
2. **Capa oculta**: Tiene conexiones recurrentes, lo que significa que el estado oculto \( h_t \) en el tiempo \( t \) se calcula usando tanto la entrada actual como el estado oculto anterior \( h_{t-1} \).
   ![imagen](https://github.com/user-attachments/assets/36e4ff84-d1cd-4abf-97ac-c934cbbd11e3)

3. **Capa de salida**: Convierte el estado oculto en una salida \( y_t \).
   ![imagen](https://github.com/user-attachments/assets/e4fc9f97-0879-4c6c-9b98-8cc230ae262a)


### **Características clave**
- Usa una **memoria interna** en la capa oculta para modelar datos secuenciales.
- Se entrena típicamente usando **descenso de gradiente y retropropagación a través del tiempo (BPTT)**.
- Es más adecuado para secuencias cortas debido a problemas como el **desvanecimiento del gradiente**.

### **Implementación en tu programa**
- Se implementa usando `Eigen` para manejar las operaciones matriciales.
- El entrenamiento ajusta los pesos \( W_x \), \( W_h \), y \( W_y \) con un método de optimización simple.
- Se expone la función `train_rnn` a Python mediante `pybind11`, permitiendo entrenar el modelo y visualizar los resultados en gráficos.
