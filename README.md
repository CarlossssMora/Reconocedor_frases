# Proyecto: Reconocedor de Frases

## 1. Uso de entorno virtual
Para ejecutar este proyecto es necesario la creación de un entorno virtual. Primeramente, ejecuta el siguiente comando en la terminal (Windows):
```bash
python -m venv venv
```
Ahora activa el entorno virtual con el siguiente comando:
```bash
venv\Scripts\activate
```

## 2. Descarga de librerías
Para la descarga de todas las librerías necesarias en el entorno virtual, ejecuta el siguiente comando
```bash
pip install -r requirements.txt
```

## 3. Prueba en vivo del reconocedor
Para la comodidad del usuario, se puede probar el reconocedor por medio de una interfaz proporcionada por Streamlit. Para probar el reconocedor, ejecute el siguiente comando:
```bash
streamlit run live_recognition.py
```