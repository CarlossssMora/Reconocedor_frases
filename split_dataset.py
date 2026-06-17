import os
import shutil

# Carpetas
origen = "dataset_general_ruido"
train_dir = "dataset_ruido_train"
test_dir = "dataset_ruido_test"

# Limpiar las carpetas si ya existían para evitar duplicados
if os.path.exists(train_dir): shutil.rmtree(train_dir)
if os.path.exists(test_dir): shutil.rmtree(test_dir)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for clase in os.listdir(origen):
    ruta_clase = os.path.join(origen, clase)
    if not os.path.isdir(ruta_clase):
        continue

    # Crear subcarpetas en train y test
    os.makedirs(os.path.join(train_dir, clase), exist_ok=True)
    os.makedirs(os.path.join(test_dir, clase), exist_ok=True)

    archivos = sorted(os.listdir(ruta_clase))
    
    for archivo in archivos:
        if not archivo.endswith(".wav"): continue
        
        ruta_origen = os.path.join(ruta_clase, archivo)
        
        # NUEVA LÓGICA: Buscar el 09 o 10 en cualquier parte del nombre
        if "_09" in archivo or "_10" in archivo:
            shutil.copy2(ruta_origen, os.path.join(test_dir, clase, archivo))
        else:
            shutil.copy2(ruta_origen, os.path.join(train_dir, clase, archivo))

print("Dataset mixto dividido exitosamente en 'dataset_ruido_train' y 'dataset_ruido_test'")