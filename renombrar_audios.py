from pathlib import Path

carpeta_base = Path("dataset")

for archivo in carpeta_base.rglob("*.wav"):
    if archivo.name.startswith("_"):
        nuevo_nombre = "checo" + archivo.name
        nueva_ruta = archivo.with_name(nuevo_nombre)

        if nueva_ruta.exists():
            print(f"Ya existe, no se renombró: {nueva_ruta}")
            continue

        archivo.rename(nueva_ruta)
        print(f"Renombrado: {archivo.name} -> {nuevo_nombre}")