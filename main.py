import psycopg2
from PredictCharacters import recognize_license_plate

# Datos de conexión |
DB_HOST = '192.168.1.77'
DB_PORT = '5432'
DB_NAME = 'vision_db'
DB_USER = 'alextcw'
DB_PASSWORD = 'root'

# Ruta al modelo
MODEL_PATH = './finalized_model.sav'

def obtener_conexion():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def buscar_placa_en_db(texto_placa):
    conn = None
    try:
        conn = obtener_conexion()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM usuarios.usuarios u WHERE u.placa = %s;", (texto_placa,))
        count = cursor.fetchone()[0]

        if count == 0:
            raise ValueError(f"No se encontró la placa: {texto_placa}")
        elif count == 1:
            cursor.execute("SELECT * FROM usuarios.usuarios u WHERE u.placa = %s;", (texto_placa,))
            resultado = cursor.fetchone()
            print("Placa encontrada:", resultado)
        else:
            print(f"Se encontraron múltiples registros con la placa {texto_placa}")

    except Exception as e:
        print(f"Error al buscar la placa en la base de datos: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    texto_placa = recognize_license_plate(MODEL_PATH)
    print(f"Texto reconocido de la placa: {texto_placa}")
    #buscar_placa_en_db(texto_placa)
