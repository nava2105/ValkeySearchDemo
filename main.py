import os
import valkey
import numpy as np
from wordfreq import top_n_list
from sentence_transformers import SentenceTransformer

# Configuración para Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

# Conexión (IMPORTANTE: decode_responses=False)
client = valkey.Valkey(host='localhost', port=6379, decode_responses=False)

print("Cargando modelo...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✓ Modelo cargado")


def clean_database():
    """LIMPIA TODO antes de empezar"""
    print("\n=== LIMPIANDO BASE DE DATOS ===")

    # Eliminar claves antiguas
    keys_to_delete = []
    for key in client.scan_iter("word:vector:*", count=1000):
        keys_to_delete.append(key)

    if keys_to_delete:
        client.delete(*keys_to_delete)
        print(f"✓ Eliminadas {len(keys_to_delete)} claves antiguas")

    # Eliminar índice si existe
    try:
        client.execute_command("FT.DROPINDEX", "words_idx")
        print("✓ Índice eliminado")
    except:
        print("No había índice previo")

    print("✓ Base de datos limpia\n")


def get_words_by_language():
    languages = {
        'es': 'spanish', 'en': 'english', 'fr': 'french',
        'ru': 'russian', 'it': 'italian', 'de': 'german',
        'pt': 'portuguese'
    }
    # languages = {
    #     'ca': 'Catalán', 'cs': 'Checo', 'da': 'Danés',
    #     'de': 'Alemán', 'el': 'Griego', 'en': 'Inglés',
    #     'es': 'Español', 'et': 'Estonio', 'eu': 'Vasco',
    #     'fi': 'Finlandés', 'fr': 'Francés', 'gl': 'Gallego',
    #     'hr': 'Croata', 'hu': 'Húngaro', 'is': 'Islandés',
    #     'it': 'Italiano', 'lt': 'Lituano', 'lv': 'Letón',
    #     'nl': 'Holandés', 'no': 'Noruego', 'pl': 'Polaco',
    #     'pt': 'Portugués', 'ro': 'Rumano', 'ru': 'Ruso',
    #     'sk': 'Eslovaco', 'sl': 'Esloveno', 'sr': 'Serbio',
    #     'sv': 'Sueco', 'tr': 'Turco', 'uk': 'Ucraniano',
    #     'ar': 'Árabe', 'fa': 'Persa/Farsi', 'he': 'Hebreo',
    #     'hi': 'Hindi', 'id': 'Indonesio', 'ja': 'Japonés',
    #     'ko': 'Coreano', 'ms': 'Malayo', 'th': 'Tailandés',
    #     'ur': 'Urdu', 'vi': 'Vietnamita', 'zh': 'Chino',
    #     'am': 'Amhárico', 'bg': 'Búlgaro', 'bn': 'Bengalí',
    #     'gu': 'Gujaratí', 'kn': 'Kannada', 'ml': 'Malayalam',
    #     'mr': 'Marathi', 'ne': 'Nepalí', 'pa': 'Punjabí',
    #     'ta': 'Tamil', 'te': 'Telugu'
    # }

    all_words = []
    for lang_code, lang_name in languages.items():
        print(f"Cargando palabras en {lang_name}...")
        words = top_n_list(lang_code, 15000)
        all_words.extend([(word, lang_code) for word in words])
    return all_words


def generate_and_store_embeddings():
    words = get_words_by_language()
    print(f"Total de palabras a procesar: {len(words)}")

    batch_size = 1000
    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        texts = [word for word, _ in batch]

        embeddings = model.encode(texts, show_progress_bar=False)

        pipeline = client.pipeline()
        for j, (word, lang_code) in enumerate(batch):
            word_hash = hash(word) & 0xFFFFFFFF
            key = f"word:vector:{lang_code}:{word_hash}"

            # **Almacenar como HASH con BYTES**
            vector_bytes = embeddings[j].astype(np.float32).tobytes()
            pipeline.hset(key, mapping={
                "word": word.encode('utf-8'),
                "language": lang_code.encode('utf-8'),
                "vector": vector_bytes
            })

        pipeline.execute()
        print(f"Lote {i // batch_size + 1} procesado ({len(batch)} palabras)")

    print("✓ Embeddings almacenados")


def create_vector_index():
    """Crear índice con sintaxis MÍNIMA garantizada"""
    print("Creando índice...")

    result = client.execute_command(
        "FT.CREATE", "words_idx",
        "ON", "HASH",
        "PREFIX", "1", "word:vector:",
        "SCHEMA",
        "vector", "VECTOR", "HNSW", "6",
        "TYPE", "FLOAT32",
        "DIM", "384",
        "DISTANCE_METRIC", "COSINE"
    )
    print(f"✓ Índice creado: {result}")
    return True


def verify_storage():
    print("\n=== VERIFICANDO ALMACENAMIENTO ===")

    count = sum(1 for key in client.scan_iter("word:vector:*", count=1000)
                if client.type(key) == b'hash')
    print(f"Claves HASH válidas: {count}")

    if count > 0:
        for key in client.scan_iter("word:vector:*", count=100):
            if client.type(key) == b'hash':
                data = client.hgetall(key)
                print(f"\nEjemplo: {key.decode()}")
                print(f"Palabra: {data.get(b'word', b'').decode()}")
                print(f"Idioma: {data.get(b'language', b'').decode()}")
                print(f"Vector: {len(data.get(b'vector', b''))} bytes")
                break


def test_search():
    print("\n=== PRUEBA BÚSQUEDA ===")

    query_word = "gatto"
    query_vector = model.encode([query_word])[0].astype(np.float32).tobytes()

    results = client.execute_command(
        "FT.SEARCH", "words_idx",
        "*=>[KNN 5 @vector $vec]",
        "PARAMS", "2", "vec", query_vector,
        "DIALECT", "2",
        "RETURN", "2", "word", "language",
        "LIMIT", "0", "5"
    )

    if len(results) > 1:
        print(f"✓ {len(results) // 2} resultados:")
        for i in range(1, len(results), 2):
            fields = results[i + 1]
            word = fields[fields.index(b'word') + 1].decode()
            lang = fields[fields.index(b'language') + 1].decode()
            print(f"  {word} ({lang})")
    else:
        print("No se encontraron resultados")


if __name__ == "__main__":
    # **SIEMPRE LIMPIAR PRIMERO**
    clean_database()

    print("1. Generando embeddings...")
    generate_and_store_embeddings()

    print("\n2. Creando índice...")
    create_vector_index()

    print("\n3. Verificando...")
    verify_storage()

    print("\n4. Probando búsqueda...")
    test_search()