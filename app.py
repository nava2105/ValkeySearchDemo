from flask import Flask, render_template, request, flash, session, redirect, url_for
import valkey
import numpy as np
from sentence_transformers import SentenceTransformer
import random

app = Flask(__name__)
app.secret_key = 'dev-key-change-in-production-semantic-dict-2025'

# === CONFIGURACIÓN GLOBAL ===
client = valkey.Valkey(host='localhost', port=6379, decode_responses=False)
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

LANGUAGES = {
    'es': 'Español', 'en': 'Inglés', 'fr': 'Francés',
    'ru': 'Ruso', 'it': 'Italiano', 'de': 'Alemán',
    'pt': 'Portugués'
}

print("Cargando modelo...")
model = SentenceTransformer(MODEL_NAME)
print("✓ Modelo cargado")


def search_similar_word(word: str, lang_filter: str = None, k: int = 5,
                        exclude_lang: str = None, max_distance: float = 1.0):
    """
    Búsqueda versátil de palabras similares con filtros.
    """
    try:
        query_vector = model.encode([word])[0].astype(np.float32).tobytes()

        # Construir consulta correctamente
        query_parts = []

        # Filtro positivo (incluir solo un idioma)
        if lang_filter:
            query_parts.append(f"@language:{lang_filter}")

        # Filtro negativo (excluir un idioma) - SINTAXIS CORRECTA
        if exclude_lang:
            # Forma 1: NOT @field:value
            query_parts.append(f"NOT @language:{exclude_lang}")
            # Forma 2 (alternativa si la anterior falla): -@field:{value}
            # query_parts.append(f"-@language:{{{exclude_lang}}}")

        query_parts.append(f"*=>[KNN {k} @vector $vec]")
        full_query = " ".join(query_parts)

        results = client.execute_command(
            "FT.SEARCH", "words_idx",
            full_query,
            "PARAMS", "2", "vec", query_vector,
            "DIALECT", "2",
            "RETURN", "3", "word", "language", "vector",
            "LIMIT", "0", str(k)
        )

        if len(results) <= 1:
            return []

        matches = []
        for i in range(1, len(results), 2):
            score = float(results[i]) if isinstance(results[i], (int, float)) else 0.0

            if score > max_distance:
                continue

            fields = results[i + 1]
            word_val = b''
            lang_val = b''

            for idx, field in enumerate(fields):
                if field == b'word' and idx + 1 < len(fields):
                    word_val = fields[idx + 1]
                elif field == b'language' and idx + 1 < len(fields):
                    lang_val = fields[idx + 1]

            if word_val and lang_val:
                matches.append({
                    'word': word_val.decode('utf-8'),
                    'language': lang_val.decode('utf-8'),
                    'score': score
                })

        return matches

    except Exception as e:
        print(f"Error en búsqueda: {e}")
        # NO lanzar, devolver vacío para que la app no caiga
        return []


def get_random_word():
    """Obtiene una palabra aleatoria de la base de datos."""
    try:
        # Intentar varias veces para evitar claves corruptas
        for attempt in range(5):
            keys = []
            for key in client.scan_iter("word:vector:*", count=200):
                keys.append(key)
                if len(keys) >= 200:
                    break

            if not keys:
                return None

            random_key = random.choice(keys)
            data = client.hgetall(random_key)

            word = data.get(b'word', b'').decode('utf-8')
            lang = data.get(b'language', b'').decode('utf-8')

            if word and lang:  # Validar que no estén vacíos
                return {'word': word, 'language': lang}

        return None

    except Exception as e:
        print(f"Error obteniendo palabra aleatoria: {e}")
        return None


def get_game_options(target_word, target_lang):
    """Genera opciones para el juego: 4 palabras similares pero NUNCA la target exacta."""
    # Obtener muchos vecinos (incluyendo el mismo idioma)
    neighbors = search_similar_word(target_word, k=50)

    # Filtrar: excluir la palabra exacta y asegurar variedad de idiomas
    valid_options = []
    used_langs = set()

    for neighbor in neighbors:
        # Excluir palabra exacta (insensible a mayúsculas)
        if neighbor['word'].lower() == target_word.lower():
            continue

        # Queremos variedad de idiomas
        if neighbor['language'] not in used_langs:
            valid_options.append({
                'word': neighbor['word'],
                'language': neighbor['language'],
                'is_correct': False
            })
            used_langs.add(neighbor['language'])

        if len(valid_options) >= 4:
            break

    # Si no alcanzamos 4 opciones, agregar aleatorias de otros idiomas
    while len(valid_options) < 4:
        random_data = get_random_word()
        if random_data and random_data['language'] not in used_langs:
            # Asegurarnos de no agregar la palabra objetivo por accidente
            if random_data['word'].lower() != target_word.lower():
                valid_options.append({
                    'word': random_data['word'],
                    'language': random_data['language'],
                    'is_correct': False
                })
                used_langs.add(random_data['language'])

    # La opción CORRECTA será la primera de la lista filtrada (la más cercana)
    if valid_options:
        valid_options[0]['is_correct'] = True

    # Mezclar para que no siempre esté en la misma posición
    random.shuffle(valid_options)

    # Guardar el índice correcto después de mezclar
    try:
        correct_index = next(i for i, opt in enumerate(valid_options) if opt['is_correct'])
        session['correct_index'] = correct_index
        session['correct_word'] = valid_options[correct_index]['word']  # Guardar para mostrar
    except StopIteration:
        session['correct_index'] = 0
        session['correct_word'] = valid_options[0]['word']

    return valid_options


@app.route('/', methods=['GET', 'POST'])
def dictionary():
    """Pestaña 1: Diccionario Semántico Multilingüe"""
    results = []
    query_word = ""
    selected_lang = "all"
    k = 5

    if request.method == 'POST':
        query_word = request.form.get('word', '').strip()
        selected_lang = request.form.get('language', 'all')
        try:
            k = max(1, min(int(request.form.get('k', 5)), 20))
        except:
            k = 5

        if query_word:
            lang_filter = None if selected_lang == "all" else selected_lang
            results = search_similar_word(query_word, lang_filter=lang_filter, k=k)

            if not results:
                flash(f"No se encontraron similares para '{query_word}'", "warning")

    return render_template('dictionary.html',
                           results=results,
                           query_word=query_word,
                           selected_lang=selected_lang,
                           k=k,
                           languages=LANGUAGES,
                           active_tab='dictionary')


@app.route('/game', methods=['GET', 'POST'])
def game():
    """Pestaña 2: Juego Conecta las Palabras"""
    if request.method == 'POST':
        if 'new_game' in request.form:
            # Nueva ronda: limpiar todo
            session.clear()
            random_data = get_random_word()
            if not random_data:
                flash("Error al cargar palabra", "error")
                return redirect(url_for('game'))

            session['game_word'] = random_data['word']
            session['game_lang'] = random_data['language']
            session['game_options'] = get_game_options(
                random_data['word'],
                random_data['language']
            )
            session['show_result'] = False
            return redirect(url_for('game'))

        elif 'check_answer' in request.form:
            # Verificar respuesta
            selected_idx = request.form.get('selected_option')
            if selected_idx and selected_idx.isdigit():
                idx = int(selected_idx)
                session['selected_idx'] = idx
                session['show_result'] = True
                session['is_correct'] = (idx == session.get('correct_index'))

            return redirect(url_for('game'))

    # Inicializar si no hay juego en curso
    if 'game_word' not in session:
        random_data = get_random_word()
        if random_data:
            session['game_word'] = random_data['word']
            session['game_lang'] = random_data['language']
            session['game_options'] = get_game_options(
                random_data['word'],
                random_data['language']
            )
            session['show_result'] = False

    return render_template('game.html',
                           target_word=session.get('game_word', ''),
                           target_lang=session.get('game_lang', ''),
                           target_lang_name=LANGUAGES.get(session.get('game_lang', ''), ''),
                           options=session.get('game_options', []),
                           show_result=session.get('show_result', False),
                           selected_idx=session.get('selected_idx', -1),
                           is_correct=session.get('is_correct', False),
                           correct_word=session.get('correct_word', ''),  # Nueva variable
                           languages=LANGUAGES,
                           active_tab='game')



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)