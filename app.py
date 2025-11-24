from flask import Flask, render_template, request, flash, session, redirect, url_for
import valkey
import numpy as np
from sentence_transformers import SentenceTransformer
import random
from sklearn.decomposition import PCA
import time

app = Flask(__name__)
app.secret_key = 'dev-key-change-in-production-semantic-dict-2025'

"""
=== GLOBAL CONFIGURATION ===
Configurate the database, the model word2vec and the Languages
"""
client = valkey.Valkey(host='localhost', port=6379, decode_responses=False)
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' # 12 Layers, Tokenization, Contextual Embeddings, Pooling, Output, Multilingual 50 languages
LANGUAGES = {
    'es': 'Español', 'en': 'Inglés', 'fr': 'Francés',
    'ru': 'Ruso', 'it': 'Italiano', 'de': 'Alemán',
    'pt': 'Portugués'
}
print("Loading model...")
model = SentenceTransformer(MODEL_NAME)
print("✓ Model loaded")


def search_similar_word(word: str, lang_filter: str = None, k: int = 5, exclude_lang: str = None, max_distance: float = 1.0):
    """
    Versatile search for similar words with filters.
    :param word:
    :param lang_filter:
    :param k:
    :param exclude_lang:
    :param max_distance:
    :return matches:
    """
    try:
        query_vector = model.encode([word])[0].astype(np.float32).tobytes() # Use the model to convert the word into a dense high-dimensional vector
        filter_parts = []
        if lang_filter:
            filter_parts.append(f"@language:{{{lang_filter}}}")
        if exclude_lang:
            filter_parts.append(f"-@language:{{{exclude_lang}}}")
        if filter_parts:
            base_expr = " ".join(filter_parts)
            if len(filter_parts) > 1:
                base_expr = f"({base_expr})"
            full_query = f"{base_expr}=>[KNN {k} @vector $vec]" # Search for the closest k vectors in the entire index
        else:
            full_query = f"*=>[KNN {k} @vector $vec]"
        results = client.execute_command(
            "FT.SEARCH", "words_idx",
            full_query,
            "PARAMS", "2", "vec", query_vector,
            "DIALECT", "2",
            "RETURN", "3", "word", "language", "vector",
            "LIMIT", "0", str(k)
        ) # ValkeySearch command for hybrid search
        # Hybrid search combines vector similarity with structured filters in real time
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
        print(f"Query usada: {full_query}")
        return []


def get_random_word():
    """
    Get a random word from the database.
    :return {word, language}:
    """
    try:
        # Try several times to avoid corrupted keys
        for _ in range(5):
            # Obtain random key on the server
            random_key = client.randomkey()
            if not random_key or not random_key.startswith(b'word:vector:'):
                continue
            # Use HMGET to obtain only necessary fields
            word, lang = client.hmget(random_key, 'word', 'language')
            if word and lang:
                return {'word': word.decode('utf-8'), 'language': lang.decode('utf-8')}
        return None
    except Exception as e:
        print(f"Error obteniendo palabra aleatoria: {e}")
        return None


def get_game_options(target_word, target_lang):
    """
    Generate options for the game: 4 similar words but NEVER the exact target.
    :param target_word:
    :param target_lang:
    :return valid_options:
    """
    # Get lots of neighbors (including the same language)
    neighbors = search_similar_word(target_word, k=50)
    # Filter: exclude the exact word and ensure language variety
    valid_options = []
    used_langs = set()
    for neighbor in neighbors:
        # Exclude exact word (case-insensitive)
        if neighbor['word'].lower() == target_word.lower():
            continue
        # Variety of languages
        if neighbor['language'] not in used_langs:
            valid_options.append({
                'word': neighbor['word'],
                'language': neighbor['language'],
                'is_correct': False
            })
            used_langs.add(neighbor['language'])
        if len(valid_options) >= 4:
            break
    # If we don't reach 4 options, add random ones from other languages
    while len(valid_options) < 4:
        random_data = get_random_word()
        if random_data and random_data['language'] not in used_langs:
            # Ensure that we do not accidentally add the word “objective.”
            if random_data['word'].lower() != target_word.lower():
                valid_options.append({
                    'word': random_data['word'],
                    'language': random_data['language'],
                    'is_correct': False
                })
                used_langs.add(random_data['language'])
    # The CORRECT option will be the first one on the filtered list (the closest one).
    if valid_options:
        valid_options[0]['is_correct'] = True
    # Mix so that it is not always in the same position.
    random.shuffle(valid_options)
    # Save the correct index after mixing
    try:
        correct_index = next(i for i, opt in enumerate(valid_options) if opt['is_correct'])
        session['correct_index'] = correct_index
        session['correct_word'] = valid_options[correct_index]['word']  # Save to display
    except StopIteration:
        session['correct_index'] = 0
        session['correct_word'] = valid_options[0]['word']
    return valid_options


def get_sample(limit=300, lang_filter="all"):
    """
    Recover vectors in batches with pipeline
    :param limit:
    :param lang_filter:
    :return words_data:
    """
    words_data = []
    keys_to_fetch = []
    # Collect keys quickly
    for key in client.scan_iter("word:vector:*", count=500):
        if len(keys_to_fetch) >= limit:
            break
        keys_to_fetch.append(key)
    if not keys_to_fetch:
        return []
    # Obtain all data in 1 call
    pipeline = client.pipeline()
    for key in keys_to_fetch:
        pipeline.hmget(key, 'word', 'language', 'vector')
    results = pipeline.execute()
    # Process results
    for result in results:
        if not result or len(result) != 3:
            continue
        word, lang, vector_bytes = result
        if not all([word, lang, vector_bytes]):
            continue
        lang_decoded = lang.decode('utf-8')
        if lang_filter != "all" and lang_filter != lang_decoded:
            continue
        words_data.append({
            'word': word.decode('utf-8'),
            'language': lang_decoded,
            'vector': np.frombuffer(vector_bytes, dtype=np.float32)
        })
    return words_data


def reduce_dimensions_fast(words_data, dimensions=2):
    """
    PCA optimization for 2D
    :param words_data:
    :param dimensions:
    :return {x, y, words, languages, explained_variance}:
    """
    if len(words_data) < 3:
        return None
    vectors = np.array([item['vector'] for item in words_data])
    # Use a maximum of 2 components for speed
    n_components = min(2, vectors.shape[0] - 1, vectors.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(vectors)
    return {
        'x': reduced[:, 0].tolist(),
        'y': reduced[:, 1].tolist(),
        'words': [item['word'] for item in words_data],
        'languages': [item['language'] for item in words_data],
        'explained_variance': float(pca.explained_variance_ratio_.sum())
    }


@app.route('/', methods=['GET', 'POST'])
def dictionary():
    """
    Tab 1: Multilingual Semantic Dictionary
    :return dictionary.html:
    """
    results = []
    query_word = ""
    selected_lang = "all"
    k = 5
    if request.method == 'POST':
        query_word = request.form.get('word', '').strip()
        selected_lang = request.form.get('language', 'all')
        try:
            k = max(1, min(int(request.form.get('k', 5)), 20)) # 20 maximum neighbors
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
    """
    Tab 2: Connect the Words Game
    :return game.html:
    """
    if request.method == 'POST':
        if 'new_game' in request.form:
            # New round: clean everything up
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
            # Verify answer
            selected_idx = request.form.get('selected_option')
            if selected_idx and selected_idx.isdigit():
                idx = int(selected_idx)
                session['selected_idx'] = idx
                session['show_result'] = True
                session['is_correct'] = (idx == session.get('correct_index'))
            return redirect(url_for('game'))
    # Initialise if no game is in progress
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
                           correct_word=session.get('correct_word', ''),  # New variable
                           languages=LANGUAGES,
                           active_tab='game')


@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    """
    2D display
    :return visualize.html:
    """
    plot_data = None
    selected_lang = "all"
    max_words = 200
    if request.method == 'POST':
        selected_lang = request.form.get('language', 'all')
        max_words = min(int(request.form.get('max_words', 200)), 10000)
    # Measure time
    start = time.time()
    words_data = get_sample(limit=max_words, lang_filter=selected_lang)
    load_time = time.time() - start
    print(f"Cargados {len(words_data)} vectores en {load_time:.2f}s")
    if len(words_data) >= 3:
        plot_data = reduce_dimensions_fast(words_data, 2)
    else:
        flash(f"Solo {len(words_data)} palabras encontradas. Ejecuta main.py primero.", "warning")
    return render_template('visualize.html',
                           plot_data=plot_data,
                           selected_lang=selected_lang,
                           max_words=max_words,
                           languages=LANGUAGES,
                           active_tab='visualize',
                           total_vectors=len(words_data),
                           load_time=f"{load_time:.2f}s")



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)