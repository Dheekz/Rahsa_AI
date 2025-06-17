# File: app.py

import streamlit as st
import google.generativeai as genai
import json
import os
import bcrypt
from config import API_KEY

# Library untuk RAG dan pemrosesan file
import pandas as pd
import docx
import fitz  # PyMuPDF
from PIL import Image
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np

# --- Konfigurasi Gemini API ---
if not API_KEY or API_KEY == "MASUKKAN_API_KEY_ANDA_DI_SINI":
    st.error("API Key Gemini belum diatur di config.py. Silakan isi.")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
    generation_model = genai.GenerativeModel('gemini-1.5-flash')
    embedding_model = genai.GenerativeModel('models/embedding-001')
except Exception as e:
    st.error(f"Gagal mengkonfigurasi Gemini API: {e}")
    st.stop()


# --- Fungsi Helper (Login, User, dll - tidak berubah) ---
def load_users():
    if not os.path.exists("users.json"):
        with open("users.json", "w") as f: json.dump({}, f)
        return {}
    try:
        with open("users.json", "r") as f: return json.load(f)
    except json.JSONDecodeError: return {}

def save_users(users_data):
    with open("users.json", "w") as f: json.dump(users_data, f, indent=4)

def save_chat(username, chat_history):
    os.makedirs("chat_history", exist_ok=True)
    with open(f"chat_history/{username}.json", "w") as f: json.dump(chat_history, f, indent=4)

def load_chat(username):
    chat_file = f"chat_history/{username}.json"
    if not os.path.exists(chat_file): return []
    try:
        with open(chat_file, "r") as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return []

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed_password):
    try:
        if isinstance(hashed_password, str): hashed_password = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password)
    except Exception: return False


# --- Fungsi untuk RAG dan Pemrosesan File ---

def get_text_from_file(uploaded_file):
    """Mengekstrak teks mentah dari berbagai jenis file untuk RAG."""
    file_bytes = uploaded_file.getvalue()
    text_content = ""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        if file_extension == '.pdf':
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text_content = "".join(page.get_text() for page in doc)
        elif file_extension == '.docx':
            doc = docx.Document(io.BytesIO(file_bytes))
            text_content = "\n".join([para.text for para in doc.paragraphs])
        elif file_extension == '.xlsx':
            df = pd.read_excel(io.BytesIO(file_bytes))
            text_content = df.to_markdown(index=False)
        else:
            return None
        return text_content
    except Exception as e:
        st.error(f"Gagal mengekstrak teks dari {uploaded_file.name}: {e}")
        return None

def get_text_chunks(text):
    """Memecah teks menjadi potongan-potongan yang lebih kecil (chunks)."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Membuat embeddings dan menyimpannya di database vektor FAISS."""
    if not text_chunks: return None
    try:
        response = genai.embed_content(model=embedding_model.model_name, content=text_chunks, task_type="retrieval_document")
        embeddings = response['embedding']
        d = len(embeddings[0])
        index = faiss.IndexFlatL2(d)
        index.add(np.array(embeddings))
        return index, text_chunks
    except Exception as e:
        st.error(f"Gagal membuat vector store: {e}")
        return None, None

def get_relevant_context(query, index, text_chunks):
    """Mencari potongan teks yang relevan dari vector store."""
    try:
        query_embedding = genai.embed_content(model=embedding_model.model_name, content=query, task_type="retrieval_query")['embedding']
        D, I = index.search(np.array([query_embedding]), k=3)
        context = "\n".join([text_chunks[i] for i in I[0]])
        return context
    except Exception as e:
        st.error(f"Gagal mengambil konteks relevan: {e}")
        return ""

def handle_general_upload(uploaded_file):
    """Menangani upload file umum (gambar)."""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension in ['.jpg', '.jpeg', '.png']:
        try:
            image = Image.open(uploaded_file)
            st.success(f"Gambar '{uploaded_file.name}' siap untuk ditanyakan.")
            return image
        except Exception as e:
            st.error(f"Gagal memuat gambar: {e}")
            return None
    return None

def format_chat_for_download(chat_history):
    formatted_string = f"Riwayat Percakapan - {st.session_state.username}\n" + "=" * 40 + "\n\n"
    for msg in chat_history:
        role = "Anda" if msg['role'] == 'user' else "Asisten AI"
        formatted_string += f"[{role}]:\n{msg['content']}\n\n" + "-" * 40 + "\n\n"
    return formatted_string

# --- Inisialisasi Aplikasi Streamlit ---
st.set_page_config(page_title="AI RAG Perusahaan", layout="wide")
st.title("🤖 Rahsa AI")

# Inisialisasi session state
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "text_chunks" not in st.session_state: st.session_state.text_chunks = None
if "processed_filename" not in st.session_state: st.session_state.processed_filename = ""
if "general_file_content" not in st.session_state: st.session_state.general_file_content = None

# --- Halaman Login & Register (tidak berubah) ---
if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["🔐 Login", "✍️ Register"])
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.form_submit_button("Login"):
                users = load_users()
                if username in users and check_password(password, users[username]):
                    st.session_state.logged_in, st.session_state.username = True, username
                    st.session_state.chat_history = load_chat(username)
                    st.success("Login berhasil!"), st.rerun()
                else: st.error("Username atau password salah.")
    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("Username Baru", key="reg_user")
            new_pass = st.text_input("Password Baru", type="password", key="reg_pass")
            if st.form_submit_button("Register"):
                users = load_users()
                if new_user in users: st.error("Username sudah terdaftar.")
                else:
                    users[new_user] = hash_password(new_pass)
                    save_users(users), st.success("Registrasi berhasil! Silakan login.")

# --- Halaman Chat Utama ---
else:
    # --- Sidebar ---
    st.sidebar.title("Opsi")
    st.sidebar.write(f"Selamat datang, **{st.session_state.username}**!")
    if st.sidebar.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.chat_history = []
        if os.path.exists(f"chat_history/{st.session_state.username}.json"):
            os.remove(f"chat_history/{st.session_state.username}.json")
        st.success("Riwayat percakapan telah dihapus."), st.rerun()
    st.sidebar.download_button(label="📥 Unduh Riwayat Chat", data=format_chat_for_download(st.session_state.chat_history),
                               file_name=f"chat_history_{st.session_state.username}.txt", mime="text/plain",
                               use_container_width=True, disabled=not st.session_state.chat_history)
    if st.sidebar.button("Logout", use_container_width=True):
        save_chat(st.session_state.username, st.session_state.chat_history)
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

    # --- Area Unggah RAG (Sidebar) ---
    st.sidebar.header("Basis Pengetahuan (RAG)")
    rag_file = st.sidebar.file_uploader("Unggah PDF/DOCX/XLSX untuk RAG.", type=['pdf', 'docx', 'xlsx'])
    if rag_file and rag_file.name != st.session_state.processed_filename:
        with st.sidebar:
            with st.spinner(f"Memproses {rag_file.name}..."):
                raw_text = get_text_from_file(rag_file)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    vector_store, chunks = get_vector_store(text_chunks)
                    if vector_store:
                        st.session_state.vector_store, st.session_state.text_chunks, st.session_state.processed_filename = vector_store, chunks, rag_file.name
                        st.success(f"'{rag_file.name}' siap ditanyai.")
    if st.session_state.processed_filename:
        st.sidebar.info(f"Konteks aktif: **{st.session_state.processed_filename}**")
        if st.sidebar.button("Hapus Konteks", use_container_width=True):
            st.session_state.vector_store, st.session_state.text_chunks, st.session_state.processed_filename = None, None, ""
            st.rerun()

    # --- Area Chat ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image" in msg:
                st.image(msg["image"])

    # --- Area Unggah File Umum (Halaman Utama) ---
    with st.expander("📎 Analisis File Sekali Pakai (Contoh: Gambar)"):
        general_file = st.file_uploader("Unggah file gambar", type=['jpg', 'jpeg', 'png'])
        if general_file:
            st.session_state.general_file_content = handle_general_upload(general_file)

    if prompt := st.chat_input("Ketik pertanyaan Anda..."):
        # Tambah pesan ke histori UI dulu
        user_message = {"role": "user", "content": prompt}
        if st.session_state.general_file_content:
            user_message["image"] = st.session_state.general_file_content
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(prompt)
            if st.session_state.general_file_content:
                st.image(st.session_state.general_file_content, width=200)

        with st.spinner("AI sedang berpikir..."):
            try:
                # Siapkan konten untuk API
                api_request = []
                final_prompt = prompt

                # Prioritas 1: File umum (gambar)
                if st.session_state.general_file_content:
                    api_request.append(prompt)
                    api_request.append(st.session_state.general_file_content)
                # Prioritas 2: Konteks RAG
                elif st.session_state.vector_store:
                    context = get_relevant_context(prompt, st.session_state.vector_store, st.session_state.text_chunks)
                    final_prompt = f"Berdasarkan konteks berikut:\n---\n{context}\n---\nJawablah pertanyaan ini: {prompt}"
                    api_request.append(final_prompt)
                # Prioritas 3: Teks biasa
                else:
                    api_request.append(prompt)

                # Format histori chat untuk API
                formatted_history = []
                for msg in st.session_state.chat_history[:-1]:
                    # Jangan masukkan gambar dari histori ke dalam request API berikutnya
                    if "image" not in msg:
                        role = "model" if msg["role"] == "assistant" else msg["role"]
                        formatted_history.append({"role": role, "parts": [{"text": msg["content"]}]})

                # Kirim ke API
                chat_session = generation_model.start_chat(history=formatted_history)
                response = chat_session.send_message(api_request)

                # Tampilkan dan simpan respons
                with st.chat_message("assistant"):
                    st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                save_chat(st.session_state.username, st.session_state.chat_history)
                
                # Reset file umum setelah digunakan
                st.session_state.general_file_content = None
                st.rerun()

            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghubungi AI: {e}")
