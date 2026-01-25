# 🚀 Deploy Online Guide - Recyclable Object Detection

## Pilihan Platform Deployment (GRATIS)

### Option 1: Hugging Face Spaces (RECOMMENDED) ⭐

**Langkah-langkah:**

1. **Buat akun Hugging Face**
   - Buka https://huggingface.co/join
   - Daftar/Login

2. **Create New Space**
   - Buka https://huggingface.co/new-space
   - Isi form:
     - **Space name**: `recyclable-detection` (atau nama lain)
     - **License**: MIT
     - **SDK**: Streamlit
     - **Visibility**: Public (gratis) atau Private

3. **Upload Files**
   - Setelah Space dibuat, klik tab "Files"
   - Upload semua file dari folder `deploy/`:
     - `app.py`
     - `best.pt`
     - `requirements.txt`
     - `README.md`
   
   Atau gunakan Git:
   ```bash
   cd deploy
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://huggingface.co/spaces/USERNAME/recyclable-detection
   git push -u origin main
   ```

4. **Tunggu Building**
   - Space akan otomatis build dan deploy
   - Proses sekitar 2-5 menit
   - URL: `https://huggingface.co/spaces/USERNAME/recyclable-detection`

---

### Option 2: Streamlit Community Cloud

**Langkah-langkah:**

1. **Push ke GitHub**
   ```bash
   cd deploy
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/USERNAME/recyclable-detection.git
   git push -u origin main
   ```

2. **Deploy di Streamlit Cloud**
   - Buka https://share.streamlit.io
   - Login dengan GitHub
   - Klik "New app"
   - Pilih repository `recyclable-detection`
   - Main file: `app.py`
   - Klik "Deploy"

⚠️ **Note**: GitHub punya limit file 100MB. Model kita 21.5MB jadi aman.

---

### Option 3: Render (Free Tier)

1. Push ke GitHub (sama seperti Option 2)
2. Buka https://render.com
3. Create New "Web Service"
4. Connect GitHub repo
5. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## File yang Harus Di-Upload

```
deploy/
├── app.py              # Aplikasi Streamlit
├── best.pt             # Model YOLOv8 (21.5 MB)
├── requirements.txt    # Dependencies
├── README.md           # Dokumentasi
└── .gitignore          # File yang diabaikan Git
```

## Troubleshooting

### "Model not found"
- Pastikan `best.pt` ada di folder yang sama dengan `app.py`

### "Memory Error" di Hugging Face
- Space gratis punya limit 16GB RAM, biasanya cukup untuk YOLOv8s

### "Build failed"
- Cek `requirements.txt` sudah benar
- Pastikan versi Python compatible (3.8-3.11)

---

## Model Performance Summary

| Metric | Value |
|--------|-------|
| mAP@0.5 | 63.4% |
| Precision | 71.8% |
| Recall | 70.0% |
| F1-Score | 0.709 |
| Model Size | 21.5 MB |
| Optimal Confidence | 0.20 |

---

**Selamat Deploy! ♻️**
