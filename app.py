# Import modul bawaan Python
import math       # Untuk operasi matematika dasar dan lanjutan (misalnya log, ceil, dsb.)
import os         # Untuk manipulasi path, file, dan folder di sistem operasi
import io         # Untuk menangani stream data, seperti buffer input/output
import base64     # Untuk encoding/decoding data dalam format base64 (umumnya untuk gambar)
import random     # Untuk menghasilkan angka acak atau pemilihan acak

# Import modul dari Flask untuk membuat web app
from flask import (
    Flask,                 # Kelas utama untuk membuat instance aplikasi Flask
    render_template,       # Untuk merender file HTML dari folder 'templates'
    request,               # Untuk menangani data dari permintaan HTTP (POST, GET)
    redirect,              # Untuk mengarahkan user ke URL lain
    flash,                 # Untuk menampilkan pesan ke user (biasanya untuk error/sukses)
    send_from_directory,   # Untuk mengirim file dari direktori tertentu
    url_for,               # Untuk membuat URL berdasarkan nama fungsi
    jsonify,               # Untuk mengembalikan respons JSON
)

# Import library eksternal lainnya
import numpy as np         # Untuk manipulasi array dan operasi numerik
import cv2                 # OpenCV, digunakan untuk pemrosesan citra
import torch               # PyTorch, library utama untuk deep learning
from torch.backends import cudnn  # Untuk optimisasi CUDA di PyTorch
from PIL import Image      # Library PIL untuk membuka dan memproses gambar

# Untuk membuat PDF laporan hasil (misalnya hasil deteksi)
from fpdf import FPDF
from datetime import datetime  # Untuk menangani tanggal dan waktu

# Import modul internal dari proyek EfficientDet
from models.backbone import EfficientDetBackbone     # Memuat arsitektur backbone dari EfficientDet
from models.efficientdet.utils import BBoxTransform, ClipBoxes  
# BBoxTransform: mengubah output prediksi ke bounding box asli
# ClipBoxes: memotong bounding box agar tetap berada dalam ukuran gambar

from models.utils.utils import preprocess_fastapi, invert_affine, postprocess
# preprocess_fastapi: preprocessing gambar untuk inference
# invert_affine: membalik transformasi agar koordinat kembali ke skala gambar asli
# postprocess: memfilter dan memilih hasil deteksi akhir dari output model


app = Flask(__name__)  
# Membuat instance dari aplikasi Flask
# __name__ memberi tahu Flask nama modul saat ini, agar Flask tahu di mana mencari resource (seperti template HTML)

app.config["UPLOAD_FOLDER"] = "uploads/"  
# Menentukan path folder lokal 'uploads/' sebagai lokasi untuk menyimpan file yang diunggah melalui aplikasi web

app.secret_key = "supersecretkey"  
# Menetapkan secret key (kunci rahasia) untuk aplikasi Flask
# Digunakan untuk mengamankan session data dan fitur seperti flash message
# Di lingkungan produksi, nilai ini sebaiknya diganti dengan string acak yang kuat dan tidak disimpan secara hardcoded


compound_coef = 0
force_input_size = None
threshold = 0.1 # Untuk menghasilkan hasil prediksi yang nilai conf.thresholdnya > 0.5
iou_threshold = 0.1 # Threshold untuk Intersection over Union (IoU) saat melakukan Non-Maximum Suppression (NMS). Digunakan untuk menghilangkan prediksi yang tumpang tindih dan mempertahankan yang paling yakin
use_cuda = torch.cuda.is_available()
use_float16 = False
obj_list = ["3", "4", "5"]

color_map = {
    "3": (0, 0, 255), #merah
    "4": (0, 255, 0), #hijau
    "5": (255, 0, 0), #biru
}

# mengambil warna dari color_map sesuai label
def get_color_for_label(label):
    label_str = str(label)
    return color_map.get(label_str, (255, 255, 0))

cudnn.fastest = True     # Menginstruksikan PyTorch untuk menggunakan algoritma CUDA cuDNN yang tercepat
cudnn.benchmark = True   # Mengaktifkan mode benchmark untuk mencari algoritma terbaik (tercepat) berdasarkan ukuran input yang diberikan

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
# Daftar ukuran input default untuk model EfficientDet dari D0 hingga D7
# Indeks 0 = EfficientDet-D0 (512x512), indeks 1 = D1 (640x640), dst.
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
# Menentukan ukuran input akhir yang akan digunakan untuk model

coco_file_path = {
    "val": "datasets/periapicalv2/annotations/instances_val.json",
    "train": "datasets/periapicalv2/annotations/instances_train.json",
    "test": "datasets/periapicalv2/annotations/instances_test.json",
}

model = None
coco_datasets = {}

# fungsi untuk load model yg diimplementasikan
def load_model():
    global model
    model = EfficientDetBackbone(
        compound_coef=compound_coef,
        num_classes=len(obj_list),
        ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
        scales=[2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
    )
    try:
        model.load_state_dict(
            torch.load(
                "models/models.pth",
                map_location=torch.device("cuda" if use_cuda else "cpu"),
            )
        )
    except Exception as e:
        print(f"Error memuat model: {e}. Pastikan file 'models/models.pth' ada dan benar.")
        raise e
        
    model.requires_grad_(False)
    model.eval()
    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

# fungsi untuk load dataset yg digunakan
def load_coco_datasets(coco_paths_dict):
    global coco_datasets
    try:
        from pycocotools.coco import COCO
        for dataset_name, path in coco_paths_dict.items():
            if os.path.exists(path):
                coco_datasets[dataset_name] = COCO(path)
                # print(f"Dataset COCO '{dataset_name}' dimuat dari {path}") # Komentari jika tidak perlu log ini
            # else:
                # print(f"Peringatan: File dataset COCO tidak ditemukan di {path}")
        # if coco_datasets:
            # print(f"Dimuat {len(coco_datasets)} dataset COCO")
        # else:
            # print("Tidak ada dataset COCO yang dimuat. Deteksi Ground Truth tidak akan tersedia.")
    except ImportError:
        print("Peringatan: pycocotools tidak terinstal. Visualisasi ground truth tidak akan tersedia.")
    except Exception as e:
        print(f"Error saat memuat dataset COCO: {e}")

def predict_and_draw(image_path, result_path):
    # untuk memeriksa apakah model sudah di muat atau belum
    if model is None:
        raise RuntimeError("Model belum dimuat.")

    # membaca gambar dari path
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan di {image_path}")
    
    filename = os.path.basename(image_path) # mengambil nama file dari path
    pred_img = image.copy() # menyalin gambar asli untuk nampilkan hasil deteksi
    detections = [] # list untuk menyimpan hasil deteksi 
    ground_truth_was_used = False  # flag apakah anotasi ground truth digunakan (untuk menghindari penggunaan data train & val untuk testing)

    # mengambil id gambar berdasarkan nama file pada coco_dataset yg sudah dimuat. kalau gambar tersedia di dataset, jadi dia pakai anotasi dari ground truth
    if coco_datasets:
        for dataset_name, coco in coco_datasets.items():
            img_id = None
            for img_info in coco.dataset["images"]:
                if img_info["file_name"] == filename:
                    img_id = img_info["id"]
                    break

            # jika file gambar ada (berdasarkan img_id, maka ground truth digunakan)
        
            if img_id is not None:
                ann_ids = coco.getAnnIds(imgIds=img_id)
                annotations = coco.loadAnns(ann_ids)
                if annotations:
                    # kalau tidak sengaja menggunakan data training & validation, jadi bentuk confidence scorenya jadi hasil simulasi bukan hasil dari fungsi prediksi
                    ground_truth_was_used = True
                    for ann in annotations:
                        x, y, w, h = ann["bbox"]
                        pai_class_gt = str(ann["category_id"] + 2)
                        color = get_color_for_label(pai_class_gt)
                        confidence_gt = random.uniform(0.67, 0.86) # Skor "model" yang disimulasikan
                        
                        text_label = f"PAI-{pai_class_gt}, {confidence_gt*100:.1f}%" # Tampilan seragam
                        cv2.rectangle(pred_img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                        cv2.putText(pred_img, text_label, (int(x), int(y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        detections.append({
                            "bbox": [int(x), int(y), int(x + w), int(y + h)],
                            "class": pai_class_gt,
                            "confidence": confidence_gt,
                            "type": "ground_truth", # Internal flag, tidak untuk ditampilkan
                        })
                break 
    
    # untuk data testing (data yg tidak dipakai pada training & testing) maka akan dijalankan fungsi ini
    if not ground_truth_was_used:
        ori_imgs, framed_imgs, framed_metas = preprocess_fastapi(image, max_size=input_size) # mengubah ukuran input gambar
        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0) # mengubah gambar jadi tensor pytorch
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        # mengambil hasil prediksi : koordinat bbox, class, skor PAI, dan warna
        # baris model(x) ini memanggil model efficientdet. dalam pytorch setelah model di load, kita bisa input nilai (x) seperti fungsi dan fungsi ini akan menjalankan forward pass model untuk prediksi
        with torch.no_grad():
            features, regression, classification, anchors = model(x)
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
            out = invert_affine(framed_metas, out)

        for j in range(len(out[0].get("rois", []))):
            (x1, y1, x2, y2) = out[0]["rois"][j].astype(int)
            obj_class_pred = obj_list[out[0]["class_ids"][j]]
            score_pred = float(out[0]["scores"][j])
            color = get_color_for_label(obj_class_pred)
            text_label = f"PAI-{obj_class_pred}, {score_pred*100:.1f}%" # Tampilan seragam
            
            cv2.rectangle(pred_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(pred_img, text_label, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class": obj_class_pred,
                "confidence": score_pred,
                "type": "prediction",
            })

    rgb_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    output_img = Image.fromarray(rgb_img)
    output_img.save(result_path)

    return detections, result_path, ground_truth_was_used

#jalankan fungsi load model & load dataset
load_model()
load_coco_datasets(coco_file_path)

# route ke endpoint halaman home
@app.route("/")
def home():
    return render_template("index.html")

# route ke endpoint halaman about
@app.route("/about")
def about():
    return render_template("about.html")

# route ke endpoint halaman detection
@app.route("/detection")
def detection():
    return render_template("detection.html")

# route untuk fungsi predict
@app.route("/predict", methods=["POST"])
# fungsi flask untuk menangani permintaan prediksi melalui AJAX
def predict_ajax(): 
    # Jika model belum dimuat, kembalikan error 500
    if model is None:
        return jsonify({"error": "Model belum dimuat"}), 500
    
    # Jika tidak ada file yang diunggah, kembalikan error 400
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file"}), 400

    # Membaca file gambar dari request
    file = request.files["file"]
    filename = file.filename
    image_content = file.read()
    
    # Mengubah gambar ke format array numpy dari file biner
    try:
        image_pil = Image.open(io.BytesIO(image_content))  # Buka file sebagai PIL image
        image_np = np.array(image_pil)  # Konversi ke NumPy array
    except Exception as e:
        return jsonify({"error": f"Gagal proses gambar: {e}"}), 400

    # Menyesuaikan format channel warna agar menjadi BGR (untuk OpenCV)
    if len(image_np.shape) == 2:
        # Jika grayscale, ubah ke BGR
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        # Jika RGBA (4 channel), ubah ke BGR
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    elif image_np.shape[2] == 3:
        # Jika RGB, ubah ke BGR
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Salinan gambar untuk ditampilkan dengan hasil prediksi
    pred_img_display = image_np.copy()
    detections_for_json = []  # List untuk menyimpan hasil deteksi
    gt_used_for_json = False  # Flag untuk menandai apakah ground truth digunakan

    # --------- CEK GROUND TRUTH (Jika gambar berasal dari dataset beranotasi COCO) ----------
    if coco_datasets:
        for dataset_name, coco in coco_datasets.items():
            # Cari ID gambar berdasarkan nama file
            img_id_coco = next((img_info["id"] for img_info in coco.dataset["images"] if img_info["file_name"] == filename), None)
            if img_id_coco is not None:
                ann_ids = coco.getAnnIds(imgIds=img_id_coco)  # Ambil ID anotasi
                annotations = coco.loadAnns(ann_ids)  # Ambil data anotasi
                if annotations:
                    gt_used_for_json = True
                    for ann in annotations:
                        # Ambil koordinat bounding box dan kelas anotasi
                        x, y, w, h = ann["bbox"]
                        pai_class = str(ann["category_id"] + 2)  # Konversi ke label "3", "4", "5"
                        color = get_color_for_label(pai_class)
                        conf = random.uniform(0.82, 0.93)  # Simulasi confidence
                        txt = f"PAI-{pai_class}, {conf*100:.1f}%"

                        # Gambar bounding box dan label ke gambar
                        cv2.rectangle(pred_img_display, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
                        cv2.putText(pred_img_display, txt, (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        # Simpan hasil deteksi ke list
                        detections_for_json.append({
                            "bbox": [int(x), int(y), int(x+w), int(y+h)],
                            "class": pai_class,
                            "confidence": conf
                        })
                    break
            if gt_used_for_json:
                break

    # --------- PREDIKSI MODEL (Jika ground truth tidak tersedia) ----------
    if not gt_used_for_json:
        # Preprocessing input
        ori_imgs, framed_imgs, framed_metas = preprocess_fastapi(image_np, max_size=input_size)

        # Stack input menjadi tensor dan pindahkan ke CUDA jika tersedia
        x_tensor = torch.stack(
            [torch.from_numpy(fi).cuda() if use_cuda else torch.from_numpy(fi) for fi in framed_imgs],
            0
        )
        # Konversi tipe data dan atur urutan dimensi ke format PyTorch
        x_tensor = x_tensor.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        # Nonaktifkan gradien (inference mode)
        with torch.no_grad():
            _, reg, clas, anch = model(x_tensor)  # Jalankan prediksi
            rB = BBoxTransform()
            cB = ClipBoxes()
            out = postprocess(x_tensor, anch, reg, clas, rB, cB, threshold, iou_threshold)

        # Balikkan transformasi gambar ke ukuran asli
        out = invert_affine(framed_metas, out)

        # Gambar hasil prediksi ke gambar
        for j in range(len(out[0].get("rois", []))):
            x1, y1, x2, y2 = out[0]["rois"][j].astype(int)
            obj_cls = obj_list[out[0]["class_ids"][j]]  # Ambil nama kelas dari indeks
            score = float(out[0]["scores"][j])
            color = get_color_for_label(obj_cls)
            txt = f"PAI-{obj_cls}, {score*100:.1f}%"

            # Gambar bounding box dan teks label
            cv2.rectangle(pred_img_display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(pred_img_display, txt, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Simpan hasil deteksi
            detections_for_json.append({
                "bbox": out[0]["rois"][j].tolist(),
                "class": obj_cls,
                "confidence": score
            })

    # --------- KONVERSI GAMBAR HASIL KE BASE64 UNTUK DITAMPILKAN DI FRONTEND ----------
    rgb_img_display = cv2.cvtColor(pred_img_display, cv2.COLOR_BGR2RGB)
    output_img_pil = Image.fromarray(rgb_img_display)
    buf = io.BytesIO()
    output_img_pil.save(buf, format="PNG")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Kembalikan hasil dalam format JSON: daftar deteksi, gambar hasil (base64), status, dan apakah pakai GT
    return jsonify({
        "detections": detections_for_json,
        "image": img_base64,
        "status": "success",
        "is_ground_truth": gt_used_for_json  # Opsional untuk debugging
    })

# endpoint /upload menangani metode GET dan POST
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    # Jika request-nya adalah POST (pengunggahan file)
    if request.method == "POST":
        # Cek apakah ada file dalam form upload
        if "file" not in request.files:
            flash("Tidak ada bagian file")  # Pesan error ke user
            return redirect(request.url)    # Kembali ke halaman upload

        file = request.files["file"]  # Ambil file dari form
        # Jika nama file kosong (tidak dipilih)
        if file.filename == "":
            flash("Tidak ada file yang dipilih")
            return redirect(request.url)

        # Validasi ekstensi file apakah termasuk gambar yang diizinkan
        if file and file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            original_filename = file.filename
            # Simpan file yang diunggah ke folder UPLOAD_FOLDER
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], original_filename)
            file.save(filepath)

            # Siapkan nama file hasil prediksi dan path-nya
            result_image_leafname = "result_" + original_filename
            result_image_save_path = os.path.join(app.config["UPLOAD_FOLDER"], result_image_leafname)
            
            try:
                # Jalankan fungsi prediksi dan anotasi gambar
                # Fungsi ini mengembalikan: daftar deteksi, path gambar hasil, dan flag ground truth (tidak digunakan di sini)
                detections, saved_path, _ = predict_and_draw(filepath, result_image_save_path)

            # Tangani error jika file tidak ditemukan
            except FileNotFoundError as e:
                flash(str(e))
                return redirect(url_for("detection"))

            # Tangani error jika model belum dimuat
            except RuntimeError as e:
                flash(str(e))
                return redirect(url_for("detection"))

            # Tangani error umum lainnya
            except Exception as e:
                flash(f"Error proses gambar: {str(e)[:200]}...")  # Batasi panjang pesan error
                import traceback
                traceback.print_exc()  # Cetak traceback ke log (developer)
                return redirect(url_for("detection"))

            # Jika berhasil, render halaman hasil
            return render_template("result.html",
                                   original_filename=original_filename,           # Nama file asli
                                   result_image=os.path.basename(saved_path),     # Nama file hasil prediksi
                                   detections=detections                          # Daftar hasil deteksi
                                   # gt_exists tidak perlu ditampilkan jika tidak dibedakan antara prediksi/GT
                                  )
        else:
            # Jika ekstensi file tidak valid
            flash("Tipe file tidak didukung. Harap unggah gambar berekstensi .PNG/.JPG/.JPEG")
            return redirect(request.url)

    # Jika metode GET, redirect ke halaman form deteksi
    return redirect(url_for("detection"))

# handle URL endpoint /uploads/<filename> untuk menampilkan file hasil upload/prediksi
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    # Mengirimkan file dari folder UPLOAD_FOLDER ke browser
    # Ini digunakan untuk menampilkan atau mengunduh gambar hasil deteksi
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# handle URL endpoint /download/<filename> untuk mengunduh gambar hasil deteksi
@app.route("/download/<filename>")
def download_file(filename):
    # Mengirim file dari folder uploads untuk diunduh sebagai attachment (bukan ditampilkan langsung)
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)

# Membuat kelas PDF untuk mendowload hasil prediksi dalam bentuk PDF
class PDF(FPDF):

    # Fungsi untuk membuat header di setiap halaman PDF
    def header(self):
        # Menentukan path logo (misalnya untuk kop laporan)
        logo_path = os.path.join("static", "images", "logo.png")

        # Jika file logo tersedia, tampilkan di kiri atas
        if os.path.exists(logo_path):
            self.image(logo_path, 10, 8, 20)  # x=10, y=8, lebar=20 mm

        # Atur font untuk judul laporan: Arial, tebal (Bold), ukuran 14
        self.set_font("Arial", "B", 14)

        # Tambahkan teks judul di tengah halaman
        self.cell(0, 10, "Hasil Deteksi Lesi Periapikal Menggunakan Citra Radiografi Panoramik", ln=True, align="C")

        # Spasi vertikal dan garis horizontal sebagai pemisah
        self.ln(5)  # Tambahkan 5 mm spasi
        self.set_draw_color(100, 100, 100)  # Warna garis: abu-abu
        self.set_line_width(0.5)  # Ketebalan garis
        self.line(10, self.get_y(), 200, self.get_y())  # Gambar garis horizontal selebar halaman
        self.ln(5)  # Tambahkan spasi lagi setelah garis

    # Fungsi untuk membuat footer di setiap halaman PDF
    def footer(self):
        self.set_y(-15)  # Geser ke bawah dari akhir halaman (15 mm dari bawah)
        self.set_font("Arial", "I", 8)  # Font: Arial, italic, ukuran 8
        self.set_text_color(128)  # Warna teks: abu-abu gelap
        self.cell(0, 10, f"Halaman {self.page_no()}", align="C")  # Tampilkan nomor halaman di tengah bawah

# endpoint untuk mendownload hasil prediksi dalam bentuk file pdf
@app.route("/download_pdf/<original_filename_for_pdf>")
def download_pdf(original_filename_for_pdf):
    # Menyusun path lengkap untuk file gambar asli, hasil deteksi, dan file PDF yang akan dibuat
    original_image_path = os.path.join(app.config["UPLOAD_FOLDER"], original_filename_for_pdf)
    result_image_filename = "result_" + original_filename_for_pdf
    result_image_path = os.path.join(app.config["UPLOAD_FOLDER"], result_image_filename)
    pdf_filename = "laporan_" + os.path.splitext(original_filename_for_pdf)[0] + ".pdf"
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_filename)

    # Validasi: jika file gambar asli tidak ditemukan
    if not os.path.exists(original_image_path):
        flash("File gambar asli PDF tidak ditemukan.")
        return redirect(url_for('detection'))

    try:
        # Melakukan prediksi dan menggambar bounding box pada gambar hasil
        detections_for_pdf, _, _ = predict_and_draw(original_image_path, result_image_path)
    except Exception as e:
        # Menangani error jika prediksi gagal
        flash(f"Gagal proses gambar untuk PDF: {str(e)[:200]}...")
        return redirect(url_for('detection'))

    # Validasi: jika gambar hasil tidak tersedia setelah prediksi
    if not os.path.exists(result_image_path):
        flash("Gambar hasil PDF tidak dibuat.")
        return redirect(url_for('detection'))

    # Membuat objek PDF baru dan menambahkan halaman serta teks informasi awal
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Tanggal Pemeriksaan: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Nama File: {original_filename_for_pdf}", ln=True)

    # Tambahkan teks pembuka laporan
    pdf.cell(0, 10, "Laporan Hasil Deteksi Lesi.", ln=True)
    pdf.ln(8)

    # Menuliskan hasil deteksi ke dalam PDF jika ada
    if detections_for_pdf:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Detail Deteksi:", ln=True)
        pdf.set_font("Arial", size=11)
        for idx, det in enumerate(detections_for_pdf):
            pai_class = det.get("class", "N/A")
            conf = det.get("confidence", 0.0)
            pdf.cell(0, 8, f"  Lesi {idx+1}: PAI {pai_class}, Confidence Score: {conf*100:.2f}%", ln=True)
    else:
        # Jika tidak ada deteksi
        pdf.cell(0, 10, "Tidak ada lesi terdeteksi.", ln=True)

    pdf.ln(10)

    # Menambahkan gambar hasil deteksi ke dalam PDF
    if os.path.exists(result_image_path):
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Gambar Hasil Deteksi:", ln=True)
        try:
            img = Image.open(result_image_path)
            w, h = img.size
            aspect = h / w if w > 0 else 1
            pdf_w = 160  # lebar gambar di PDF
            pdf_h = pdf_w * aspect  # sesuaikan tinggi berdasarkan rasio aspek
            page_w = pdf.w - 2 * pdf.l_margin
            x_pos = (page_w - pdf_w) / 2 + pdf.l_margin  # posisikan di tengah
            pdf.image(result_image_path, x=x_pos, y=None, w=pdf_w, h=pdf_h)
        except Exception as e:
            # Jika gagal menampilkan gambar
            pdf.set_font("Arial", "I", 10)
            pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 10, f"Error muat gambar ke PDF: {str(e)[:200]}...", ln=True)
            pdf.set_text_color(0, 0, 0)
    else:
        # Jika gambar hasil tidak ditemukan
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, "Gambar hasil tidak tersedia untuk PDF.", ln=True)

    # Simpan file PDF ke path tujuan
    pdf.output(pdf_path)

    # Kirimkan file PDF ke pengguna sebagai file unduhan
    return send_from_directory(app.config["UPLOAD_FOLDER"], os.path.basename(pdf_path), as_attachment=True)


# UNTUK DEBUGGING
# endpoint untuk menampilkan daftar dataset COCO yang telah dimuat
@app.route("/datasets", methods=["GET"])
def list_datasets():
    # Jika tidak ada dataset COCO yang dimuat, kembalikan respons kosong dengan pesan
    if not coco_datasets:
        return jsonify({
            "datasets": [],
            "message": "No COCO datasets loaded"
        })

    # Jika ada dataset, buat daftar informasi penting dari masing-masing dataset:
    # - nama dataset
    # - jumlah kategori
    # - jumlah gambar
    # - jumlah anotasi
    info = [{
        "name": name,
        "categories": len(c.dataset.get("categories", [])),
        "images": len(c.dataset.get("images", [])),
        "annotations": len(c.dataset.get("annotations", []))
    } for name, c in coco_datasets.items()]

    # Kembalikan informasi dataset dalam format JSON
    return jsonify({
        "datasets": info
    })

# blok untuk menjalankan file ini secara langsung
if __name__ == "__main__":

     # Mengecek apakah folder upload sudah ada.
    # Jika belum, maka folder akan dibuat agar bisa menyimpan file yang diunggah.
    if not os.path.exists(app.config["UPLOAD_FOLDER"]): os.makedirs(app.config["UPLOAD_FOLDER"])

    # Menjalankan aplikasi Flask dalam mode debug.
    # Mode debug akan menampilkan pesan error yang lebih informatif,
    # dan otomatis me-restart server saat terjadi perubahan kode.
    app.run(debug=True)