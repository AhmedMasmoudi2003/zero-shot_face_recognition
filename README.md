# Zero-Shot Face Recognition Attendance System

A high-performance zero-shot face recognition system powered by **FaceNet (InceptionResnetV1)** and **Pinecone Vector Database**. The system recognizes individuals in real time without retraining the neural network—simply add a photo or enroll through your webcam, and similarity search identifies the person instantly.

> [!NOTE]
> **Project Status**:
> - 🟢 **Real-Time Webcam Face Recognition Interface**: **Ready & fully functional!** Follow the quick-start guide below to test it immediately.
> - 🟡 **Full FastAPI Attendance Backend**: In development (API endpoints for employee management & automated check-in/check-out are functional, but there is no frontend yet).

---

## ⚡ Quick Start: Real-Time Webcam Face Recognition

Follow these steps to run the real-time face detection and recognition interface on your computer.

### Step 1: Clone the Repository & Navigate to Backend
```bash
git clone https://github.com/your-username/zero-shot_face_recognition.git
cd zero-shot_face_recognition/Backend
```

---

### Step 2: Set Up Pinecone (Vector Database)
The system uses Pinecone to store and search 512-dimensional facial embeddings using cosine similarity.

1. Go to [Pinecone (pinecone.io)](https://www.pinecone.io/) and create a free account.
2. In your Pinecone dashboard, navigate to **API Keys** and copy your API key.
3. *(Optional)* You do **not** need to manually create an index—the scripts will automatically create `faces-index` (dimension: 512, metric: cosine) if it does not exist yet.

---

### Step 3: Configure Environment Variables (`.env`)
Create a `.env` file in the `Backend` directory (or copy `.env.example`):

```powershell
cp .env.example .env
```

Open `.env` and fill in your Pinecone credentials:

```dotenv
# REQUIRED for face recognition
PINECONE_API_KEY=your_actual_pinecone_api_key_here
PINECONE_INDEX_NAME=faces-index
PINECONE_ENVIRONMENT=us-east-1-aws

# OPTIONAL: Only needed if you run the full FastAPI backend later.
# NOT NEEDED for the webcam recognition test!
DATABASE_URL=postgresql://postgres:admin@localhost:5432/attendance_db
```

> [!IMPORTANT]
> **PostgreSQL is NOT needed for the webcam test.** You only need your `PINECONE_API_KEY` to run the camera interface and enroll faces.

---

### Step 4: Set Up Virtual Environment & Dependencies

1. **Create and activate a virtual environment:**

   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
     *(If PowerShell blocks execution scripts, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Optional: If you have an NVIDIA GPU, PyTorch will automatically utilize CUDA for real-time inference).*

---

### Step 5: Enroll Faces into the Vector DB

You have two simple ways to enroll faces:

#### Option A: Batch Enroll from Photos (Recommended)
1. Drop any clear photo of a person into the `Backend/scripts/images/` folder.
2. Name the image after the person (e.g., `ahmed.jpg`, `sarah.png`, `john.jpg`).
3. Run the batch enrollment script:
   ```bash
   cd scripts
   python batch_enroll.py
   ```
   *The script checks Pinecone to avoid duplicate uploads, extracts the 512-d facial embedding, and upserts it into your Pinecone index.*

#### Option B: Live Webcam Enrollment
If you want to enroll directly from your webcam:
```bash
cd scripts
python enroll.py
```
- Enter the person's unique ID and display name when prompted.
- Look directly into the webcam and press **`SPACE`** to capture and enroll.

---

### Step 6: Run the Real-Time Face Recognition Interface

Launch the real-time webcam detector:

#### ⚡ Fast 1-Line Command (Windows PowerShell):
```powershell
cd D:\work\AI\zero-shot_face_recognition\Backend\scripts; ..\.venv\Scripts\python.exe recognize.py
```

#### Standard Command (From `Backend/scripts/`):
```bash
python recognize.py
```

### 🎮 What to Expect
- A window titled **"Smart Face Recognition"** will open with your live camera feed.
- **Dual-Threaded Architecture**:
  - **Thread 1 (Tracking)**: Detects and smoothly tracks faces at ~30 FPS using MTCNN.
  - **Thread 2 (Recognition)**: Extracts the face embedding and queries Pinecone every 1.0 second.
- **Visual Feedback**:
  - 🟩 **Green bounding box + Name & Score** (e.g. `Ahmed (0.95)`) if similarity $\ge 0.51$.
  - 🟥 **Red bounding box + `Unknown`** if the person is not enrolled.
- **Exit**: Click the camera window and press **`q`** to quit.

---

## ⚙️ Configuration & Tuning (`scripts/recognize.py`)

You can customize the recognition behavior by modifying these constants at the top of `scripts/recognize.py`:

| Setting | Default | Description |
| :--- | :--- | :--- |
| `SIMILARITY_THRESHOLD` | `0.51` | Minimum cosine similarity required to match identity. Lower for more leniency, higher for stricter verification. |
| `TOP_K` | `1` | Number of closest vectors retrieved from Pinecone. |
| `query_interval` | `1.0` | How often (in seconds) the background thread queries Pinecone. |
| `cv2.VideoCapture(0)` | `0` | Camera index. Change to `1` if you use an external USB webcam. |

---

## 🏛 Full System Architecture (FastAPI + PostgreSQL)

*This is the extended backend architecture for enterprise attendance tracking. A Next.js frontend is planned for future releases.*

### System Flow
```mermaid
graph TD
    A[Employee Approaches Camera] --> B[Face Detection & Tracking]
    B --> C[Generate Facial Embedding]
    C --> D[Pinecone Vector Search]
    D -- Match Found --> E[Identity Verified]
    D -- No Match --> F[Access Denied / Unknown]
    E --> G{Check Today's Logs in Postgres}
    G -- No Active Record --> H[Log Check-in]
    G -- Active Record Found --> I[Log Check-out]
    H --> J[Status: Present]
    I --> K[Session Closed]
```

### Relational Database Schema (PostgreSQL)

- **`employees`**: Stores employee profiles (`id`, `name`, `employee_code`, `department_id`, `face_vector_id`, `is_active`).
- **`attendance`**: Tracks automated entry and exit sessions (`id`, `employee_id`, `date`, `check_in`, `check_out`, `status`).
- **`departments`**: Company department organization.
- **`employee_images`**: Historical facial capture records.

### Running the Full FastAPI Backend (Optional)
If you have PostgreSQL installed and configured in `.env`:
```powershell
cd Backend
.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8001
```
- Interactive API Documentation: [http://localhost:8001/docs](http://localhost:8001/docs)
- Endpoints:
  - `POST /recognize`: Send an image to detect identity.
  - `POST /attendance`: Send an image to auto-toggle Check-in / Check-out.
  - `GET /employees`: View employee list and current presence status.

---

## 📦 Technology Stack

- **Face Detection & Alignment**: MTCNN (`facenet-pytorch`)
- **Facial Embeddings**: InceptionResnetV1 pretrained on VGGFace2 (512 dimensions)
- **Vector Database**: Pinecone (Cosine Similarity Metric)
- **Computer Vision & UI**: OpenCV
- **Backend API (Optional)**: FastAPI & Uvicorn
- **Relational Database (Optional)**: PostgreSQL & SQLAlchemy
