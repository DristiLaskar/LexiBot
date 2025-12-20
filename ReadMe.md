![Legaicy Banner](./assets/banner.png)

# 👩‍⚖️ Legaicy

## Your AI-Powered Legal Co-Pilot

We empower users by demystifying complex legal documents and procedures.

### 🔎 Smart Contract Vulnerability Analysis
Our system meticulously scans smart legal contracts to pinpoint potential loopholes and security risks, helping you prevent costly errors and disputes.

### 🤖 Intelligent LegalBOT
Have a legal question? Get instant answers from our AI-powered LegalBOT. It understands the nuances of your query and provides responses based on the legal framework of your country, making the advice relevant and actionable. Currently available for India, USA and Germany.

### 🚀 Powered by Cutting-Edge AI
Our services are built on a foundation of a self fine-tuned Gemma3 model and a sophisticated multi-agent RAG architecture with FAISS, ensuring high accuracy and contextual understanding.

---

## 👥 Team Members
- Angshuman Patar
- Suman Dutta
- Satyam Das
- Siman Jyoti Nath

---

## ⚙️ Installation & Setup

#### 1. Clone the repository:
   - ```bash
     git clone https://github.com/suman-dutta2913/legaicy.git
     ```
#### 2. Navigate into the project directory:
   - ```bash
     cd legaicy
     ```
#### 3. Create your virtual environment:
   - ```bash
     conda create -p venv
     ```
#### 4. Activate your virtual environment:
   - ```bash
     conda activate venv/
     ```
#### 5. Install dependencies:
   - ```bash
     pip install -r requirements.txt
     ```
#### 6. Ensure CUDA toolkit is installed:
   - ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
     ```

#### 7. Start the FastAPI server:
   - ```bash
     uvicorn server:app --reload
     ```
## ⚙️ Run React application (Separate Terminal)
#### 1. Activate your virtual environment:
   - ```bash
     conda activate venv/
     ```
#### 2. Navigate into the Frontend directory:
   - ```bash
     cd Frontend
     ```
#### 3. Start the React application:
   - ```bash
     npm run dev
     ```
## ⚙️ System Requirements
- CPU: 4-core Processor (Intel Core i5, AMD Ryzen 5 or equivalent)
- RAM: 16 GB
- GPU: NVIDIA GPU with CUDA support and at least 6 GB of VRAM (e.g., NVIDIA GeForce RTX 4050, GTX 1660 Ti).
- Storage: 10 GB of free space (SSD recommended for faster model loading).
- Software:
- NVIDIA Drivers and CUDA Toolkit 11.8 (as specified in installation).
- Git, Conda, and Node.js (v18 or higher).

