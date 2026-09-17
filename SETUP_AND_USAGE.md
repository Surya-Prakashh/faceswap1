# FaceSwap - Setup and Usage Guide

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
  - Alternative: CPU mode (very slow, ~50x slower)
- **CPU**: Multi-core processor (4+ cores)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ for video files and models
- **OS**: Windows, Linux, or macOS

### Software Requirements
- Python 3.6 or later
- CUDA 9.2+ (for GPU acceleration)
- cuDNN 7.0+ (NVIDIA deep learning library)

---

## Installation

### Step 1: Clone or Set Up Project
```bash
cd d:/Training/faceswap1
cd DeepFakeTorch
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies**:
```
torch==1.3.1+cu92          # PyTorch with CUDA 9.2
torchvision==0.4.2+cu92    # Vision models
opencv-python==4.4.0.42    # Computer vision
facenet-pytorch==2.4.1     # MTCNN for face detection
dlib==19.21.0              # Facial landmarks (68-point)
numpy==1.16.5              # Array operations
scikit-image==0.15.0       # Image processing
Pillow==7.2.0              # Image I/O
```

### Step 4: Download Required Models
```bash
# Download dlib facial landmark model
# Place in: DeepFakeTorch/shape_predictor_68_face_landmarks.dat
# Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

---

## Usage Workflow

### PHASE 1: Prepare Input Videos

Place your video files:
```
DeepFakeTorch/
├── video_person_a.mp4    (Person A - source)
└── video_person_b.mp4    (Person B - target)
```

**Video Requirements**:
- Format: MP4, AVI, or MOV
- Duration: 10+ seconds (≈ 100-300 frames minimum)
- Resolution: 720p+ recommended
- FPS: 24-60 fps
- Content: Clear face shots, good lighting

---

### PHASE 2: Extract Faces from Videos

#### Command for Person A:
```bash
python face_detect.py -video_src video_person_a.mp4 -out_name face_a -align_faces 1
```

**Parameters**:
- `-video_src`: Path to input video
- `-out_name`: Output directory (creates `face_a/` folder)
- `-align_faces`: 1 = align faces (recommended), 0 = no alignment
- `-margin`: Pixels around detected face (default: 5)

#### Command for Person B:
```bash
python face_detect.py -video_src video_person_b.mp4 -out_name face_b -align_faces 1
```

**Expected Output**:
```
face_a/
├── 0000.png (128×128)
├── 0001.png
├── 0002.png
└── ... (hundreds of aligned face images)

face_b/
├── 0000.png
├── 0001.png
└── ...
```

**Verification**:
```bash
# Check number of extracted faces
dir /s /b face_a | find /c ".png"  # Windows
ls face_a | wc -l                  # macOS/Linux
```

Expected: 100-500 faces per video

---

### PHASE 3: Train the Model

#### Basic Training Command:
```bash
python train.py -face_a_dir face_a -face_b_dir face_b \
                -batch_size 1 -n_steps 10000 \
                -save_iter 1000 -model_name faceswap_model
```

#### Advanced Training Command:
```bash
python train.py -face_a_dir face_a \
                -face_b_dir face_b \
                -saved_dir saved_models \
                -batch_size 1 \
                -n_steps 50000 \
                -save_iter 2000 \
                -discriminator True \
                -model_name faceswap_50k
```

**Parameters**:
```
-face_a_dir DIR           Directory with Face A images (default: face_a)
-face_b_dir DIR           Directory with Face B images (default: face_b)
-saved_dir DIR            Where to save model checkpoints (default: saved_models)
-batch_size INT           Batch size (default: 1; increase if GPU memory allows)
-n_steps INT              Total training iterations (default: 10000)
                          → 10K: 1-2 hours, 50K: 5-10 hours
-save_iter INT            Save model every N iterations (default: 1000)
-discriminator BOOL       Use adversarial loss (default: False)
                          → Improves quality but takes longer
-model_name STR           Model filename (saved as saved_models/model_name.pt)
```

**Training Output**:
```
loading data...
Iteration 0:     Loss = 0.8234
Iteration 100:   Loss = 0.4521
Iteration 200:   Loss = 0.3847
...
Iteration 10000: Loss = 0.0521

saved_models/
├── faceswap_model_1000.pt
├── faceswap_model_2000.pt
├── faceswap_model_3000.pt
└── faceswap_model_10000.pt  ← Use this final model
```

**Training Tips**:
- **Start with**: 5,000-10,000 iterations for testing
- **For production**: 20,000-50,000 iterations
- **GPU out of memory?** Reduce `-batch_size` or skip discriminator
- **Too slow?** Reduce `-n_steps` or use fewer frames in face_a/face_b

**Loss Convergence**:
```
Good convergence:  Loss drops quickly in first 1000 iter, stabilizes by 5000 iter
Poor convergence:  Loss stays high or increases → check data quality
Divergence:        Loss becomes NaN or inf → reduce learning rate
```

---

### PHASE 4: Perform Face Swap on Video

#### Basic Inference Command:
```bash
python video_writer.py -original_video video_person_a.mp4 \
                       -model_location saved_models/faceswap_model.pt \
                       -decoder b \
                       -out_name output_swapped
```

**Parameters**:
```
-original_video PATH      Input video to process
-model_location PATH      Trained model (.pt file)
-decoder CHAR             Which decoder to use: 'a' or 'b'
                          → 'a' = decode with Face A features
                          → 'b' = decode with Face B features
-out_name STR             Output video filename (without extension)
```

**Decoder Choice**:
```
If training with:
  -face_a_dir = video_person_a.mp4
  -face_b_dir = video_person_b.mp4

Then:
  decoder='a' → Person A's face stays mostly the same
  decoder='b' → Person A's face becomes Person B's face ← Most common use
```

**Processing**:
```
Tracking frame: 1
Tracking frame: 2
...
Tracking frame: 300
Video saved to: output_swapped.mp4
```

**Output Video**:
```
output_swapped.mp4
├── Same resolution as input
├── Same FPS as input
├── Same duration as input
└── Faces swapped frame-by-frame
```

**Expected Duration**:
- 10 minutes of video: ~5-30 minutes processing on GPU
- Speed: ~1-5 fps depending on GPU

---

## Complete Example Workflow

### Example: Swap faces between Alice.mp4 and Bob.mp4

```bash
# Step 1: Extract faces
python face_detect.py -video_src Alice.mp4 -out_name face_alice -align_faces 1
python face_detect.py -video_src Bob.mp4 -out_name face_bob -align_faces 1

# Step 2: Train model
python train.py -face_a_dir face_alice -face_b_dir face_bob \
                -batch_size 1 -n_steps 10000 -save_iter 1000 \
                -model_name alice_bob_model

# Step 3: Swap Alice's face with Bob's face
python video_writer.py -original_video Alice.mp4 \
                       -model_location saved_models/alice_bob_model.pt \
                       -decoder b \
                       -out_name Alice_with_Bob_face

# Step 4 (Optional): Swap Bob's face with Alice's face
python video_writer.py -original_video Bob.mp4 \
                       -model_location saved_models/alice_bob_model.pt \
                       -decoder a \
                       -out_name Bob_with_Alice_face
```

**Result**:
```
Alice_with_Bob_face.mp4    ← Alice with Bob's face
Bob_with_Alice_face.mp4    ← Bob with Alice's face
```

---

## Troubleshooting

### Issue: "No faces detected in frame"
**Cause**: Video quality too low or face too small
**Solution**:
- Use higher resolution video (720p+)
- Ensure good lighting
- Increase `-margin` parameter in face_detect.py

### Issue: "CUDA out of memory"
**Cause**: GPU doesn't have enough VRAM
**Solutions**:
- Reduce `-batch_size` to 1 (already minimal)
- Use CPU mode (slow): Set `device = torch.device('cpu')`
- Reduce image resolution in `train.py` (change 128 to 96)

### Issue: Model training is very slow
**Cause**: Running on CPU instead of GPU
**Solution**:
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- If False, install CUDA and cuDNN
- Force GPU: Add `device = torch.device('cuda:0')` at top of train.py

### Issue: Output video has artifacts/misaligned faces
**Cause**: Insufficient training or poor face quality
**Solutions**:
- Train for more iterations (20K-50K)
- Enable discriminator: `-discriminator True`
- Use clearer video footage
- Increase margin: `-margin 10` in face_detect.py

### Issue: Output video is corrupted or won't play
**Cause**: Codec issue
**Solution**:
- Re-encode output: 
```bash
ffmpeg -i output_swapped.mp4 -c:v libx264 -c:a aac output_final.mp4
```

---

## Performance Benchmarks

### On NVIDIA RTX 2080 (11GB VRAM):

| Phase | Duration | Notes |
|-------|----------|-------|
| Face Extraction (300 frames) | 2-3 min | MTCNN detection ~5-10 fps |
| Training (10K iterations) | 1-2 hours | Batch size 1 |
| Training (50K iterations) | 5-10 hours | With adversarial loss |
| Inference (300 frames) | 5-10 min | Output video generation |
| **Total** | **7-20 hours** | For complete pipeline |

### On NVIDIA GTX 1080 Ti (11GB VRAM):
- Training: ~2-3x slower than RTX 2080
- Inference: Similar speed

### On CPU Only (Intel i7, 16GB RAM):
- Training: 50x-100x slower than GPU
- **Not recommended**: Use GPU if possible

---

## Best Practices

✓ **Data Quality**
  - Use 10+ second videos with clear face shots
  - Maintain consistent lighting and angles
  - Avoid extreme head rotations
  - Remove glasses/sunglasses for better alignment

✓ **Training**
  - Start with 5K-10K iterations for testing
  - Monitor loss curve (should decrease smoothly)
  - Save multiple checkpoints to find best model
  - Use discriminator for higher quality results

✓ **Inference**
  - Test on short video first (10 seconds)
  - Adjust margin/alignment if faces look incorrect
  - Process entire video once model is ready
  - Post-process if needed (color correction, smoothing)

✓ **Resource Management**
  - Close other GPU-hungry applications
  - Monitor GPU memory: `nvidia-smi -l 1` (Linux/Windows)
  - Use GPU profiling to optimize batch size
  - Checkpoints allow resuming training if interrupted

---

## Advanced Usage

### Resume Training from Checkpoint
```python
# In train.py, add before training loop:
if os.path.exists('saved_models/faceswap_model_5000.pt'):
    model.load_state_dict(torch.load('saved_models/faceswap_model_5000.pt'))
    start_iter = 5000  # Continue from iteration 5000
```

### Use Different Decoders
```bash
# Swap with Face A decoder
python video_writer.py -original_video video.mp4 \
                       -model_location model.pt \
                       -decoder a -out_name output_a

# Swap with Face B decoder
python video_writer.py -original_video video.mp4 \
                       -model_location model.pt \
                       -decoder b -out_name output_b
```

### Combine Multiple Models
Train multiple models with different data and ensemble predictions:
```python
models = [load_model('model1.pt'), load_model('model2.pt')]
for model in models:
    pred = model(input_frame)
    # Average predictions for smoother output
```

---

## Next Steps

See the project documentation:
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Overview for interviews
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical deep dive
- [WORKFLOW.md](WORKFLOW.md) - Detailed pipeline explanation

---

**Questions?** Check comments in source files or README.md
