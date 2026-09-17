# FaceSwap Workflow - Step-by-Step Pipeline

## Complete End-to-End Workflow

This document explains the exact sequence of operations from raw video to swapped output.

---

## PHASE 1: DATA PREPARATION

### Step 1.1: Video Input
**Input**: Two video files
- `video_source.mp4` - Contains source person (person A)
- `video_target.mp4` - Contains target person (person B)

**Requirement**: At least 100-200 frames per video for training

### Step 1.2: Face Extraction Using MTCNN
**File**: `face_detect.py`
**Command**:
```bash
python face_detect.py -video_src video_source.mp4 -out_name face_a -align_faces 1
python face_detect.py -video_src video_target.mp4 -out_name face_b -align_faces 1
```

**Process Flow**:
```
Each frame in video:
  ↓
Load frame with cv2.VideoCapture()
  ↓
Convert PIL.Image → RGB format
  ↓
Run MTCNN detection:
  - Input: RGB image
  - Output: Bounding boxes for all faces detected
  ↓
For each detected face:
  - Extract coordinates: (x1, y1, x2, y2)
  - Add margin (default: 5 pixels)
  - Create square bounding box:
    * y1 += margin, y2 -= margin
    * Calculate diff = y2 - y1
    * Center x coordinate: mid_x = (x1 + x2) / 2
    * Square box: x1 = mid_x - diff/2, x2 = mid_x + diff/2
  ↓
Crop image to bounding box
  ↓
Save as PNG: face_a/0000.png, face_a/0001.png, ...
```

**Output Structure**:
```
face_a/
  ├── 0000.png (128×128)
  ├── 0001.png
  ├── 0002.png
  └── ... (100-500 images)

face_b/
  ├── 0000.png (128×128)
  ├── 0001.png
  ├── 0002.png
  └── ... (100-500 images)
```

**Key Parameters**:
- `margin`: Padding around face (recommended: 5-10 pixels)
- `align_faces`: Apply dlib rotation normalization (1=yes, 0=no)
- Output format: 128×128 PNG (standard for training)

---

## PHASE 2: FACE ALIGNMENT (Optional but Recommended)

### Step 2.1: Normalize Face Orientation
**File**: `img_rotate.py`
**Purpose**: Ensure consistent upright face orientation

**Process Flow**:
```
For each extracted face image:
  ↓
Load image with dlib
  ↓
Detect 68 facial landmarks using dlib predictor
  ↓
Calculate rotation angle:
  - Direction vector: d = (right_eye - left_eye) / distance
  - Angle: a = atan2(d.y, d.x) in degrees
  ↓
Calculate scale factor:
  - Eye distance: eyes_d = ||right_eye - left_eye||
  - Face size: face_size_x = eyes_d * 2
  - Scale: scale_factor = output_size / (face_size_x * 2)
  ↓
Build transformation matrix:
  - Rotation around eye center
  - Scale to fit in output (256×256)
  - Translation to center face
  ↓
Apply warpAffine with boundary replicate mode
  ↓
Output: Normalized 256×256 face image
```

**Visualization**:
```
Original frame:        After alignment:
  ╱─────╲                    ┌─────┐
 ╱ face  ╲  (tilted)   →    │face │ (upright)
╱─────────╲                  └─────┘
```

---

## PHASE 3: MODEL TRAINING

### Step 3.1: Dataset Loading
**File**: `train.py`
**Command**:
```bash
python train.py -face_a_dir face_a -face_b_dir face_b \
                -batch_size 1 -n_steps 10000 \
                -save_iter 1000 -model_name faceswap_model
```

**Dataset Preparation**:
```python
# Load datasets with transformations
dataset_a = datasets.ImageFolder(root='face_a', transform=Compose([
    transforms.Resize((128, 128)),      # Standardize size
    transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
    transforms.ToTensor(),              # Convert to [0,1] range
]))

dataset_b = datasets.ImageFolder(root='face_b', transform=...)

# Load entire dataset into memory
train_dataset_array_a = next(iter(dataloader_a))[0].numpy()  # Shape: (N, 3, 128, 128)
train_dataset_array_b = next(iter(dataloader_b))[0].numpy()

np.save('a.npy', train_dataset_array_a)  # Cache for faster loading
np.save('b.npy', train_dataset_array_b)
```

### Step 3.2: Model Architecture Initialization
**File**: `model.py`

**Architecture**:
```python
class AutoEncoder(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(
            Downscale(3, 64),      # 128 → 64
            Downscale(64, 128),    # 64 → 32
            Downscale(128, 256),   # 32 → 16
            Downscale(256, 256),   # 16 → 8
            Downscale(256, 512),   # 8 → 4
            Downscale(512, 512),   # 4 → 2
            # Output: (batch, 512, 2, 2) → flattened to 128-dim latent
        )
        
        self.decoder_a = nn.Sequential(
            Upscale(512, 512),     # 2 → 4
            Upscale(512, 256),     # 4 → 8
            ...
            Upscale(64, 3),        # 64 → 128
        )
        
        self.decoder_b = nn.Sequential(...)  # Similar structure
```

**Downscale Block**:
```
Input (C_in, H, W)
  ↓ Conv2d(kernel=4, stride=2, padding=1)
  ↓ BatchNorm2d
  ↓ LeakyReLU(0.2)
Output (C_out, H/2, W/2)
```

**Upscale Block**:
```
Input (C_in, H, W)
  ↓ ConvTranspose2d(kernel=4, stride=2, padding=1)
  ↓ BatchNorm2d
  ↓ ReLU
Output (C_out, H*2, W*2)
```

### Step 3.3: Training Loop
**File**: `train.py`
**Duration**: 10,000-100,000 iterations (2-48 hours on GPU)

**Each Training Iteration**:
```python
for iteration in range(n_steps):
    # 1. Sample random batch
    idx_a = random.randint(0, len(train_dataset_array_a)-1)
    idx_b = random.randint(0, len(train_dataset_array_b)-1)
    
    img_a = train_dataset_array_a[idx_a]  # (3, 128, 128)
    img_b = train_dataset_array_b[idx_b]  # (3, 128, 128)
    
    # Convert to tensor and move to GPU
    img_a = Variable(torch.from_numpy(img_a).float()).to(device)
    img_b = Variable(torch.from_numpy(img_b).float()).to(device)
    
    # 2. FORWARD PASS - Swapped reconstruction
    # Encode both images using shared encoder
    latent_a = model.encoder(img_a)  # (1, 128)
    latent_b = model.encoder(img_b)  # (1, 128)
    
    # Cross-decode (swap)
    recon_a_from_b = model.decoder_a(latent_b)  # Face A features decoded as A
    recon_b_from_a = model.decoder_b(latent_a)  # Face B features decoded as B
    
    # Self-reconstruction (reconstruction)
    recon_a = model.decoder_a(latent_a)  # Face A → Face A
    recon_b = model.decoder_b(latent_b)  # Face B → Face B
    
    # 3. LOSS CALCULATION
    # Reconstruction loss
    loss_recon_a = L1Loss(recon_a, img_a) + SSIM_loss(recon_a, img_a)
    loss_recon_b = L1Loss(recon_b, img_b) + SSIM_loss(recon_b, img_b)
    
    # Cross-reconstruction loss
    loss_cross_a = L1Loss(recon_a_from_b, img_a)
    loss_cross_b = L1Loss(recon_b_from_a, img_b)
    
    # Adversarial loss (if enabled)
    if use_discriminator:
        validity_a = discriminator(recon_a)
        validity_b = discriminator(recon_b)
        loss_adv = BCE_loss(validity_a, ones) + BCE_loss(validity_b, ones)
    
    # Total loss
    total_loss = loss_recon_a + loss_recon_b + loss_cross_a + loss_cross_b + λ*loss_adv
    
    # 4. BACKWARD PASS
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # 5. LOGGING & SAVING
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: Loss = {total_loss.item():.4f}")
    
    if iteration % save_iter == 0:
        torch.save(model.state_dict(), f'saved_models/model_iter_{iteration}.pt')
```

**Convergence Indicators**:
```
Iteration 0:     Loss ≈ 0.5-1.0 (random initialization)
Iteration 1000:  Loss ≈ 0.2-0.4 (noticeable improvement)
Iteration 5000:  Loss ≈ 0.05-0.15 (good convergence)
Iteration 10000: Loss ≈ 0.01-0.05 (plateau reached)
```

---

## PHASE 4: INFERENCE & VIDEO RECONSTRUCTION

### Step 4.1: Load Trained Model
**File**: `video_writer.py`
**Command**:
```bash
python video_writer.py -original_video original.mp4 \
                       -model_location faceswap_model.pt \
                       -decoder b \
                       -out_name output_video
```

**Model Loading**:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoEncoder(image_channels=3).to(device)
model.load_state_dict(torch.load('faceswap_model.pt'))
model.eval()  # Set to evaluation mode (disable dropout/batch norm training)

# Video setup
cap = cv2.VideoCapture('original.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
```

### Step 4.2: Frame-by-Frame Processing

**For each frame in video**:
```python
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. EXTRACT FACE
    img_face = extract_face(frame, align=True, margin=5)
    # Returns: PIL Image (128×128) of detected face
    
    # 2. PREPROCESS FOR MODEL
    img_face_array = np.array(img_face)  # Convert PIL to NumPy
    img_face_array = cv2.resize(img_face_array, (128, 128))  # Ensure size
    
    # Convert color space and format
    img_face_rgb = cv2.cvtColor(img_face_array, cv2.COLOR_BGR2RGB)
    img_tensor = img_face_rgb[:, :, ::-1].transpose((2, 0, 1)).copy()  # CHW format
    img_tensor = torch.from_numpy(img_tensor).float().div(255).unsqueeze(0)  # Normalize to [0,1]
    # Shape: (1, 3, 128, 128)
    
    # 3. INFERENCE
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():  # No gradients needed for inference
        latent = model.encoder(img_tensor)
        
        # Decode with target decoder (e.g., decoder_b for Face B)
        swapped_face = model.decoder_b(latent)  # (1, 3, 128, 128)
    
    # 4. POSTPROCESS
    swapped_face = swapped_face.cpu().detach().numpy()
    swapped_face = swapped_face[0]  # Remove batch dimension: (3, 128, 128)
    swapped_face = (swapped_face * 255).astype(np.uint8)  # [0,1] → [0,255]
    swapped_face = swapped_face.transpose((1, 2, 0))  # CHW → HWC
    swapped_face = cv2.cvtColor(swapped_face, cv2.COLOR_RGB2BGR)  # RGB → BGR
    
    # 5. BLEND INTO ORIGINAL FRAME
    blended_frame = swap_faces(frame, swapped_face, img_face_landmarks)
    # Uses Delaunay triangulation to blend face seamlessly
    
    # 6. WRITE TO OUTPUT VIDEO
    video_out.write(blended_frame)
```

### Step 4.3: Face Blending with Delaunay Triangulation
**File**: `faceswap.py`

**Blending Process**:
```
Swapped face (128×128)         Original frame (1920×1080)
        +                                 +
        |                                 |
        v                                 v
Calculate 68 landmarks          Find corresponding face region
from swapped face               with original landmarks
        |                                 |
        +────────────┬────────────────────+
                     |
                     v
        Create Delaunay triangles
        connecting 68 landmarks
                     |
                     v
    For each triangle:
    1. Get coordinates in swapped face
    2. Get coordinates in original frame
    3. Calculate affine transformation
    4. Warp swapped triangle to match frame geometry
    5. Blend smoothly with original background
                     |
                     v
        Output: Seamlessly blended frame
```

**Key Code**:
```python
# Calculate Delaunay triangulation
triangles = calculateDelaunayTriangles(rect, landmarks)

# For each triangle
for triangle in triangles:
    # Get corner points
    pt1, pt2, pt3 = triangle
    
    # Find affine transform from swapped face space to frame space
    warpMat = cv2.getAffineTransform(srcTri, dstTri)
    
    # Warp triangle
    warpedTri = cv2.warpAffine(swapped_triangle, warpMat, size)
    
    # Blend with original
    blended_output[region] = warpedTri + original_frame[region] * (1-alpha)
```

**Why Delaunay?**
- ✓ Avoids long, thin triangles
- ✓ Maintains local geometric consistency
- ✓ Creates smooth boundaries
- ✓ Handles face shape differences

---

## PHASE 5: OUTPUT VIDEO

**Final Output**:
```
output_video.mp4
├── Codec: mp4v (H.264)
├── Resolution: Original frame size (e.g., 1920×1080)
├── Frame Rate: Original FPS (e.g., 30 fps)
├── Duration: Same as original video
└── Face: Swapped seamlessly frame-by-frame
```

**Result**: Original person → Person with Face B's features and appearance

---

## Timeline & Performance

| Phase | Duration | Key Metric |
|-------|----------|-----------|
| Face Extraction | 1-5 min | ~5-10 fps per video |
| Face Alignment | 1-5 min | Optional preprocessing |
| Model Training | 2-24 hours | Depends on GPU; 10K-50K iterations |
| Video Inference | 5-30 min | ~1-5 fps per video |
| **Total** | **3-30 hours** | Parallelizable except training |

**GPU Acceleration Impact**:
- Training on CPU: 1 week+ (impractical)
- Training on GPU (NVIDIA): 2-24 hours
- Speedup: 50-100x

---

**Next**: See [SETUP_AND_USAGE.md](SETUP_AND_USAGE.md) for detailed commands and setup instructions.
