# FaceSwap Architecture & Technical Details

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FACESWAP SYSTEM                            │
└─────────────────────────────────────────────────────────────────┘

STAGE 1: DATA PREPARATION
┌──────────────────────────────────────────────────────────────┐
│ Input: Video files (source & target person)                 │
│ Process: face_detect.py (MTCNN-based extraction)            │
│ Output: Aligned face images (128x128 PNG files)             │
└──────────────────────────────────────────────────────────────┘
                            ↓
STAGE 2: MODEL TRAINING  
┌──────────────────────────────────────────────────────────────┐
│ Input: Two folders of face images                           │
│ Architecture:                                               │
│   ├─ Shared Encoder (learns common features)               │
│   ├─ Face-A Decoder (reconstructs Face A)                  │
│   └─ Face-B Decoder (reconstructs Face B)                  │
│                                                             │
│ Loss Functions:                                             │
│   • Reconstruction Loss (L1)                               │
│   • SSIM Loss (perceptual quality)                         │
│   • Adversarial Loss (face realism)                        │
│                                                             │
│ Training: train.py (10,000+ iterations)                    │
│ Output: Trained model weights (.pt file)                   │
└──────────────────────────────────────────────────────────────┘
                            ↓
STAGE 3: INFERENCE (FACE SWAPPING)
┌──────────────────────────────────────────────────────────────┐
│ Input: Original video                                       │
│ Process: video_writer.py (frame-by-frame)                  │
│   1. Extract face from frame (MTCNN)                        │
│   2. Encode face using shared encoder                       │
│   3. Decode using opposite person's decoder                │
│   4. Blend using Delaunay triangulation (faceswap.py)      │
│   5. Write back to video                                    │
│                                                             │
│ Output: Video with swapped faces                           │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Face Detection & Extraction (`face_detect.py`)
**Purpose**: Extract face regions from video frames

**Key Algorithm**: MTCNN (Multi-task Cascaded Convolutional Networks)
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device, margin=50, 
              select_largest=True, image_size=256)
```

**Key Features**:
- GPU-accelerated detection
- Returns bounding boxes and landmarks
- Margin adjustment for context
- Square cropping for uniform training

**Key Function**:
```
extract_face(frame, align=True, margin=5)
  → Detects face in frame
  → Creates square bounding box around face
  → Returns cropped PIL Image (128×128)
```

### 2. Face Alignment & Rotation (`img_rotate.py`)
**Purpose**: Normalize face orientation for consistent training

**Key Algorithm**: dlib facial landmarks + geometric transformation
```python
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
```

**Process**:
1. Detect face using dlib (68 landmarks)
2. Calculate eye center and angle between eyes
3. Apply rotation + scale transform
4. Translate to center face in output image
5. Handle boundary issues with replicate padding

**Why Important**: Consistent face orientation improves model convergence during training

### 3. Neural Network Architecture (`model.py`)

#### AutoEncoder
```
Input (3×128×128) 
  ↓
ENCODER (Shared):
  Downscale(3→64) → Downscale(64→128) → ... → Downscale(512→512)
  Output: 128-dim latent space
  ↓
DECODER A or B (Person-specific):
  Upscale(128→512) → ... → Upscale(64→3)
  Output: (3×128×128)
```

**Key Features**:
- **Shared Encoder**: Learns common facial features (both people have eyes, nose, mouth)
- **Separate Decoders**: Learn person-specific details (unique face structure, skin texture)
- **Bottleneck**: 128-dimensional latent space acts as identity filter
- **Skip Connections**: Preserve fine details through upsampling

**Architecture Details**:
```python
class Downscale(nn.Module):
    # Conv2d + BatchNorm + LeakyReLU

class Upscale(nn.Module):
    # ConvTranspose2d + BatchNorm + ReLU

class ResBlock(nn.Module):
    # Residual connections for deep networks
```

#### Discriminator
```
Input (3×128×128)
  ↓
Conv2d(3→128, k=5, s=2) + LeakyReLU
Conv2d(128→64, k=5, s=2) + LeakyReLU
... (5 blocks)
Sigmoid (binary classification)
```

**Purpose**: Forces generated faces to look realistic (adversarial loss)

### 4. Face Geometry Manipulation (`faceswap.py`)
**Purpose**: Blend swapped face naturally into original frame

**Key Technique**: Delaunay Triangulation
```
68 Facial Landmarks (from dlib)
  ↓
Delaunay Triangulation (creates triangles connecting nearby landmarks)
  ↓
Affine Transform (apply per-triangle)
  ↓
Smooth Blending (handle boundaries)
```

**Key Functions**:
- `calculateDelaunayTriangles()`: Creates mesh from landmarks
- `applyAffineTransform()`: Warps swapped face to match frame geometry
- `rectContains()`: Validates triangle vertices
- `warpTriangle()`: Applies geometric transformation per triangle

**Why Delaunay?**: 
- Avoids long, thin triangles that distort during warping
- Creates natural-looking blending at face boundaries
- Handles face shape differences between two people

### 5. Loss Functions (`SSIM.py`)
**Purpose**: Measures quality of reconstructed faces

**SSIM (Structural Similarity Index)**:
```
SSIM = (2μ₁μ₂ + C₁)(2σ₁₂ + C₂) / ((μ₁² + μ₂² + C₁)(σ₁² + σ₂² + C₂))
```

**Why SSIM over MSE?**
- MSE: Pixel-level Euclidean distance (not perceptually accurate)
- SSIM: Considers luminance, contrast, and structure (human perception)
- Better prevents blurry outputs

**Usage**: Combined with L1 reconstruction loss and adversarial loss

### 6. Data Pipeline (`Iter.py`, `train.py`)
**Data Loading**:
```python
transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])
```

**Batch Processing**:
- Loads entire dataset into memory (NumPy arrays)
- Random sampling during training
- GPU transfer for accelerated processing

### 7. Training Loop (`train.py`)
**Algorithm**:
```
for each iteration:
  1. Sample batch from Face A and Face B
  2. Encode both: z_a = encoder(img_a), z_b = encoder(img_b)
  3. Decode swapped: recon_a = decoder_a(z_b), recon_b = decoder_b(z_a)
  4. Calculate losses:
     L_recon = L1_loss + SSIM_loss
     L_adv = discriminator_loss (if enabled)
     L_total = L_recon + λ*L_adv
  5. Backprop and update weights
  6. Save model every N iterations
```

**Hyperparameters**:
- Batch size: 1 (can be increased for larger memory)
- Iterations: 10,000+ for convergence
- Learning rate: Default Adam optimizer (0.0002 typically)
- Lambda: Weighting factor for adversarial loss

### 8. Inference Pipeline (`video_writer.py`)
**Frame Processing**:
```
For each frame in video:
  1. Extract face using MTCNN
  2. Resize to 128×128
  3. Convert BGR→RGB and normalize [0,255]→[0,1]
  4. Create tensor (1×3×128×128)
  5. Encode with shared encoder
  6. Decode with target decoder
  7. Denormalize and convert back to BGR
  8. Blend into original frame using Delaunay (faceswap.py)
  9. Write frame to output video
```

**Output**:
- Video codec: mp4v
- Resolution: Same as input
- FPS: Same as input
- Frame-by-frame consistency maintained

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Deep Learning | PyTorch 1.3.1 | Model training & inference |
| GPU Acceleration | CUDA 9.2 | 10-100x faster training |
| Face Detection | MTCNN (facenet_pytorch 2.4.1) | Real-time face detection |
| Landmarks | dlib 19.21.0 | 68-point facial landmarks |
| Computer Vision | OpenCV 4.4.0 | Image processing, video I/O |
| Image Similarity | scikit-image 0.15.0 | SSIM computation |
| Array Operations | NumPy 1.16.5 | Tensor operations |
| Image I/O | Pillow 7.2.0 | Image loading/saving |

## Why This Architecture?

✓ **Autoencoders**: Unsupervised learning of identity-independent features  
✓ **Shared Encoder**: Reduces parameters, forces feature sharing  
✓ **Separate Decoders**: Captures person-specific details  
✓ **Adversarial Loss**: Ensures realistic outputs  
✓ **Delaunay Triangulation**: Handles face shape differences geometrically  
✓ **SSIM Loss**: Perceptually-aligned quality metric  
✓ **GPU Acceleration**: Makes training feasible (hours vs. weeks)  

---

**Next**: See [WORKFLOW.md](WORKFLOW.md) for step-by-step pipeline explanation.
