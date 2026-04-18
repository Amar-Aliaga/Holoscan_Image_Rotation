# V4L2 Camera Rotation with HoloViz Display

This is a Holoscan application that demonstrates:
1. **V4L2 Camera Capture** - Captures frames from a V4L2 camera device (e.g., `/dev/video0`)
2. **Frame Rotation** - Rotates captured frames 90 degrees using CUDA kernels on the GPU
3. **HoloViz Visualization** - Displays the rotated frames in real-time using HoloViz

## Architecture

The application follows a pipeline architecture:

```
V4L2 Camera → Format Converter (YUYV→RGBA) → Rotation Operator → HoloViz Display
```

### Components

- **V4L2VideoCaptureOp**: Captures video frames from `/dev/video0` at 640x480 resolution
- **FormatConverterOp**: Converts YUYV format to RGBA for GPU processing
- **RotationOperator**: Custom CUDA-based operator that rotates frames 90 degrees
- **HolovizOp**: Displays the processed frames in a window

## Requirements

- Holoscan SDK (with V4L2 support)
- CUDA Toolkit (for GPU-accelerated rotation)
- A V4L2-compatible camera (built-in laptop camera, USB webcam, etc.)

## Building

From the `test` directory:

```bash
cd build
cmake ..
make
```

## Running

### Check your camera device

First, verify your camera is available:

```bash
ls -l /dev/video*
```

### Run the application

```bash
./holoscan_app
```

The application will:
1. Open `/dev/video0` (adjust in `main.cpp` if your camera is different)
2. Capture frames at 30 FPS
3. Rotate each frame 90 degrees clockwise on the GPU
4. Display the rotated frames in a HoloViz window

## Camera Setup

If your camera is not at `/dev/video0`, you can:

1. Find your camera:
   ```bash
   v4l2-ctl --list-devices
   ```

2. Edit `main.cpp` and change:
   ```cpp
   Arg("device") = std::string("/dev/video0"),  // Change to your device
   ```

## Resolution and Framerate

To modify the resolution or framerate, edit these lines in `main.cpp`:

```cpp
Arg("width") = 640U,      // Change width
Arg("height") = 480U,     // Change height
Arg("framerate") = 30U,   // Change FPS
```

## File Structure

```
test/
├── main.cpp                    # Application entry point
├── rotation_operator.hpp       # Rotation operator declaration
├── rotation_operator.cu        # CUDA implementation of rotation
├── app_config.yaml            # Optional configuration file
├── CMakeLists.txt             # Build configuration
└── build/
    └── holoscan_app           # Compiled executable
```

## How the Rotation Works

The `RotationOperator` performs a 90-degree clockwise rotation using CUDA:

- **Input**: Frame of size (height, width, channels) in GPU memory
- **CUDA Kernel**: Each thread processes one pixel, mapping:
  - `output[x][height-1-y] = input[y][x]`
- **Output**: Rotated frame (width, height, channels)

The rotation happens entirely on the GPU for maximum performance.

## Troubleshooting

### Camera not found
```
Failed to open /dev/video0
```
Check that your camera is connected and listed in `/dev/video*`

### CUDA errors
Ensure CUDA is properly installed and compatible with your GPU

### No display output
Verify X11 or Wayland is running and the display environment is set correctly

## References

- [Holoscan Documentation](https://docs.nvidia.com/holoscan/)
- [V4L2 API](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/v4l2.html)
- [CUDA Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
# first_holoscan_program
