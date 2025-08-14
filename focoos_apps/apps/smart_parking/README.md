# Smart Parking Application

## Overview

The **Smart Parking Application** is an AI-powered solution for detecting and monitoring parking space occupancy in real-time. It uses computer vision to analyze video feeds or images and provides detailed analytics about parking utilization.

### Features

- 🚗 **Real-time Vehicle Detection**: Detect vehicles in parking spaces using advanced AI models
- 📊 **Occupancy Analytics**: Track occupied vs. available parking slots
- 🎯 **Custom Zone Definition**: Interactive tool to define parking zones
- 📹 **Video Processing**: Process video files with annotated output
- 📈 **Performance Metrics**: Monitor model FPS

## Prerequisites

### Focoos AI SDK Access

You'll need:
1. **API Key**: Get your API key from [Focoos AI Platform](https://app.focoos.ai)
2. **Model Reference**: Use the provided model reference or your own trained model
3. **Runtime Type**: Choose between CPU, GPU, or optimized runtimes


## Usage Example

The Smart Parking Application automatically handles parking zones based on whether a zones file exists:

### Python API

```python
from focoos_apps.apps.smart_parking import SmartParkingApp

# Initialize the application
parking_app = SmartParkingApp(
    input_video="path/to/input.mp4",           # Input video file
    model_ref="hub://your_model_reference",    # Model reference
    api_key="your_api_key_here",               # Your Focoos API key
    output_video="path/to/output.mp4"          # Output video file
    zones_file="path/to/zones.json",           # Zones file (optional)
    runtime="tensorrt",                   # Runtime type (cpu, cuda, tensorrt)
)

# Run the application
parking_app.run()
```

### Command Line Interface

You can also use the application via command line:

```bash
# Process video with existing zones
focoos-apps smart-parking \
    --input-video input.mp4 \
    --model-ref hub://your_model_reference \
    --api-key your_api_key_here \
    --output-video output.mp4 \
    --zones-file zones.json \
    --runtime tensorrt
```

**Application Behavior:**

- **If `zones_file` is provided and exists**: The app processes the video using the pre-defined parking zones
- **If `zones_file` is not provided or doesn't exist**: The app automatically:
  1. Extracts the first frame from the video (if you want to create zones from an image, you can use the `Load Image` button)
  2. Opens the interactive zone editor
  3. Waits for you to create parking zones
  4. Saves the zones to the specified file
  5. Processes the entire video with your newly created zones

## Zone Editor Usage

The interactive zone editor allows you to define parking spaces:

### Controls

- **Load Image**: Select an image to work with
- **Left Click**: Add vertices to create a parking zone (4 vertices per zone)
- **Undo Point**: Remove the last added vertex
- **Undo Region**: Remove the last completed region
- **Export**: Save zones to JSON file

### Zone Creation Process

1. **Load Image (optional)**: Click "Load Image" and select your parking lot image
2. **Define Zones**: Click 4 points to define each parking space
   - Start from any corner
   - Click in clockwise or counter-clockwise order
   - The zone will automatically complete after the 4th click
3. **Repeat**: Continue adding zones for all parking spaces
4. **Export**: Click "Export" to save your zones to `path/to/zones.json`

### Zone File Format

The zones are saved in JSON format:

```json
{
  "zones": [
    {
      "id": 0,
      "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    },
    {
      "id": 1,
      "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ]
}
```



## Output Format

### Video Output

The processed video includes:
- **Point** for each detected vehicle
- **Zone highlighting** for each parking space
- **Occupancy status** (occupied/available) for each zone
- **Summary overlay** with total counts and FPS

### Data Output

The application provides structured data:

```python
@dataclass
class ParkingSummary:
    occupied_slots: int      # Number of occupied parking spaces
    available_slots: int     # Number of available parking spaces
    total_detections: int    # Total vehicles detected
    model_fps: float         # Processing speed in FPS
```

## Performance Optimization

### Runtime Selection

Choose the appropriate runtime for your hardware:

- **CPU**: `"cpu"` - Good for development and testing
- **GPU**: `"cuda"` - Faster processing with NVIDIA GPU  
- **Optimized**: `"tensorrt"` - Best performance with NVIDIA TensorRT

## Troubleshooting

### Common Issues

**1. Tkinter Not Available**
```
Error: Tkinter is not available
```
**Solution**: Install tkinter for your platform:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS
brew install python-tk

# Windows
# Usually included with Python installation
```

### Getting Help

- 📧 **Email**: info@focoos.ai
- 🐛 **Issues**: [GitHub Issues](https://github.com/FocoosAI/focoos-apps/issues)
- 📚 **Documentation**: [Focoos AI Docs](https://focoosai.github.io/focoos/)



## License

This application is part of the Focoos Apps repository and is licensed under the same terms as the main project.
