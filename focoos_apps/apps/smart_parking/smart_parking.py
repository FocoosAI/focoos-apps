"""
Smart Parking Application for Focoos AI.

This module provides a complete smart parking solution that uses AI to detect
parking occupancy in videos or images. It includes zone-based detection,
interactive zone editing, and video processing capabilities.

The application consists of:
- Data structures for parking results and summaries
- Interactive polygon zone editor for defining parking areas
- Smart parking detection application using Focoos AI models
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ...core.app_base import BaseApp


# -------------------------------
# Data structures
# -------------------------------


@dataclass
class ParkingSummary:
    occupied_slots: int
    available_slots: int
    total_detections: int
    model_fps: float


@dataclass
class ParkingResult:
    annotated_image: np.ndarray
    summary: ParkingSummary


# -------------------------------
# Interactive polygon editor (zones authoring)
# -------------------------------


class PolygonZonesEditor:
    """
    Lightweight Tkinter-based tool to draw one or more polygons on an image and export them to JSON.

    Usage:
        editor = PolygonZonesEditor()
        editor.run()  # opens a small UI where you can:
                       # 1) Load an image
                       # 2) Left-click to add vertices (4 vertices per region)
                       # 3) Region automatically completes after 4th vertex
                       # 4) Undo last region
                       # 5) Export to the specified zones JSON file
    """

    def __init__(self, image_path: Optional[str | Path] = None, zones_file: Optional[str | Path] = None) -> None:
        """
        Initialize the polygon zones editor.
        
        Sets up the Tkinter interface and initializes the drawing canvas.
        Optionally loads an initial image if provided.
        
        Args:
            image_path: Optional path to an image file to load initially
            zones_file: Path where zones will be exported (defaults to zones.json)
        """
        try:
            import tkinter as tk  # type: ignore
            from tkinter import filedialog, messagebox  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            print(
                "Tkinter is not available. Install tk for your platform (e.g., `sudo apt-get install python3-tk`).",
                exc,
            )
            self._tk = None  # sentinel to disable run()
            return

        self._tk = tk
        self._filedialog = filedialog
        self._messagebox = messagebox

        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._photo_image = None

        self._image = None  # PIL.Image, typed at runtime
        self._image_w: int = 0
        self._image_h: int = 0

        self._max_canvas_w: int = 1280
        self._max_canvas_h: int = 720

        self._regions: List[List[Tuple[int, int]]] = []
        self._draft_points: List[Tuple[int, int]] = []
        
        # Store the initial image path and zones file path
        self._initial_image_path = Path(image_path) if image_path else None
        self._zones_file_path = Path(zones_file) if zones_file else Path("zones.json")

    def run(self) -> None:
        """
        Start the interactive zone editor interface.
        
        Creates the main window with toolbar buttons and drawing canvas.
        If an initial image was provided, it will be automatically loaded.
        The interface remains open until the user closes the window.
        """
        if self._tk is None:  # pragma: no cover - environment dependent
            return

        self._root = self._tk.Tk()
        self._root.title("Focoos Zones Editor")
        self._root.resizable(False, False)

        toolbar = self._tk.Frame(self._root)
        toolbar.pack(side=self._tk.TOP, fill=self._tk.X)

        self._tk.Button(toolbar, text="Load Image", command=self._on_load_image).pack(side=self._tk.LEFT)
        self._tk.Button(toolbar, text="Undo Point", command=self._on_undo_point).pack(side=self._tk.LEFT)
        self._tk.Button(toolbar, text="Undo Region", command=self._on_undo_region).pack(side=self._tk.LEFT)
        self._tk.Button(toolbar, text="Export", command=self._on_export).pack(side=self._tk.LEFT)

        self._canvas = self._tk.Canvas(self._root, bg="white")
        self._canvas.pack(side=self._tk.BOTTOM)
        self._canvas.bind("<Button-1>", self._on_canvas_click)

        # Auto-load image if provided
        if self._initial_image_path is not None:
            self._load_image_from_path(self._initial_image_path)

        self._root.mainloop()

    # ---- UI callbacks

    def _load_image_from_path(self, image_path: str | Path) -> None:
        """
        Load an image from a specific path and display it on the canvas.
        
        Resizes the image to fit within the maximum canvas dimensions
        while maintaining aspect ratio. Clears any existing regions.
        
        Args:
            image_path: Path to the image file to load
        """
        assert self._tk is not None
        assert self._canvas is not None

        from PIL import Image, ImageTk  # scoped import to avoid hard dependency if editor is unused

        img = Image.open(image_path)
        self._image_w, self._image_h = img.size

        aspect = self._image_w / max(self._image_h, 1)
        canvas_w = min(self._max_canvas_w, int(self._max_canvas_h * aspect)) if aspect >= 1 else int(self._max_canvas_h * aspect)
        canvas_h = min(self._max_canvas_h, int(canvas_w / max(aspect, 1e-6))) if aspect >= 1 else self._max_canvas_h

        resized = img.resize((canvas_w, canvas_h))
        self._photo_image = ImageTk.PhotoImage(resized)
        self._image = img

        self._canvas.config(width=canvas_w, height=canvas_h)
        self._canvas.create_image(0, 0, anchor=self._tk.NW, image=self._photo_image)
        self._regions.clear()
        self._draft_points.clear()

    def _on_load_image(self) -> None:
        """
        Handle the Load Image button click.
        
        Opens a file dialog for selecting an image file and loads it
        into the editor if a file is selected.
        """
        path = self._filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if path:
            self._load_image_from_path(path)

    def _on_canvas_click(self, event: Any) -> None:
        """
        Handle mouse clicks on the canvas for adding polygon vertices.
        
        Adds the clicked point to the draft points list. When 4 points
        are collected, automatically completes the region and starts a new one.
        
        Args:
            event: Tkinter mouse event containing x, y coordinates
        """
        if self._canvas is None:
            return
        self._draft_points.append((event.x, event.y))
        
        # Automatically complete region after 4 vertices
        if len(self._draft_points) == 4:
            self._regions.append(self._draft_points.copy())
            self._draft_points.clear()
        
        self._draw()

    def _on_undo_point(self) -> None:
        """
        Handle the Undo Point button click.
        
        Removes the last added draft point if any exist.
        Shows an info message if no points are available to undo.
        """
        if not self._draft_points:
            self._messagebox.showinfo("Undo Point", "No points to undo")
            return
        self._draft_points.pop()
        self._draw()

    def _on_undo_region(self) -> None:
        """
        Handle the Undo Region button click.
        
        Removes the last completed region if any exist.
        Shows an info message if no regions are available to undo.
        """
        if not self._regions:
            self._messagebox.showinfo("Undo", "No regions to undo")
            return
        self._regions.pop()
        self._draw()

    def _on_export(self) -> None:
        """
        Handle the Export button click.
        
        Exports all completed regions to a JSON file. Scales the coordinates
        from canvas space back to original image space. Shows success or
        warning messages as appropriate.
        """
        if self._image is None or self._canvas is None:
            self._messagebox.showwarning("Export", "Load an image and draw regions first")
            return

        scale_x = self._image_w / max(self._canvas.winfo_width(), 1)
        scale_y = self._image_h / max(self._canvas.winfo_height(), 1)

        data = [
            {"points": [(int(x * scale_x), int(y * scale_y)) for (x, y) in region]}
            for region in self._regions
        ]

        out_path = self._zones_file_path
        out_path.write_text(json.dumps(data, indent=2))
        self._messagebox.showinfo("Export", f"Saved {len(data)} region(s) to {out_path}")

    def _draw(self) -> None:
        """
        Redraw the canvas with current image, regions, and draft points.
        
        Clears the canvas and redraws the background image, completed regions
        in blue, draft polylines in orange, and draft points as red circles.
        """
        if self._canvas is None or self._photo_image is None:
            return
        assert self._tk is not None

        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=self._tk.NW, image=self._photo_image)

        # committed regions
        for region in self._regions:
            if len(region) >= 2:
                self._canvas.create_polygon(*sum(region, ()), outline="blue", fill="", width=2)

        # draft polyline and points
        if len(self._draft_points) >= 2:
            self._canvas.create_line(*sum(self._draft_points, ()), fill="orange", width=2)
        
        # Draw individual points for draft region (larger and more visible)
        for i, (x, y) in enumerate(self._draft_points):
            # Draw a larger circle with outline for better visibility
            self._canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="red", outline="white", width=2)


# -------------------------------
# Smart parking application (Focoos)
# -------------------------------


class SmartParkingApp(BaseApp):
    """
    Zone-based parking occupancy detection using a Focoos model for object detection.

    This application processes videos or images to detect parking occupancy
    in predefined zones. It uses AI object detection to identify vehicles
    and determines if they are within specified parking areas.

    Inputs:
        - image: np.ndarray in BGR (OpenCV convention)
        - zones_file: JSON file with format: [{"points": [[x, y], ...]}, ...]

    Behavior:
        - Runs inference via `self.model(image)`
        - Computes object centroids and checks whether they fall inside zone polygons
        - Annotates the frame with colored zones and stats banner
    """

    def __init__(
        self, 
        input_video: str | Path,
        model_ref: str,
        output_video: Optional[str | Path] = None,
        api_key: Optional[str] = None,
        runtime: Optional[str] = "cpu",
        image_size: Optional[int] = None,
        zones_file: Optional[str | Path] = None,
    ) -> None:
        """
        Initialize the Smart Parking application.
        
        Sets up the AI model, loads parking zones, and configures visualization
        parameters. If zones file doesn't exist, zones can be created interactively.
        
        Args:
            input_video: Path to input video file for processing
            output_video: Path where annotated output video will be saved
            api_key: Focoos API key for model access (inherited from BaseApp)
            model_ref: Model reference identifier (inherited from BaseApp)
            runtime: Runtime type for model execution (inherited from BaseApp)
            image_size: Input image size for model optimization (inherited from BaseApp)
            zones_file: Path to JSON file containing parking zone definitions
        """
        super().__init__(
            api_key=api_key,
            model_ref=model_ref,
            runtime=runtime,
            image_size=image_size,
        )

        # Video input and output
        self._input_video = Path(input_video)
        if not self._input_video.exists():
            raise FileNotFoundError(f"Input video file not found: {self._input_video}")

        self._output_video = Path(output_video) if output_video else self._input_video.with_suffix(".annotated.mp4")
        
        # Initialize zones
        self._zones_path = Path(zones_file) if zones_file else None
        self._zones: List[Dict[str, Any]] = []
        
        # Load zones if file exists, otherwise will be created interactively
        if self._zones_path is not None and self._zones_path.exists():
            self._zones = self._load_zones(self._zones_path)
        
        # FPS tracking
        self._inference_times: List[float] = []
        self._max_fps_samples = 10  # Keep last 10 samples for FPS calculation

        # Visualization options
        self._color_available = self._hex_to_bgr("#63DCA7")      # green focoos for available
        self._color_occupied = self._hex_to_bgr("#E23670")       # magenta focoos for occupied
        self._color_centroid = self._hex_to_bgr("#025EE6")       # blue focoos for centroid
        self._line_thickness = 2
        self._font_scale = 0.5
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    # ---- Public API

    def process(self, image_bgr: np.ndarray) -> ParkingResult:
        """
        Process a single image to detect parking occupancy.
        
        Runs AI inference on the input image, checks vehicle detections
        against defined parking zones, and returns annotated results.
        
        Args:
            image_bgr: Input image in BGR format (OpenCV convention)
            
        Returns:
            ParkingResult containing annotated image and summary statistics
            
        Raises:
            RuntimeError: If model is not initialized
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized. Provide an init_config with model_ref or set self.model.")

        # Track inference time
        start_time = time.time()
        results = self.model(image_bgr)
        inference_time = time.time() - start_time
        
        # Update FPS tracking
        self._inference_times.append(inference_time)
        if len(self._inference_times) > self._max_fps_samples:
            self._inference_times.pop(0)
        
        # Calculate average FPS
        avg_inference_time = sum(self._inference_times) / len(self._inference_times)
        model_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0

        occupied_count = 0
        total_zones = len(self._zones)
        annotated = image_bgr.copy()

        # Evaluate occupancy per zone
        for zone in self._zones:
            pts = np.asarray(zone["points"], dtype=np.int32).reshape((-1, 1, 2))
            is_occupied = False

            for result in results.detections:
                box = result.bbox if result.bbox is not None else np.empty(4)
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                inside = cv2.pointPolygonTest(pts, (cx, cy), measureDist=False)
                
                if inside >= 0:
                    # draw centroid for detection inside the zone
                    cv2.circle(annotated, (cx, cy), radius=max(3, self._line_thickness * 2), color=self._color_occupied, thickness=-1)
                    is_occupied = True
                    break
                else:
                    # draw centroid for each other detection
                    cv2.circle(annotated, (cx, cy), radius=max(3, self._line_thickness * 2), color=self._color_centroid, thickness=-1)
                
            if is_occupied:
                occupied_count += 1

            # draw the zone polygon
            cv2.polylines(
                annotated,
                [pts],
                isClosed=True,
                color=self._color_occupied if is_occupied else self._color_available,
                thickness=self._line_thickness,
            )

        available_count = total_zones - occupied_count
        self._draw_stats_banner(annotated, occupied_count, available_count, model_fps)

        return ParkingResult(
            annotated_image=annotated,
            summary=ParkingSummary(
                occupied_slots=occupied_count,
                available_slots=available_count,
                total_detections=len(results.detections),
                model_fps=model_fps,
            ),
        )

    def process_video(self) -> None:
        """
        Process the entire input video and save the annotated output.
        
        This method will process all frames in the input video, apply parking detection,
        and save the annotated video to the output path. Uses the VideoProcessor
        class for efficient video handling.
        """
        from ...core.io import VideoProcessor
        
        with VideoProcessor(self._input_video, self._output_video) as processor:
            processor.process_video(self._process_frame_for_video)
    
    def run(self) -> None:
        """
        Run the appropriate processing based on initialization parameters.

        If no zones are loaded (zones file doesn't exist), starts the zone editor
        on the first frame to create zones interactively.
        """
        if not self._zones:
            self._create_zones_interactively()
        
        self.process_video()
    
    # ---- Internals
    
    def _create_zones_interactively(self) -> None:
        """
        Create zones interactively using the first frame of the input video.
        
        Extracts the first frame from the input video, saves it temporarily,
        and launches the PolygonZonesEditor to allow the user to draw parking zones.
        After zones are created, they are loaded and the temporary file is cleaned up.
        
        Raises:
            RuntimeError: If video cannot be read or zones are not created
        """
        # Read the first frame
        cap = cv2.VideoCapture(str(self._input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Error reading video file: {self._input_video}")
        
        ret, first_frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError("Could not read first frame from video")
        
        # Save the first frame temporarily
        temp_frame_path = Path("temp_first_frame.jpg")
        cv2.imwrite(str(temp_frame_path), first_frame)
        
        try:
            # Start the zone editor with the first frame already loaded
            print("Starting zone editor. Please draw parking zones on the first frame.")
            print("Instructions:")
            print("1. Left-click to add vertices for each parking zone")
            print("2. Region automatically completes after 4 vertices")
            print("3. Repeat for all parking zones")
            print("4. Click 'Export' to save the zones file")
            print("5. Close the editor window")
            
            editor = PolygonZonesEditor(temp_frame_path, self._zones_path)
            editor.run()
            
            # Check if zones.json was created and use the specified zones file path if provided
            zones_file = self._zones_path if self._zones_path is not None else Path("zones.json")
            if zones_file.exists():
                self._zones = self._load_zones(zones_file)
                print(f"Loaded {len(self._zones)} zones from {zones_file}")
            else:
                raise RuntimeError("No zones were created. Please create zones and export them.")
                
        finally:
            # Clean up temporary file
            if temp_frame_path.exists():
                temp_frame_path.unlink()

    
    def _process_frame_for_video(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for video output.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Annotated frame
        """
        result = self.process(frame)
        return result.annotated_image

    @staticmethod
    def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
        """
        Convert hex color string to BGR tuple for OpenCV.
        
        Args:
            hex_color: Hex color string (e.g., "#63DCA7")
            
        Returns:
            BGR tuple (blue, green, red)
        """
        # Remove the # if present
        hex_color = hex_color.lstrip('#')
        
        # Convert hex to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        # Return as BGR (OpenCV format)
        return (b, g, r)

    @staticmethod
    def _load_zones(path: Path) -> List[Dict[str, Any]]:
        """
        Load parking zones from a JSON file.
        
        Parses the JSON file and validates the structure to ensure
        it contains the expected zone format.
        
        Args:
            path: Path to the zones JSON file
            
        Returns:
            List of zone dictionaries, each containing a 'points' field
            
        Raises:
            ValueError: If JSON structure is invalid or missing required fields
        """
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError("Zones JSON must be a list of objects with a 'points' field")
        for entry in data:
            if "points" not in entry or not isinstance(entry["points"], (list, tuple)):
                raise ValueError("Each zone must contain a 'points' array")
        return data

    def _put_label(self, image: np.ndarray, text: str, anchor: Tuple[int, int]) -> None:
        """
        Draw a text label with background on the image.
        
        Creates a dark background rectangle behind the text for better
        visibility and draws the text in white.
        
        Args:
            image: Image to draw on (modified in place)
            text: Text string to display
            anchor: (x, y) coordinates for text placement
        """
        x, y = anchor
        (w, h), baseline = cv2.getTextSize(text, self._font, self._font_scale, thickness=1)
        pad = 3
        cv2.rectangle(
            image,
            (x + 6, y - h - baseline - pad),
            (x + 6 + w + 2 * pad, y + baseline + pad),
            color=(30, 30, 30),
            thickness=-1,
        )
        cv2.putText(image, text, (x + 6 + pad, y + pad), self._font, self._font_scale, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    def _draw_stats_banner(self, image: np.ndarray, occupied: int, available: int, model_fps: float) -> None:
        """
        Draw a statistics banner on the image.
        
        Creates a dark banner at the top of the image showing occupancy
        statistics and model performance metrics.
        
        Args:
            image: Image to draw on (modified in place)
            occupied: Number of occupied parking slots
            available: Number of available parking slots
            model_fps: Current model FPS performance
        """
        text = f"Occupied: {occupied}   Available: {available}   Model FPS: {model_fps:.1f}"
        (w, h), _ = cv2.getTextSize(text, self._font, self._font_scale + 0.1, thickness=2)
        margin = 10
        cv2.rectangle(image, (margin - 6, margin - 6), (margin + w + 6, margin + h + 6), (40, 40, 40), thickness=-1)
        cv2.putText(
            image,
            text,
            (margin, margin + h),
            self._font,
            self._font_scale + 0.1,
            (255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )


__all__ = [
    "PolygonZonesEditor",
    "SmartParkingApp",
    "ParkingResult",
    "ParkingSummary",
]


