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
                       # 5) Export to zones.json
    """

    def __init__(self, image_path: Optional[str | Path] = None) -> None:
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
        
        # Store the initial image path
        self._initial_image_path = Path(image_path) if image_path else None

    def run(self) -> None:
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
        """Load an image from a specific path."""
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
        """Handle the Load Image button click."""
        path = self._filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if path:
            self._load_image_from_path(path)

    def _on_canvas_click(self, event: Any) -> None:
        if self._canvas is None:
            return
        self._draft_points.append((event.x, event.y))
        
        # Automatically complete region after 4 vertices
        if len(self._draft_points) == 4:
            self._regions.append(self._draft_points.copy())
            self._draft_points.clear()
        
        self._draw()

    def _on_undo_point(self) -> None:
        if not self._draft_points:
            self._messagebox.showinfo("Undo Point", "No points to undo")
            return
        self._draft_points.pop()
        self._draw()

    def _on_undo_region(self) -> None:
        if not self._regions:
            self._messagebox.showinfo("Undo", "No regions to undo")
            return
        self._regions.pop()
        self._draw()

    def _on_export(self) -> None:
        if self._image is None or self._canvas is None:
            self._messagebox.showwarning("Export", "Load an image and draw regions first")
            return

        scale_x = self._image_w / max(self._canvas.winfo_width(), 1)
        scale_y = self._image_h / max(self._canvas.winfo_height(), 1)

        data = [
            {"points": [(int(x * scale_x), int(y * scale_y)) for (x, y) in region]}
            for region in self._regions
        ]

        out_path = Path("zones.json")
        out_path.write_text(json.dumps(data, indent=2))
        self._messagebox.showinfo("Export", f"Saved {len(data)} region(s) to {out_path}")

    def _draw(self) -> None:
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
    Zone-based parking occupancy using a Focoos model for object detection.

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
        zones_file: Optional[str | Path] = None, 
        model_path: Optional[str] = None,
        api_key: Optional[str] = None,
        model_ref: Optional[str] = None,
        runtime_type: Optional[str] = None,
        image_size: Optional[int] = None,
        input_video: Optional[str | Path] = None,
        output_video: Optional[str | Path] = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            api_key=api_key,
            model_ref=model_ref,
            runtime_type=runtime_type,
            image_size=image_size,
        )
        
        # Initialize zones
        self._zones_path = Path(zones_file) if zones_file else None
        self._zones: List[Dict[str, Any]] = []
        
        if self._zones_path is not None:
            if not self._zones_path.exists():
                raise FileNotFoundError(f"Zones file not found: {self._zones_path}")
            self._zones = self._load_zones(self._zones_path)

        # Visualization options
        self._color_available = (0, 160, 255)  # orange-ish for available
        self._color_occupied = (60, 220, 20)   # green-ish for occupied
        self._color_centroid = (240, 30, 180)  # magenta
        self._line_thickness = 2
        self._font_scale = 0.5
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Video processing
        self._input_video = Path(input_video) if input_video else None
        self._output_video = Path(output_video) if output_video else None
        
        # FPS tracking
        self._inference_times: List[float] = []
        self._max_fps_samples = 10  # Keep last 10 samples for FPS calculation

    # ---- Public API

    def process(self, image_bgr: np.ndarray) -> ParkingResult:
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
                cls_name = self.cls_names[result.cls_id]
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                inside = cv2.pointPolygonTest(pts, (cx, cy), measureDist=False)
                
                # draw centroid for each detection
                cv2.circle(annotated, (cx, cy), radius=max(3, self._line_thickness * 2), color=self._color_centroid, thickness=-1)
                
                if inside >= 0:
                    self._put_label(annotated, cls_name, (cx, cy))
                    is_occupied = True
                    break

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
        and save the annotated video to the output path.
        """
        if self._input_video is None or self._output_video is None:
            raise ValueError("Both input_video and output_video must be provided for video processing")
        
        from ...core.io import VideoProcessor
        
        with VideoProcessor(self._input_video, self._output_video) as processor:
            processor.process_video(self._process_frame_for_video)
    
    def run(self) -> None:
        """
        Run the appropriate processing based on initialization parameters.
        
        If input_video and output_video are provided, processes the video.
        If no zones file is provided, starts the zone editor on the first frame.
        Otherwise, this method does nothing (use process() for single images).
        """
        if self._input_video is not None and self._output_video is not None:
            # If no zones are loaded, create them interactively from the first frame
            if not self._zones:
                self._create_zones_interactively()
            
            self.process_video()
    
    def _create_zones_interactively(self) -> None:
        """
        Create zones interactively using the first frame of the input video.
        """
        if self._input_video is None:
            raise ValueError("Input video is required for interactive zone creation")
        
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
            print("4. Click 'Export' to save zones.json")
            print("5. Close the editor window")
            
            editor = PolygonZonesEditor(temp_frame_path)
            editor.run()
            
            # Check if zones.json was created
            zones_file = Path("zones.json")
            if zones_file.exists():
                self._zones_path = zones_file
                self._zones = self._load_zones(self._zones_path)
                print(f"Loaded {len(self._zones)} zones from zones.json")
            else:
                raise RuntimeError("No zones were created. Please create zones and export them.")
                
        finally:
            # Clean up temporary file
            if temp_frame_path.exists():
                temp_frame_path.unlink()
    
    def create_zones_from_image(self, image_path: str | Path) -> None:
        """
        Create zones interactively from a specific image.
        
        Args:
            image_path: Path to the image file to use for zone creation
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Start the zone editor with the image already loaded
        print("Starting zone editor. Please draw parking zones on the image.")
        print("Instructions:")
        print("1. Left-click to add vertices for each parking zone")
        print("2. Region automatically completes after 4 vertices")
        print("3. Repeat for all parking zones")
        print("4. Click 'Export' to save zones.json")
        print("5. Close the editor window")
        
        editor = PolygonZonesEditor(image_path)
        editor.run()
        
        # Check if zones.json was created
        zones_file = Path("zones.json")
        if zones_file.exists():
            self._zones_path = zones_file
            self._zones = self._load_zones(self._zones_path)
            print(f"Loaded {len(self._zones)} zones from zones.json")
        else:
            raise RuntimeError("No zones were created. Please create zones and export them.")
    
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

    # ---- Internals

    @staticmethod
    def _load_zones(path: Path) -> List[Dict[str, Any]]:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError("Zones JSON must be a list of objects with a 'points' field")
        for entry in data:
            if "points" not in entry or not isinstance(entry["points"], (list, tuple)):
                raise ValueError("Each zone must contain a 'points' array")
        return data

    def _put_label(self, image: np.ndarray, text: str, anchor: Tuple[int, int]) -> None:
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


