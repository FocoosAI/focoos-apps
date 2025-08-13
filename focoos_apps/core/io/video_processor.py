"""
Video processing utilities for reading and writing video files.

This module provides the VideoProcessor class for efficient video file handling,
including reading frames, writing processed frames, and managing video properties.
It supports context manager usage for automatic resource cleanup.
"""

import cv2
from pathlib import Path
from typing import Optional, Tuple


class VideoProcessor:
    """Handles video file reading and writing operations."""
    
    def __init__(self, input_path: str | Path, output_path: str | Path):
        """
        Initialize video processor.
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input video file not found: {self.input_path}")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(str(self.input_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Error reading video file: {self.input_path}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path), 
            fourcc, 
            self.fps, 
            (self.width, self.height)
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Error creating output video file: {self.output_path}")
    
    def __enter__(self):
        """
        Context manager entry.
        
        Returns:
            self: The VideoProcessor instance for use in with statement
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - cleanup resources.
        
        Automatically releases video capture and writer resources
        when exiting a with statement.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.release()
    
    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Read a single frame from the video.
        
        Returns:
            Tuple of (success, frame) where frame is None if reading failed
        """
        return self.cap.read()
    
    def write_frame(self, frame: cv2.Mat) -> None:
        """
        Write a frame to the output video.
        
        Args:
            frame: Frame to write (numpy array)
        """
        self.writer.write(frame)
    
    def get_properties(self) -> Tuple[int, int, int, int]:
        """
        Get video properties.
        
        Returns:
            Tuple of (width, height, fps, frame_count)
        """
        return self.width, self.height, self.fps, self.frame_count
    
    def release(self) -> None:
        """Release video capture and writer resources."""
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        # Only destroy windows if we're in an environment that supports it
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Ignore errors in headless environments where GUI support is not available
            pass
    
    def process_video(self, frame_processor) -> None:
        """
        Process entire video with a frame processor function.
        
        Reads all frames from the input video, applies the provided
        frame processor function to each frame, and writes the processed
        frames to the output video. Automatically handles resource cleanup.
        
        Args:
            frame_processor: Function that takes a frame and returns processed frame
        """
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                processed_frame = frame_processor(frame)
                self.writer.write(processed_frame)
        finally:
            self.release()
