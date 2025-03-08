"""
This module was created using Claude Sonnet 3.7. It contains a graphical user
interface (GUI) for generating images using diffusion models. Note that
huggingface also offers a web-based GUI called 'gradio'
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import sys
import os
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QSlider,
    QPushButton,
    QFileDialog,
    QSpinBox,
    QDoubleSpinBox,
    QProgressBar,
    QMessageBox,
    QGroupBox,
    QSplitter,
    QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap
from PIL.ImageQt import ImageQt
from .models import DiffusionModel
from .utils import get_default_config, save_image


# =========================================================================== #
#                            Image Generation Thread                          #
# =========================================================================== #
class ImageGenerationThread(QThread):
    """Thread for generating images to keep the UI responsive."""

    update_progress = pyqtSignal(int)
    generation_complete = pyqtSignal(object, int)  # image, seed
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        model,
        prompt,
        width,
        height,
        steps,
        guidance,
        seed,
        negative_prompt,
    ) -> None:
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.negative_prompt = negative_prompt

    def run(self) -> None:
        try:
            # Generate the image
            image, actual_seed = self.model.generate_image(
                prompt=self.prompt,
                width=self.width,
                height=self.height,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                seed=self.seed,
                negative_prompt=self.negative_prompt,
            )

            # Emit the completion signal with the image and seed
            self.generation_complete.emit(image, actual_seed)

        except Exception as e:
            self.error_occurred.emit(str(e))


# =========================================================================== #
#                                GUI Class                                    #
# =========================================================================== #
class DiffusionGUI(QMainWindow):
    """GUI for generating images using diffusion models."""

    def __init__(self) -> None:
        super().__init__()

        # Load configuration
        self.config = get_default_config()

        # Find the default model
        self.default_model = next(
            (
                model["id"]
                for model in self.config["models"]
                if model.get("default")
            ),
            self.config["models"][0]["id"],
        )

        # Initialize the model (will load on first use)
        self.model = DiffusionModel(model_id=self.default_model)

        # Initialize UI
        self.init_ui()

    def init_ui(self) -> None:
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle("Diffusion Image Generator")
        self.setMinimumSize(1000, 700)

        # Set the style
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QLabel {
                color: #cdd6f4;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 8px;
                font-weight: bold;
                margin-top: 1em;
                padding-top: 10px;
                color: #cdd6f4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 8px;
                color: #cdd6f4;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #74c7ec;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #89dceb;
            }
            QPushButton:pressed {
                background-color: #94e2d5;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #7f849c;
            }
            QSlider::groove:horizontal {
                border: 1px solid #45475a;
                height: 8px;
                background: #313244;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #74c7ec;
                border: 1px solid #74c7ec;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QProgressBar {
                border: 1px solid #45475a;
                border-radius: 6px;
                background-color: #313244;
                color: #cdd6f4;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #74c7ec;
                width: 20px;
                border-radius: 5px;
            }
            QSplitter::handle {
                background-color: #45475a;
                width: 2px;
            }
            QFrame#imageFrame {
                background-color: #181825;
                border: 1px solid #45475a;
                border-radius: 8px;
            }
        """
        )

        # Create central widget
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Create control panel (left side)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMinimumWidth(350)

        # Add title
        title_label = QLabel("Diffusion Image Generator")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; margin: 10px 0 20px 0; color: #cba6f7;"
        )
        control_layout.addWidget(title_label)

        # Create input group
        input_group = QGroupBox("Input Settings")
        input_layout = QVBoxLayout()

        # Add prompt input
        prompt_label = QLabel("Prompt:")
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your image description...")
        input_layout.addWidget(prompt_label)
        input_layout.addWidget(self.prompt_input)

        # Add negative prompt input
        neg_prompt_label = QLabel("Negative Prompt:")
        self.neg_prompt_input = QLineEdit()
        self.neg_prompt_input.setPlaceholderText(
            "What you DON'T want in the image..."
        )
        input_layout.addWidget(neg_prompt_label)
        input_layout.addWidget(self.neg_prompt_input)

        # Add model selection
        model_label = QLabel("Model:")
        self.model_selector = QComboBox()
        for model in self.config["models"]:
            self.model_selector.addItem(model["name"], model["id"])
        input_layout.addWidget(model_label)
        input_layout.addWidget(self.model_selector)

        # Set the default model
        for i in range(self.model_selector.count()):
            if self.model_selector.itemData(i) == self.default_model:
                self.model_selector.setCurrentIndex(i)
                break

        input_group.setLayout(input_layout)
        control_layout.addWidget(input_group)

        # Create parameter group
        param_group = QGroupBox("Generation Parameters")
        param_layout = QVBoxLayout()

        # Add width and height inputs in a row
        size_layout = QHBoxLayout()

        width_label = QLabel("Width:")
        self.width_input = QSpinBox()
        self.width_input.setRange(256, 1024)
        self.width_input.setSingleStep(64)
        self.width_input.setValue(self.config["parameters"]["width"])
        size_layout.addWidget(width_label)
        size_layout.addWidget(self.width_input)

        height_label = QLabel("Height:")
        self.height_input = QSpinBox()
        self.height_input.setRange(256, 1024)
        self.height_input.setSingleStep(64)
        self.height_input.setValue(self.config["parameters"]["height"])
        size_layout.addWidget(height_label)
        size_layout.addWidget(self.height_input)

        param_layout.addLayout(size_layout)

        # Add steps slider
        steps_label = QLabel(
            f"Inference Steps: {self.config['parameters']['num_inference_steps']}"
        )
        self.steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.steps_slider.setMinimum(10)
        self.steps_slider.setMaximum(100)
        self.steps_slider.setValue(
            self.config["parameters"]["num_inference_steps"]
        )
        self.steps_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.steps_slider.setTickInterval(10)
        self.steps_slider.valueChanged.connect(
            lambda v: steps_label.setText(f"Inference Steps: {v}")
        )
        param_layout.addWidget(steps_label)
        param_layout.addWidget(self.steps_slider)

        # Add guidance scale input
        guidance_label = QLabel("Guidance Scale:")
        self.guidance_input = QDoubleSpinBox()
        self.guidance_input.setRange(1.0, 20.0)
        self.guidance_input.setSingleStep(0.5)
        self.guidance_input.setValue(
            self.config["parameters"]["guidance_scale"]
        )
        param_layout.addWidget(guidance_label)
        param_layout.addWidget(self.guidance_input)

        # Add seed input
        seed_layout = QHBoxLayout()
        seed_label = QLabel("Seed (0 for random):")
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 2**31 - 1)
        self.seed_input.setValue(0)  # Default to random seed
        seed_layout.addWidget(seed_label)
        seed_layout.addWidget(self.seed_input)
        param_layout.addLayout(seed_layout)

        param_group.setLayout(param_layout)
        control_layout.addWidget(param_group)

        # Add output directory selection
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()

        output_dir_label = QLabel("Output Directory:")
        output_layout.addWidget(output_dir_label)

        output_dir_row = QHBoxLayout()
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setText(
            os.path.abspath(self.config["parameters"]["output_dir"])
        )

        browse_button = QPushButton("Browse")
        browse_button.setMaximumWidth(100)
        browse_button.clicked.connect(self.browse_output_dir)

        output_dir_row.addWidget(self.output_dir_input)
        output_dir_row.addWidget(browse_button)
        output_layout.addLayout(output_dir_row)

        output_group.setLayout(output_layout)
        control_layout.addWidget(output_group)

        # Add generate button
        self.generate_button = QPushButton("Generate Image")
        self.generate_button.clicked.connect(self.generate_image)
        self.generate_button.setMinimumHeight(50)
        self.generate_button.setStyleSheet(
            """
            QPushButton {
                background-color: #a6e3a1;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #94e2d5;
            }
        """
        )
        control_layout.addWidget(self.generate_button)

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready")
        control_layout.addWidget(self.progress_bar)

        # Add spacer at the bottom of control panel
        control_layout.addStretch()

        # Add the control panel to the splitter
        splitter.addWidget(control_panel)

        # Create image display panel (right side)
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)

        # Image display title
        image_title = QLabel("Generated Image")
        image_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_title.setStyleSheet(
            "font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #cba6f7;"
        )
        image_layout.addWidget(image_title)

        # Create frame for image
        image_frame = QFrame()
        image_frame.setObjectName("imageFrame")
        image_frame.setFrameShape(QFrame.Shape.StyledPanel)
        image_frame_layout = QVBoxLayout(image_frame)

        # Add image display
        self.image_display = QLabel("Your generated image will appear here")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setMinimumSize(512, 512)
        image_frame_layout.addWidget(self.image_display)

        image_layout.addWidget(image_frame, 1)  # 1 = stretch factor

        # Add image info label
        self.image_info = QLabel("")
        self.image_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_info.setStyleSheet("color: #cdd6f4; font-style: italic;")
        image_layout.addWidget(self.image_info)

        # Add save button (disabled by default)6
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_current_image)
        self.save_button.setEnabled(False)
        self.save_button.setMinimumHeight(40)
        self.save_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f9e2af;
                color: #1e1e2e;
            }
            QPushButton:hover {
                background-color: #fab387;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #7f849c;
            }
        """
        )
        image_layout.addWidget(self.save_button)

        # Store the current generated image and seed
        self.current_image = None
        self.current_seed = None

        # Add the image panel to the splitter
        splitter.addWidget(image_panel)

        # Set initial splitter sizes
        splitter.setSizes([350, 650])

        # Set the central widget
        self.setCentralWidget(central_widget)

        # Setup a timer for animated "loading" effect
        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self.update_loading_animation)
        self.loading_dots = 0

    def browse_output_dir(self) -> None:
        """Open a dialog to select the output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_dir_input.text()
        )
        if directory:
            self.output_dir_input.setText(directory)

    def update_loading_animation(self) -> None:
        """Update the loading animation in the progress bar."""
        self.loading_dots = (self.loading_dots + 1) % 4
        dots = "." * self.loading_dots
        self.progress_bar.setFormat(f"Generating{dots}")

    def generate_image(self) -> None:
        """Generate an image based on the current settings."""
        # Check if prompt is empty
        if not self.prompt_input.text().strip():
            QMessageBox.warning(self, "Warning", "Please enter a prompt.")
            return

        # Disable the generate button during generation
        self.generate_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Initializing...")

        # Start the loading animation
        self.loading_timer.start(300)

        # Get values from UI
        prompt = self.prompt_input.text()
        model_id = self.model_selector.currentData()
        width = self.width_input.value()
        height = self.height_input.value()
        steps = self.steps_slider.value()
        guidance = self.guidance_input.value()
        seed = self.seed_input.value() if self.seed_input.value() > 0 else None
        negative_prompt = (
            self.neg_prompt_input.text()
            if self.neg_prompt_input.text()
            else None
        )

        # Update the model if the selected model has changed
        if self.model.model_id != model_id:
            self.model = DiffusionModel(model_id=model_id)

        # Create and start the generation thread
        self.generation_thread = ImageGenerationThread(
            model=self.model,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed,
            negative_prompt=negative_prompt,
        )

        # Connect signals
        self.generation_thread.generation_complete.connect(
            self.display_generated_image
        )
        self.generation_thread.error_occurred.connect(
            self.handle_generation_error
        )

        # Start the thread
        self.generation_thread.start()

    def display_generated_image(self, image: Image.Image, seed: int):
        """Display the generated image."""
        # Stop the loading animation
        self.loading_timer.stop()

        # Store the current image and seed
        self.current_image = image
        self.current_seed = seed

        # Convert PIL Image to QPixmap
        q_image = ImageQt(image)
        pixmap = QPixmap.fromImage(q_image)

        # Resize if necessary while maintaining aspect ratio
        display_size = self.image_display.size()
        if (
            pixmap.width() > display_size.width()
            or pixmap.height() > display_size.height()
        ):
            pixmap = pixmap.scaled(
                display_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # Set the pixmap to the display
        self.image_display.setPixmap(pixmap)

        # Update image info
        width = image.width
        height = image.height
        model_name = self.model_selector.currentText()
        self.image_info.setText(
            f"Generated with {model_name}, size: {width}x{height}, seed: {seed}"
        )

        # Enable the save button
        self.save_button.setEnabled(True)

        # Reset the progress bar
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Generation Complete!")

        # Re-enable the generate button
        self.generate_button.setEnabled(True)

    def handle_generation_error(self, error_message: str) -> None:
        """Handle errors that occur during image generation."""
        # Stop the loading animation
        self.loading_timer.stop()

        # Show error message
        QMessageBox.critical(
            self, "Error", f"Error generating image: {error_message}"
        )

        # Reset the progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Generation Failed")

        # Re-enable the generate button
        self.generate_button.setEnabled(True)

    def save_current_image(self) -> None:
        """Save the currently displayed image."""
        if self.current_image is None:
            return

        # Get the output directory
        output_dir = self.output_dir_input.text()

        # Save the image
        try:
            filepath = save_image(
                image=self.current_image,
                output_dir=output_dir,
                filename_prefix=self.config["parameters"]["filename_prefix"],
                prompt=self.prompt_input.text(),
            )

            # Show success message
            QMessageBox.information(
                self, "Success", f"Image saved successfully to:\n{filepath}"
            )

        except Exception as e:
            # Show error message
            QMessageBox.critical(self, "Error", f"Error saving image: {e}")


# =========================================================================== #
#                                Main Function                                #
# =========================================================================== #
def run_gui() -> None:
    """Run the GUI application."""
    app = QApplication(sys.argv)
    window = DiffusionGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
