import sys
import os
import numpy as np
import matplotlib
import sympy
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QLabel, QSlider, QPushButton, QComboBox,
                            QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox, QDial, QScrollArea, QSizePolicy, QFileDialog,
                            QDialog, QTextEdit, QMessageBox, QTabWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPalette, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from sympy.printing.mathematica import mathematica_code
from matplotlib import cm
import platform

from visualization import plot_polarization_intensity
from point_groups import (point_group_components, create_tensor_with_relations,
                        get_all_point_groups, get_components_for_group, str_to_indices)
from symbolic_calculator import calculate_symbolic_shg_expressions

# Define preset parameters for common crystals
# Coefficient values are multipliers for the independent components of the point group. These values are for demonstration;
# for practical applications, refer to precise literature values and consider units.
from models.crystal_presets import COMMON_CRYSTALS

# --- Euler Angle Rotation Helper Functions ---
def get_rotation_matrix(phi_c, theta_c, psi_c):
    from core.tensor_operations import TensorOperations
    return TensorOperations.get_rotation_matrix(phi_c, theta_c, psi_c)

def rotate_tensor(tensor_crystal, R):
    """ Transform a 3rd-rank tensor from the crystal frame to the lab frame using rotation matrix R. """
    tensor_lab = np.zeros((3,3,3), dtype=tensor_crystal.dtype)
    for p in range(3):
        for q in range(3):
            for r in range(3):
                sum_val = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            sum_val += R[p,i] * R[q,j] * R[r,k] * tensor_crystal[i,j,k]
                tensor_lab[p,q,r] = sum_val
    return tensor_lab
# --- End of Euler Angle Rotation Helper Functions ---

# Check if LaTeX is available - a safer method
def is_latex_available():
    try:
        # Simply check if the 'latex' command is installed
        import shutil
        return shutil.which('latex') is not None
    except:
        return False

# Configure matplotlib for font and style
# Please ensure that at least one of the fonts in the following list is available on your system.
# For example: 'Microsoft YaHei', 'SimHei', 'SimSun', or 'Arial'.
# This list is updated to be more suitable for macOS, while retaining common Windows fonts as fallbacks.
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Microsoft YaHei', 'SimHei', 'SimSun', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # Correctly display the minus sign
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['lines.linewidth'] = 2

# Disable LaTeX, use plain text rendering
use_latex = False  # Directly set to False to avoid LaTeX-related issues

# Create a custom QSlider class that supports double-click events
class ClickableSlider(QSlider):
    doubleClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)  # Default to horizontal slider
        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self and event.type() == QEvent.MouseButtonDblClick:
            self.doubleClicked.emit()
            return True
        return super().eventFilter(obj, event)

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_constrained_layout(True)
        self.axes = self.fig.add_subplot(111, polar=True)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.logo_ax_on_fig = None

    def draw_logo_on_figure(self, q_image_logo):
        if q_image_logo is None or q_image_logo.isNull():
            print("Logo QImage is null, skipping drawing logo on figure.")
            return
        if self.logo_ax_on_fig:
            try:
                self.logo_ax_on_fig.remove()
            except Exception as e:
                print(f"Error removing previous logo_ax_on_fig: {e}")
            self.logo_ax_on_fig = None
        try:
            ptr = q_image_logo.bits()
            ptr.setsize(q_image_logo.byteCount())
            if q_image_logo.format() != QImage.Format_RGBA8888 and q_image_logo.format() != QImage.Format_ARGB32:
                 temp_image = q_image_logo.convertToFormat(QImage.Format_RGBA8888)
                 if temp_image.isNull():
                     print("Error: Could not convert QImage to RGBA8888 for logo.")
                     return
                 ptr = temp_image.bits()
                 ptr.setsize(temp_image.byteCount())
                 logo_numpy_array = np.array(ptr).reshape(temp_image.height(), temp_image.width(), 4)
            else:
                 logo_numpy_array = np.array(ptr).reshape(q_image_logo.height(), q_image_logo.width(), 4)
            fig_width_px, fig_height_px = self.fig.get_size_inches() * self.fig.dpi
            image_aspect_ratio = q_image_logo.width() / q_image_logo.height()
            margin_fig_rel = 0.01
            logo_height_fig_rel = 0.08
            logo_width_px = logo_height_fig_rel * fig_height_px * image_aspect_ratio
            logo_width_fig_rel = logo_width_px / fig_width_px
            rect_left_fig = margin_fig_rel
            rect_bottom_fig = 1.0 - margin_fig_rel - logo_height_fig_rel
            self.logo_ax_on_fig = self.fig.add_axes(
                [rect_left_fig, rect_bottom_fig, logo_width_fig_rel, logo_height_fig_rel],
                anchor='NW',
                zorder=20
            )
            self.logo_ax_on_fig.imshow(logo_numpy_array)
            self.logo_ax_on_fig.axis('off')
        except Exception as e:
            print(f"Error drawing logo on figure: {e}")
            import traceback
            traceback.print_exc()

class ManualInputWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Input for Nonlinear Polarization Tensor")
        self.setGeometry(150, 150, 1200, 800)

        # Main layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)

        # Left control panel
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.main_layout.addWidget(self.left_widget, 1)

        # Tensor input area
        self.tensor_box = QGroupBox("Nonlinear Polarization Tensor Input")
        self.tensor_layout = QVBoxLayout()
        self.tensor_box.setLayout(self.tensor_layout)

        # Instruction label
        self.instruction_label = QLabel("Enter d_ij components of the 3x6 matrix, or select a preset crystal:")
        self.tensor_layout.addWidget(self.instruction_label)

        # Preset crystal selection
        self.preset_box = QGroupBox("Preset Crystals")
        self.preset_layout = QVBoxLayout()
        self.preset_box.setLayout(self.preset_layout)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom", "MoS2 (2H)", "BaTiO3", "LiNbO3", "KDP"])
        self.preset_combo.currentIndexChanged.connect(self.load_preset)
        self.preset_layout.addWidget(self.preset_combo)

        self.tensor_layout.addWidget(self.preset_box)

        # Tensor input grid
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)

        # Create input fields for a 3x6 matrix
        self.tensor_inputs = {}

        # Add column labels
        for j in range(6):
            label = QLabel(f"{j+1}")
            label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(label, 0, j+1)

        # Add row labels and input fields
        for i in range(3):
            # Row label
            row_label = QLabel(f"d{i+1}j")
            row_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.grid_layout.addWidget(row_label, i+1, 0)

            # 6 input fields per row
            for j in range(6):
                input_field = QDoubleSpinBox()
                input_field.setRange(-100, 100)
                input_field.setDecimals(2)
                input_field.setSingleStep(0.1)
                input_field.setValue(0.0)
                input_field.setFixedWidth(80)
                self.grid_layout.addWidget(input_field, i+1, j+1)

                # Save reference - use (row, col) index
                self.tensor_inputs[(i, j)] = input_field

        # Add scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.grid_widget)
        self.tensor_layout.addWidget(scroll_area)

        self.left_layout.addWidget(self.tensor_box)

        # Button area
        self.buttons_widget = QWidget()
        self.buttons_layout = QHBoxLayout(self.buttons_widget)

        self.plot_button = QPushButton("Plot Polarization Intensity")
        self.plot_button.clicked.connect(self.plot)
        self.buttons_layout.addWidget(self.plot_button)

        self.back_button = QPushButton("Back to Slider Mode")
        self.back_button.clicked.connect(self.go_back)
        self.buttons_layout.addWidget(self.back_button)

        self.left_layout.addWidget(self.buttons_widget)

        # Right plotting area
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setContentsMargins(5, 5, 0, 0) # left, top, right, bottom margins
        self.main_layout.addWidget(self.plot_widget, 2)

        self.canvas = PlotCanvas(self, width=8, height=8)
        self.plot_layout.addWidget(self.canvas)

        # Initialize tensor and plot parameters
        self.phi = 0.0
        self.tensor = np.zeros((3, 3, 3))
        self.alpha = 0.0 # Initialize alpha for ManualInputWindow
        self.update_input_polarization_controls_manual() # Set initial visibility

        # Azimuthal angle control
        self.phi_control = QWidget()
        self.phi_layout = QHBoxLayout(self.phi_control)

        self.phi_label = QLabel("Azimuthal Angle φ (deg):")
        self.phi_label.setToolTip("The angle between the projection of the incident light propagation direction on the XY plane\nand the positive X-axis (0° to 359°). This defines the azimuth of the incident light.")

        self.phi_slider = QSlider(Qt.Horizontal)
        self.phi_slider.setMinimum(0)
        self.phi_slider.setMaximum(359)
        self.phi_slider.setValue(0)
        self.phi_slider.valueChanged.connect(self.update_phi)
        self.phi_layout.addWidget(self.phi_slider)

        self.phi_display = QLabel("0°")
        self.phi_layout.addWidget(self.phi_display)

        self.left_layout.addWidget(self.phi_control)

        # Input polarization mode selection (ManualInputWindow)
        self.input_polarization_control_widget = QWidget()
        self.input_polarization_layout = QHBoxLayout(self.input_polarization_control_widget)

        self.input_polarization_label_manual = QLabel("Input Polarization:")
        self.input_polarization_layout.addWidget(self.input_polarization_label_manual)

        self.input_polarization_combo_manual = QComboBox()
        self.input_polarization_combo_manual.addItems(["Default (θ-polarized)", "Linear Polarization", "Left-handed Circular (LCP)", "Right-handed Circular (RCP)"])
        self.input_polarization_combo_manual.currentIndexChanged.connect(self.update_input_polarization_controls_manual)
        self.input_polarization_layout.addWidget(self.input_polarization_combo_manual)
        self.left_layout.addWidget(self.input_polarization_control_widget)

        # Linear polarization angle alpha control (ManualInputWindow - initially hidden)
        self.alpha_control_widget_manual = QWidget()
        self.alpha_layout_manual = QHBoxLayout(self.alpha_control_widget_manual)
        self.alpha_label_manual = QLabel("Polarization Angle α (deg):")
        self.alpha_layout_manual.addWidget(self.alpha_label_manual)
        self.alpha_layout_manual.setContentsMargins(0,0,0,0) # Remove unnecessary margins

        self.alpha_slider_manual = QSlider(Qt.Horizontal)
        self.alpha_slider_manual.setMinimum(0)
        self.alpha_slider_manual.setMaximum(180)
        self.alpha_slider_manual.setValue(0)
        self.alpha_slider_manual.valueChanged.connect(self.update_alpha_display_manual)
        self.alpha_layout_manual.addWidget(self.alpha_slider_manual)

        self.alpha_display_manual = QLabel("0°")
        self.alpha_layout_manual.addWidget(self.alpha_display_manual)
        self.left_layout.addWidget(self.alpha_control_widget_manual)
        self.alpha_control_widget_manual.setVisible(False)

        # Detection polarization selection
        self.detection_widget = QWidget()
        self.detection_layout = QHBoxLayout(self.detection_widget)

        self.detection_label = QLabel("Detection Polarization:")
        self.detection_layout.addWidget(self.detection_label)

        self.detection_combo = QComboBox()
        self.detection_combo.addItems(["Total Intensity |P|²", "Parallel Mode (∥)", "Perpendicular Mode (⊥)"])
        self.detection_combo.currentIndexChanged.connect(self.plot)
        self.detection_layout.addWidget(self.detection_combo)

        self.left_layout.addWidget(self.detection_widget)

    def update_phi(self):
        """Update azimuthal angle"""
        self.phi = self.phi_slider.value()
        self.phi_display.setText(f"{self.phi}°")
        self.plot()

    def load_preset(self, index):
        """Load preset crystal data"""
        if index == 0:  # Custom
            return

        # Reset all input fields
        for input_field in self.tensor_inputs.values():
            input_field.setValue(0.0)

        if index == 1:  # MoS2 (2H)
            # Non-zero components for 2H-MoS2 (D3h point group)
            # https://doi.org/10.1103/PhysRevB.87.161403
            # d_31 = d_32 = 1.0
            self.tensor_inputs[(2, 0)].setValue(1.0)  # d_31
            self.tensor_inputs[(2, 1)].setValue(1.0)  # d_32
            self.tensor_inputs[(0, 4)].setValue(1.0)  # d_15
            self.tensor_inputs[(0, 4)].setValue(1.0)  # d_15
            self.tensor_inputs[(1, 3)].setValue(1.0)  # d_24
            self.tensor_inputs[(1, 3)].setValue(1.0)  # d_24

        elif index == 2:  # BaTiO3
            # Non-zero components for BaTiO3 (4mm point group)
            # Simplified values, actual values should be looked up in literature
            self.tensor_inputs[(2, 2)].setValue(10.0)  # d_33
            self.tensor_inputs[(2, 0)].setValue(3.0)   # d_31
            self.tensor_inputs[(2, 1)].setValue(3.0)   # d_32
            self.tensor_inputs[(0, 4)].setValue(3.0)   # d_15
            self.tensor_inputs[(1, 3)].setValue(3.0)   # d_24

        elif index == 3:  # LiNbO3
            # Non-zero components for LiNbO3 (3m point group)
            # Simplified values, actual values should be looked up in literature
            self.tensor_inputs[(2, 2)].setValue(20.0)  # d_33
            self.tensor_inputs[(2, 0)].setValue(5.0)   # d_31
            self.tensor_inputs[(2, 1)].setValue(5.0)   # d_32
            self.tensor_inputs[(0, 4)].setValue(4.0)   # d_15
            self.tensor_inputs[(1, 3)].setValue(4.0)   # d_24
            self.tensor_inputs[(1, 0)].setValue(-4.0)  # d_22 = -d_21
            self.tensor_inputs[(0, 5)].setValue(4.0)   # d_16

        elif index == 4:  # KDP
            # Non-zero components for KDP (42m point group)
            # Simplified values, actual values should be looked up in literature
            self.tensor_inputs[(0, 3)].setValue(1.0)   # d_14
            self.tensor_inputs[(1, 4)].setValue(1.0)   # d_25
            self.tensor_inputs[(2, 5)].setValue(1.0)   # d_36

        self.plot()

    def plot(self):
        """Plot the polarization intensity polar plot"""
        # Initialize d-matrix (3x6)
        d_matrix = np.zeros((3, 6))

        # Get tensor values from input fields
        for (i, j), input_field in self.tensor_inputs.items():
            d_matrix[i, j] = input_field.value()

        # Clear current figure
        self.canvas.axes.clear()

        # Plot new figure
        phi_rad = self.phi * np.pi/180  # Convert to radians
        theta_range = np.linspace(0, 2*np.pi, 360)

        # Calculate polarization intensity
        P = []
        for theta in theta_range:
            # Calculate incident electric field direction k_hat (spherical to Cartesian)
            k_hat = np.array([
                np.sin(theta)*np.cos(phi_rad),
                np.sin(theta)*np.sin(phi_rad),
                np.cos(theta)
            ])

            # Calculate actual electric field E_omega based on input polarization mode (ManualInputWindow)
            input_pol_mode = self.input_polarization_combo_manual.currentText()
            E_omega = np.zeros(3)
            alpha_rad = np.deg2rad(self.alpha)

            if input_pol_mode == "Linear Polarization":
                vec_theta = np.array([np.cos(theta) * np.cos(phi_rad),
                                      np.cos(theta) * np.sin(phi_rad),
                                      -np.sin(theta)])
                vec_phi = np.array([-np.sin(phi_rad),
                                    np.cos(phi_rad),
                                    0.0])
                E_omega = np.cos(alpha_rad) * vec_theta + np.sin(alpha_rad) * vec_phi

            elif input_pol_mode == "Left-handed Circular (LCP)":
                vec_theta = np.array([np.cos(theta) * np.cos(phi_rad),
                                      np.cos(theta) * np.sin(phi_rad),
                                      -np.sin(theta)])
                vec_phi = np.array([-np.sin(phi_rad),
                                    np.cos(phi_rad),
                                    0.0])
                E_omega = (1/np.sqrt(2)) * (vec_theta + 1j * vec_phi) # Complex E_omega

            elif input_pol_mode == "Right-handed Circular (RCP)":
                vec_theta = np.array([np.cos(theta) * np.cos(phi_rad),
                                      np.cos(theta) * np.sin(phi_rad),
                                      -np.sin(theta)])
                vec_phi = np.array([-np.sin(phi_rad),
                                    np.cos(phi_rad),
                                    0.0])
                E_omega = (1/np.sqrt(2)) * (vec_theta - 1j * vec_phi) # Complex E_omega

            else: # "Default" mode - p-polarized
                vec_theta = np.array([np.cos(theta) * np.cos(phi_rad),
                                      np.cos(theta) * np.sin(phi_rad),
                                      -np.sin(theta)])
                E_omega = vec_theta # Real E_omega
                # When mode is "Default (θ-polarized)", behavior is same as linear polarization with alpha=0
                E_omega = vec_theta

            # Construct 6-element Voigt vector [Ex², Ey², Ez², 2EyEz, 2ExEz, 2ExEy]
            Ex, Ey, Ez = E_omega[0], E_omega[1], E_omega[2] # These can be complex
            E_voigt = np.array([
                Ex*Ex, Ey*Ey, Ez*Ez, # E_i*E_j terms
                2*Ey*Ez, 2*Ex*Ez, 2*Ex*Ey
            ], dtype=np.complex128) # Ensure E_voigt is complex

            # Calculate P_i = d_il * E_voigt_l using matrix multiplication
            P_i = np.dot(d_matrix, E_voigt)

            # Get detection polarization selection
            detection_pol = self.detection_combo.currentIndex()

            if detection_pol == 0:  # Total Intensity |P|²
                intensity = np.linalg.norm(P_i)**2
            else:
                # Parallel and perpendicular mode detection - consider xy-plane only

                # Get the projection direction of the incident fundamental light E_omega in the xy-plane
                # E_omega is already the normalized incident electric field vector
                E_omega_xy_projection = np.array([Ex, Ey, 0.0]) # Ex, Ey are components of E_omega
                E_omega_xy_norm = np.linalg.norm(E_omega_xy_projection)

                # If E_omega_xy_projection is nearly a zero vector (E-field almost perpendicular to xy-plane), use unit x-vector as reference
                if E_omega_xy_norm < 1e-9: # Use a small threshold
                    E_incident_parallel_direction_xy = np.array([1.0, 0.0, 0.0])
                else:
                    # Normalize to get the direction vector in the xy-plane
                    E_incident_parallel_direction_xy = E_omega_xy_projection / E_omega_xy_norm

                # Calculate the component of the second harmonic P_i in the xy-plane
                P_xy = np.array([P_i[0], P_i[1], 0.0])

                if detection_pol == 1:  # Parallel Mode (∥) - SHG component in xy-plane parallel to E_omega's projection in xy-plane
                    # Calculate the square of the projection of P_xy onto E_incident_parallel_direction_xy
                    P_parallel_component_xy = np.dot(P_xy, E_incident_parallel_direction_xy)
                    intensity = np.abs(P_parallel_component_xy)**2
                elif detection_pol == 2:  # Perpendicular Mode (⊥) - SHG component in xy-plane perpendicular to E_omega's projection in xy-plane
                    # Unit vector in xy-plane perpendicular to E_incident_parallel_direction_xy
                    # Rotated 90 degrees counter-clockwise
                    E_incident_perpendicular_direction_xy = np.array([-E_incident_parallel_direction_xy[1], E_incident_parallel_direction_xy[0], 0.0])

                    # Calculate the square of the projection of P_xy onto the perpendicular direction
                    P_perpendicular_component_xy = np.dot(P_xy, E_incident_perpendicular_direction_xy)
                    intensity = np.abs(P_perpendicular_component_xy)**2

            P.append(intensity)

        # Draw the polar plot
        self.canvas.axes.plot(theta_range, P, lw=2, color='purple')

        # Get preset name
        preset_name = self.preset_combo.currentText()

        # Get detection polarization text description
        detection_text = self.detection_combo.currentText()

        # Get input polarization mode (ManualInputWindow)
        input_pol_mode = self.input_polarization_combo_manual.currentText()
        input_pol_text = f"Input: {input_pol_mode}"
        if input_pol_mode == "Linear Polarization":
            input_pol_text += f" (α={self.alpha}°)"

        # Update title to include detection polarization info
        title_text = f"{preset_name} - {detection_text}\n{input_pol_text}, φ={self.phi:.1f}°"

        self.canvas.axes.set_title(title_text, y=-0.15, va='top')
        self.canvas.axes.grid(True, linestyle='--', alpha=0.5)
        self.canvas.draw()

    def go_back(self):
        """Close current window and return to main window"""
        self.parent().show()
        self.close()

    def update_input_polarization_controls_manual(self):
        """Show or hide alpha control based on selected input polarization mode (ManualInputWindow)"""
        mode = self.input_polarization_combo_manual.currentText()
        is_linear_polarization = (mode == "Linear Polarization")
        self.alpha_control_widget_manual.setVisible(is_linear_polarization)

        if is_linear_polarization:
            self.alpha_label_manual.setToolTip("The angle between the electric field oscillation direction of linearly polarized light and the θ-direction unit vector.\nThe θ-direction vector corresponds to p-polarization, and the φ-direction corresponds to s-polarization.\nα=0° means E-field along θ-direction; α=90° means E-field along φ-direction.")
        else:
            self.alpha_label_manual.setToolTip("")

        self.plot() # Re-plot when mode changes

    def update_alpha_display_manual(self, update_plot=True):
        """Update alpha angle display (ManualInputWindow)"""
        try:
            self.alpha = self.alpha_slider_manual.value()
            self.alpha_display_manual.setText(f"{self.alpha}°")
            if update_plot:
                self.plot()
        except Exception as e:
            print(f"Error updating alpha display (ManualInputWindow): {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nonlinear Polarization Simulator")
        self.setGeometry(100, 100, 1200, 800)

        self.logo_qimage_for_plot = None
        try:
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                logo_path_main = os.path.join(sys._MEIPASS, 'IPE_logo.png')
            else:
                logo_path_main = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'IPE_logo.png')
            if os.path.exists(logo_path_main):
                pixmap = QPixmap(logo_path_main)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaledToHeight(120, Qt.SmoothTransformation)
                    self.logo_qimage_for_plot = scaled_pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
                else:
                    print("Warning: Failed to load IPE_logo.png into QPixmap.")
            else:
                print(f"Warning: IPE_logo.png not found at {logo_path_main}")
        except Exception as e:
            print(f"Error preparing logo QImage: {e}")
        self.phi = 0.0
        self.component_values = {}
        self.component_widgets = {}
        self.base_tensor = None
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        self.controls_scroll_area = QScrollArea()
        self.controls_scroll_area.setWidgetResizable(True)
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_scroll_area.setWidget(self.controls_widget)
        self.main_layout.addWidget(self.controls_scroll_area, 1)
        self.group_box = QGroupBox("Crystal Point Group")
        self.group_layout = QVBoxLayout()
        self.group_box.setLayout(self.group_layout)
        self.group_combo = QComboBox()
        self.group_combo.addItems(get_all_point_groups())
        self.group_combo.currentIndexChanged.connect(self.update_point_group)
        self.group_layout.addWidget(self.group_combo)
        self.components_label = QLabel("Non-zero Components:")
        self.group_layout.addWidget(self.components_label)
        self.components_list = QLabel()
        self.group_layout.addWidget(self.components_list)
        self.controls_layout.addWidget(self.group_box)
        self.common_crystal_box = QGroupBox("Common Crystal Presets")
        self.common_crystal_layout = QVBoxLayout()
        self.common_crystal_box.setLayout(self.common_crystal_layout)
        self.common_crystal_label = QLabel("Select Preset Crystal:")
        self.common_crystal_layout.addWidget(self.common_crystal_label)
        self.common_crystal_combo = QComboBox()
        self.common_crystal_combo.addItems(["Custom"] + list(COMMON_CRYSTALS.keys()))
        self.common_crystal_combo.currentIndexChanged.connect(self.load_common_crystal)
        self.common_crystal_layout.addWidget(self.common_crystal_combo)
        self.controls_layout.addWidget(self.common_crystal_box)
        self.params_box = QGroupBox("Parameters")
        self.params_layout = QGridLayout()
        self.params_box.setLayout(self.params_layout)
        self.scan_mode_label = QLabel("Scan Mode:")
        self.params_layout.addWidget(self.scan_mode_label, 0, 0)
        self.scan_mode_combo = QComboBox()
        self.scan_mode_combo.addItems(["Incident Angle Scan (θ-polar)",
                                       "Polarization Angle Scan (α vs I)",
                                       "Polarization Angle Scan (α-polar)",
                                       "3D θ-α Scan"])
        self.scan_mode_combo.setCurrentIndex(2) # Default to "Polarization Angle Scan (α-polar)"
        self.scan_mode_combo.currentIndexChanged.connect(self.update_scan_mode_controls)
        self.params_layout.addWidget(self.scan_mode_combo, 0, 1, 1, 2)
        self.fixed_theta_label = QLabel("Fixed Incident Angle θ (deg):")
        self.params_layout.addWidget(self.fixed_theta_label, 1, 0)
        self.fixed_theta_spinbox = QDoubleSpinBox()
        self.fixed_theta_spinbox.setRange(0, 180) # 0 (normal) to 180
        self.fixed_theta_spinbox.setValue(0.0) # Default normal incidence
        self.fixed_theta_spinbox.setSuffix("°")
        self.fixed_theta_spinbox.valueChanged.connect(self.plot) # Redraw on value change
        self.params_layout.addWidget(self.fixed_theta_spinbox, 1, 1, 1, 2)
        self.alpha_scan_phi_inc_label = QLabel("Fixed Beam Azimuth φ<sub>inc</sub> (deg):")
        self.params_layout.addWidget(self.alpha_scan_phi_inc_label, 2, 0)
        self.alpha_scan_phi_inc_spinbox = QDoubleSpinBox()
        self.alpha_scan_phi_inc_spinbox.setRange(0, 359)
        self.alpha_scan_phi_inc_spinbox.setValue(0.0)
        self.alpha_scan_phi_inc_spinbox.setSuffix("°")
        self.alpha_scan_phi_inc_spinbox.valueChanged.connect(self.plot)
        self.params_layout.addWidget(self.alpha_scan_phi_inc_spinbox, 2, 1, 1, 2)
        self.euler_phi_label = QLabel("Crystal φc (Z rot, deg):")
        self.params_layout.addWidget(self.euler_phi_label, 3, 0)
        self.euler_phi_spinbox = QDoubleSpinBox()
        self.euler_phi_spinbox.setRange(0, 360)
        self.euler_phi_spinbox.setValue(0.0)
        self.euler_phi_spinbox.setSuffix("°")
        self.euler_phi_spinbox.valueChanged.connect(self.plot)
        self.params_layout.addWidget(self.euler_phi_spinbox, 3, 1, 1, 2)
        self.euler_theta_label = QLabel("Crystal θc (Y' rot, deg):")
        self.params_layout.addWidget(self.euler_theta_label, 4, 0)
        self.euler_theta_spinbox = QDoubleSpinBox()
        self.euler_theta_spinbox.setRange(0, 180)
        self.euler_theta_spinbox.setValue(0.0)
        self.euler_theta_spinbox.setSuffix("°")
        self.euler_theta_spinbox.valueChanged.connect(self.plot)
        self.params_layout.addWidget(self.euler_theta_spinbox, 4, 1, 1, 2)
        self.euler_psi_label = QLabel("Crystal ψc (Z'' rot, deg):")
        self.params_layout.addWidget(self.euler_psi_label, 5, 0)
        self.euler_psi_spinbox = QDoubleSpinBox()
        self.euler_psi_spinbox.setRange(0, 360)
        self.euler_psi_spinbox.setValue(0.0)
        self.euler_psi_spinbox.setSuffix("°")
        self.euler_psi_spinbox.valueChanged.connect(self.plot)
        self.params_layout.addWidget(self.euler_psi_spinbox, 5, 1, 1, 2)
        self.phi_label = QLabel("Beam Azimuth φ (deg):")
        self.phi_label.setToolTip("The angle between the projection of the incident light propagation direction on the XY plane and the positive X-axis (0° to 359°).\nThis defines the azimuth of the incident light (for incident angle scan mode only).")
        self.params_layout.addWidget(self.phi_label, 6, 0)
        self.phi_slider = QSlider(Qt.Horizontal)
        self.phi_slider.setMinimum(0)
        self.phi_slider.setMaximum(359)
        self.phi_slider.setValue(0)
        self.phi_slider.valueChanged.connect(self.update_phi_display)
        self.params_layout.addWidget(self.phi_slider, 6, 1)
        self.phi_display = QLabel("0°")
        self.params_layout.addWidget(self.phi_display, 6, 2)
        self.tensor_label = QLabel("Overall Tensor Intensity:")
        self.params_layout.addWidget(self.tensor_label, 7, 0)
        self.tensor_slider = QSlider(Qt.Horizontal)
        self.tensor_slider.setMinimum(1)
        self.tensor_slider.setMaximum(20)
        self.tensor_slider.setValue(10)
        self.tensor_slider.valueChanged.connect(self.update_tensor_display)
        self.params_layout.addWidget(self.tensor_slider, 7, 1)
        self.tensor_display = QLabel("1.0")
        self.params_layout.addWidget(self.tensor_display, 7, 2)
        self.input_polarization_label = QLabel("Input Polarization:")
        self.params_layout.addWidget(self.input_polarization_label, 8, 0)
        self.input_polarization_combo = QComboBox()
        self.input_polarization_combo.addItems(["Default (θ-polarized)", "Linear Polarization", "Left-handed Circular (LCP)", "Right-handed Circular (RCP)"])
        self.input_polarization_combo.currentIndexChanged.connect(self.update_input_polarization_controls)
        self.params_layout.addWidget(self.input_polarization_combo, 8, 1, 1, 2)
        self.alpha_control_widget = QWidget()
        self.alpha_control_layout = QHBoxLayout(self.alpha_control_widget)
        self.alpha_control_layout.setContentsMargins(0,0,0,0)
        self.alpha_label = QLabel("Polarization Angle α (deg):")
        self.alpha_control_layout.addWidget(self.alpha_label)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(359)
        self.alpha_slider.setValue(0)
        self.alpha_slider.valueChanged.connect(self.update_alpha_display)
        self.alpha_control_layout.addWidget(self.alpha_slider)
        self.alpha_display = QLabel("0°")
        self.alpha_control_layout.addWidget(self.alpha_display)
        self.params_layout.addWidget(self.alpha_control_widget, 9, 0, 1, 3)
        self.alpha_control_widget.setVisible(False)
        self.detection_label = QLabel("Detection Polarization:")
        self.params_layout.addWidget(self.detection_label, 10, 0)
        self.detection_combo = QComboBox()
        self.detection_combo.addItems(["Total Intensity |P|²", "Parallel Mode (∥)", "Perpendicular Mode (⊥)"])
        self.detection_combo.currentIndexChanged.connect(self.plot)
        self.params_layout.addWidget(self.detection_combo, 10, 1, 1, 2)
        self.controls_layout.addWidget(self.params_box)
        self.component_box = QGroupBox("Component-wise Adjustment")
        self.component_layout = QVBoxLayout()
        self.component_box.setLayout(self.component_layout)
        self.controls_layout.addWidget(self.component_box)
        self.buttons_widget = QWidget()
        self.buttons_layout = QHBoxLayout(self.buttons_widget)
        self.plot_button = QPushButton("Plot Polarization Intensity")
        self.plot_button.clicked.connect(self.plot)
        self.buttons_layout.addWidget(self.plot_button)
        self.reset_button = QPushButton("Reset Parameters")
        self.reset_button.clicked.connect(self.reset_params)
        self.buttons_layout.addWidget(self.reset_button)
        self.manual_input_button = QPushButton("Manual Input Mode")
        self.manual_input_button.clicked.connect(self.open_manual_input)
        self.buttons_layout.addWidget(self.manual_input_button)
        self.plot_3d_button = QPushButton("Plot 3D SHG Pattern")
        self.plot_3d_button.clicked.connect(self.open_3d_plot_window)
        self.buttons_layout.addWidget(self.plot_3d_button)
        self.save_image_button = QPushButton("Save Image")
        self.save_image_button.clicked.connect(self.save_current_plot_image)
        self.buttons_layout.addWidget(self.save_image_button)
        self.controls_layout.addWidget(self.buttons_widget)
        self.import_experiment_button = QPushButton("导入实验数据 / 提取 2ω 强度")
        self.import_experiment_button.clicked.connect(self.open_experimental_import)
        self.controls_layout.addWidget(self.import_experiment_button)
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setContentsMargins(5, 5, 0, 0)
        self.main_layout.addWidget(self.plot_widget, 2)
        self.canvas = PlotCanvas(self, width=8, height=8)
        self.plot_layout.addWidget(self.canvas)
        self.selected_group = self.group_combo.currentText()
        if self.selected_group:
            self.update_point_group(0)
            self.reset_params(False)
        self.alpha = 0
        self.update_input_polarization_controls()
        self.update_scan_mode_controls()
        QApplication.processEvents()
        self.plot()
        self.show_symbolic_button = QPushButton("Show Symbolic Expressions")
        self.show_symbolic_button.setFixedHeight(40)
        self.show_symbolic_button.clicked.connect(self.handle_show_symbolic_expressions)
        self.controls_layout.addWidget(self.show_symbolic_button)

    def open_experimental_import(self):
        from ui.experimental_import import show_experimental_import
        show_experimental_import(self)

    def _calculate_lab_frame_d_matrix(self):
        from core.tensor_operations import TensorOperations
        tensor = create_tensor_with_relations(self.selected_group, values=self.component_values)
        tensor = tensor * (self.tensor_slider.value()/10.0)
        rotation = TensorOperations.get_rotation_matrix(self.euler_phi_spinbox.value(), self.euler_theta_spinbox.value(), self.euler_psi_spinbox.value())
        return TensorOperations.tensor_to_voigt(TensorOperations.rotate_tensor_vectorized(tensor, rotation))
    def update_point_group(self, index, called_by_common_crystal_load=False):
        try:
            if self.group_combo.count() == 0:
                return
            self.selected_group = self.group_combo.currentText()
            components = get_components_for_group(self.selected_group)
            comp_str = []
            for c in components:
                indices = str_to_indices(c)
                i, j, k = indices
                voigt_map = {
                    (0, 0): 1, (1, 1): 2, (2, 2): 3,
                    (1, 2): 4, (2, 1): 4, (0, 2): 5,
                    (2, 0): 5, (0, 1): 6, (1, 0): 6
                }
                l = voigt_map.get((j, k), 0)
                if l > 0:
                    comp_str.append(f"d{i+1}{l}")
                else:
                    comp_str.append(f"d{i+1}{j+1}{k+1}")
            self.components_list.setText(", ".join(comp_str))
            self.base_tensor = create_tensor_with_relations(self.selected_group)
            self.component_values = {comp: 1.0 for comp in components}
            self.clear_component_sliders()
            self.create_component_sliders(components)
            if not called_by_common_crystal_load:
                if self.common_crystal_combo.currentText() != "Custom":
                    self.common_crystal_combo.blockSignals(True)
                    self.common_crystal_combo.setCurrentIndex(0)
                    self.common_crystal_combo.blockSignals(False)
            self.plot()
        except Exception as e:
            print(f"Error updating point group: {e}")
    def clear_component_sliders(self):
        try:
            for i in reversed(range(self.component_layout.count())):
                item = self.component_layout.itemAt(i)
                if item is None:
                    continue
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                layout = item.layout()
                if layout is not None:
                    while layout.count():
                        layout_item = layout.takeAt(0)
                        if layout_item.widget():
                            layout_item.widget().deleteLater()
                    self.component_layout.removeItem(layout)
            self.component_widgets = {}
        except Exception as e:
            print(f"Error clearing component sliders: {e}")
    def create_component_sliders(self, components):
        try:
            if len(components) > 6:
                grid_layout = QGridLayout()
                self.component_layout.addLayout(grid_layout)
                row, col = 0, 0
                max_cols = 2
                for comp in components:
                    widget = QWidget()
                    layout = QVBoxLayout(widget)
                    indices = str_to_indices(comp)
                    i, j, k = indices
                    voigt_map = {
                        (0, 0): 1, (1, 1): 2, (2, 2): 3,
                        (1, 2): 4, (2, 1): 4, (0, 2): 5,
                        (2, 0): 5, (0, 1): 6, (1, 0): 6
                    }
                    l = voigt_map.get((j, k), 0)
                    if l > 0:
                        label = QLabel(f"d{i+1}{l}:")
                    else:
                        label = QLabel(f"d{i+1}{j+1}{k+1}:")
                    label.setAlignment(Qt.AlignCenter)
                    layout.addWidget(label)
                    slider_layout = QHBoxLayout()
                    slider = ClickableSlider()
                    slider.setMinimum(-1000)
                    slider.setMaximum(1000)
                    slider.setValue(100)
                    slider.setTickPosition(QSlider.TicksBelow)
                    slider.setTickInterval(200)
                    slider.setMinimumWidth(150)
                    slider.setProperty("component", comp)
                    slider.valueChanged.connect(self.update_component_value_slider)
                    def create_reset_handler(component=comp):
                        return lambda: self.reset_component(component)
                    slider.doubleClicked.connect(create_reset_handler())
                    value_label = QLabel("1.00")
                    value_label.setAlignment(Qt.AlignCenter)
                    value_label.setMinimumWidth(50)
                    slider_layout.addWidget(slider)
                    slider_layout.addWidget(value_label)
                    layout.addLayout(slider_layout)
                    self.component_widgets[comp] = {
                        'slider': slider,
                        'label': value_label
                    }
                    grid_layout.addWidget(widget, row, col)
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
            else:
                for comp in components:
                    widget = QWidget()
                    layout = QHBoxLayout(widget)
                    label = QLabel(f"χ{comp}:")
                    label.setMinimumWidth(40)
                    layout.addWidget(label)
                    slider = ClickableSlider()
                    slider.setMinimum(-1000)
                    slider.setMaximum(1000)
                    slider.setValue(100)
                    slider.setTickPosition(QSlider.TicksBelow)
                    slider.setTickInterval(200)
                    slider.setMinimumWidth(180)
                    slider.setProperty("component", comp)
                    slider.valueChanged.connect(self.update_component_value_slider)
                    def create_reset_handler(component=comp):
                        return lambda: self.reset_component(component)
                    slider.doubleClicked.connect(create_reset_handler())
                    value_label = QLabel("1.00")
                    value_label.setMinimumWidth(50)
                    value_label.setAlignment(Qt.AlignCenter)
                    layout.addWidget(slider)
                    layout.addWidget(value_label)
                    self.component_widgets[comp] = {
                        'slider': slider,
                        'label': value_label
                    }
                    self.component_layout.addWidget(widget)
        except Exception as e:
            print(f"Error creating component sliders: {e}")
    def update_component_value_slider(self):
        try:
            sender = self.sender()
            if not sender:
                return
            comp = sender.property("component")
            if not comp:
                return
            raw_value = sender.value()
            value = raw_value / 100.0
            if comp in self.component_widgets and 'label' in self.component_widgets[comp]:
                self.component_widgets[comp]['label'].setText(f"{value:.2f}")
                self.component_values[comp] = value
                self.plot()
        except Exception as e:
            print(f"Error updating component value: {e}")
    def update_phi_display(self, update_plot=True):
        try:
            self.phi = self.phi_slider.value()
            self.phi_display.setText(f"{self.phi}°")
            if update_plot:
                self.plot()
        except Exception as e:
            print(f"Error updating angle display: {e}")
    def update_tensor_display(self, update_plot=True):
        try:
            value = self.tensor_slider.value() / 10.0
            self.tensor_display.setText(f"{value:.1f}")
            if update_plot:
                self.plot()
        except Exception as e:
            print(f"Error updating tensor intensity display: {e}")
    def reset_params(self, update_plot=True):
        try:
            if not hasattr(self, 'phi_slider') or not hasattr(self, 'tensor_slider'):
                return
            self.phi_slider.setValue(0)
            self.tensor_slider.setValue(10)
            for comp, widgets in self.component_widgets.items():
                if 'slider' in widgets:
                    widgets['slider'].setValue(100)
                if 'label' in widgets:
                    widgets['label'].setText("1.00")
                self.component_values[comp] = 1.0
            self.update_phi_display(False)
            self.update_tensor_display(False)
            if update_plot:
                self.plot()
        except Exception as e:
            print(f"Error resetting parameters: {e}")
    def reset_component(self, component):
        try:
            if component in self.component_widgets and 'slider' in self.component_widgets[component]:
                self.component_widgets[component]['slider'].setValue(100)
                if 'label' in self.component_widgets[component]:
                    self.component_widgets[component]['label'].setText("1.00")
                self.component_values[component] = 1.0
                self.plot()
        except Exception as e:
            print(f"Error resetting component {component}: {e}")
    def plot(self):
        if not getattr(self, 'selected_group', None):
            return
        from core.shg_calculator import SHGCalculator, ScanParameters, PolarizationMode, DetectionMode
        try:
            text = self.scan_mode_combo.currentText()
            alpha_scan = text.startswith('Polarization Angle Scan')
            cartesian = '(α vs I)' in text
            scan = np.linspace(0, 2*np.pi, 361)
            theta = np.full_like(scan, np.deg2rad(self.fixed_theta_spinbox.value())) if alpha_scan else scan
            phi = np.deg2rad(self.alpha_scan_phi_inc_spinbox.value() if alpha_scan else self.phi_slider.value())
            mode_text = self.input_polarization_combo.currentText().lower()
            mode = PolarizationMode.LINEAR if alpha_scan or 'linear' in mode_text else PolarizationMode.LCP if 'lcp' in mode_text else PolarizationMode.RCP if 'rcp' in mode_text else PolarizationMode.DEFAULT
            params = ScanParameters(theta, phi, scan if alpha_scan else np.deg2rad(self.alpha_slider.value()), mode, DetectionMode(self.detection_combo.currentIndex()))
            intensity = SHGCalculator(self._calculate_lab_frame_d_matrix()).calculate_intensity_vectorized(params)
            self.canvas.fig.clear()
            self.canvas.logo_ax_on_fig = None
            self.canvas.axes = self.canvas.fig.add_subplot(111, polar=not cartesian)
            self.canvas.axes.plot(np.rad2deg(scan) if cartesian else scan, intensity, color='purple')
            self.canvas.axes.grid(True, alpha=0.3)
            self.canvas.axes.set_ylabel('Relative transverse intensity')
            if cartesian:
                self.canvas.axes.set_xlabel('Polarization angle (deg)')
            self.current_plot_data_x, self.current_plot_data_y = scan, intensity
            self.current_plot_data_type = 'alpha_cartesian' if cartesian else 'alpha_polar' if alpha_scan else 'theta_polar'
            self.current_plot_labels = ('Angle (rad)', 'Intensity')
            self.update_plot_title()
            self.canvas.draw_idle()
        except Exception as error:
            self.canvas.axes.set_title(str(error))
            self.canvas.draw_idle()
    def update_plot_title(self):
        display_name = self.selected_group
        display_name = display_name.replace("₁", "1").replace("₂", "2").replace("₃", "3")
        display_name = display_name.replace("₄", "4").replace("₆", "6")
        display_name = display_name.replace("ᵥ", "v").replace("ₕ", "h").replace("ₘ", "m")
        display_name = display_name.replace("ᵤ", "u").replace("ₐ", "a").replace("ᵢ", "i")
        display_name = display_name.replace("̄", "")
        detection_text = self.detection_combo.currentText()
        scan_mode = self.scan_mode_combo.currentText()
        phi_c = self.euler_phi_spinbox.value()
        theta_c = self.euler_theta_spinbox.value()
        psi_c = self.euler_psi_spinbox.value()
        crystal_orientation_text = f"Crystal Orientation: φc={phi_c:.1f}°, θc={theta_c:.1f}°, ψc={psi_c:.1f}°"
        if scan_mode == "Polarization Angle Scan (α vs I)":
            fixed_theta_val = self.fixed_theta_spinbox.value()
            fixed_phi_inc_val = self.alpha_scan_phi_inc_spinbox.value()
            title_text = f"{display_name} - {detection_text}\nPolarization Angle Scan (Cartesian) @ Beam θinc={fixed_theta_val:.1f}°, φinc={fixed_phi_inc_val:.1f}°\n{crystal_orientation_text}"
        elif scan_mode == "Polarization Angle Scan (α-polar)":
            fixed_theta_val = self.fixed_theta_spinbox.value()
            fixed_phi_inc_val = self.alpha_scan_phi_inc_spinbox.value()
            title_text = f"{display_name} - {detection_text}\nPolarization Angle Scan (Polar) @ Beam θinc={fixed_theta_val:.1f}°, φinc={fixed_phi_inc_val:.1f}°\n{crystal_orientation_text}"
        else:
            input_pol_mode = self.input_polarization_combo.currentText()
            input_pol_text = f"Input: {input_pol_mode}"
            if input_pol_mode == "Linear Polarization" or input_pol_mode == "Default (θ-polarized)":
                if self.input_polarization_combo.isVisible():
                    input_pol_text += f" (α={self.alpha}°)"
            title_text = f"{display_name} - {detection_text}\n{input_pol_text}, Beam φ={self.phi:.1f}°\n{crystal_orientation_text}"
        if hasattr(self.canvas, 'axes') and self.canvas.axes is not None:
            self.canvas.axes.set_title(title_text, y=-0.15, va='top')
        else:
            print("Warning: canvas.axes not available for title setting.")
    def update_scan_mode_controls(self):
        mode = self.scan_mode_combo.currentText()
        is_alpha_scan_mode_cartesian = (mode == "Polarization Angle Scan (α vs I)")
        is_alpha_scan_mode_polar = (mode == "Polarization Angle Scan (α-polar)")
        is_theta_alpha_3d_scan_mode = (mode == "3D θ-α Scan")
        is_any_alpha_scan_mode = is_alpha_scan_mode_cartesian or is_alpha_scan_mode_polar
        self.fixed_theta_label.setVisible(is_any_alpha_scan_mode and not is_theta_alpha_3d_scan_mode)
        self.fixed_theta_spinbox.setVisible(is_any_alpha_scan_mode and not is_theta_alpha_3d_scan_mode)
        self.alpha_scan_phi_inc_label.setVisible(is_any_alpha_scan_mode or is_theta_alpha_3d_scan_mode)
        self.alpha_scan_phi_inc_spinbox.setVisible(is_any_alpha_scan_mode or is_theta_alpha_3d_scan_mode)
        self.phi_label.setVisible(not is_any_alpha_scan_mode)
        self.phi_slider.setVisible(not is_any_alpha_scan_mode)
        self.phi_display.setVisible(not is_any_alpha_scan_mode)
        self.input_polarization_label.setVisible(not is_any_alpha_scan_mode)
        self.input_polarization_combo.setVisible(not is_any_alpha_scan_mode)
        if not is_any_alpha_scan_mode:
            self.update_input_polarization_controls()
        else:
            self.alpha_control_widget.setVisible(False)
        self.plot()
    def update_input_polarization_controls(self):
        mode = self.input_polarization_combo.currentText()
        is_linear_polarization = (mode == "Linear Polarization")
        self.alpha_control_widget.setVisible(is_linear_polarization)
        if is_linear_polarization:
            self.alpha_label.setToolTip("Angle between the E-field of linearly polarized light and the θ-direction unit vector.\nθ-direction (p-pol) is in the plane of incidence. φ-direction (s-pol) is perpendicular to it.\nα=0°: E-field along θ-direction; α=90°: E-field along φ-direction.")
        else:
            self.alpha_label.setToolTip("")
        self.plot()
    def update_alpha_display(self, update_plot=True):
        try:
            self.alpha = self.alpha_slider.value()
            self.alpha_display.setText(f"{self.alpha}°")
            if update_plot:
                self.plot()
        except Exception as e:
            print(f"Error updating alpha display: {e}")
    def open_3d_plot_window(self):
        if not hasattr(self, 'base_tensor') or self.base_tensor is None:
            print("Base tensor not initialized, cannot plot 3D figure.")
            return
        current_scan_mode_for_3d = self.scan_mode_combo.currentText()
        if current_scan_mode_for_3d == "3D θ-α Scan":
            self._plot_3d_theta_alpha_scan()
        else:
            self._plot_3d_shg_radiation_pattern()
    def _plot_3d_shg_radiation_pattern(self):
        d_matrix = self._calculate_lab_frame_d_matrix()
        if d_matrix is None:
            print("Cannot calculate d_matrix, unable to plot 3D figure.")
            return
        if self.fixed_theta_spinbox.isVisible():
            theta_inc_rad = np.deg2rad(self.fixed_theta_spinbox.value())
            phi_inc_rad = np.deg2rad(self.alpha_scan_phi_inc_spinbox.value())
            current_input_pol_mode = "Linear Polarization"
            alpha_inc_rad = np.deg2rad(0)
        elif self.input_polarization_combo.isVisible():
            theta_inc_rad = np.deg2rad(self.fixed_theta_spinbox.value())
            phi_inc_rad = np.deg2rad(self.phi_slider.value())
            current_input_pol_mode = self.input_polarization_combo.currentText()
            alpha_inc_rad = np.deg2rad(self.alpha_slider.value())
        else:
            theta_inc_rad = np.deg2rad(0)
            phi_inc_rad = np.deg2rad(0)
            current_input_pol_mode = "Default (θ-polarized)"
            alpha_inc_rad = np.deg2rad(0)
        vec_theta_inc = np.array([
            np.cos(theta_inc_rad) * np.cos(phi_inc_rad),
            np.cos(theta_inc_rad) * np.sin(phi_inc_rad),
            -np.sin(theta_inc_rad)
        ])
        vec_phi_inc = np.array([
            -np.sin(phi_inc_rad),
            np.cos(phi_inc_rad),
            0.0
        ])
        E_omega = np.zeros(3, dtype=np.complex128)
        if current_input_pol_mode == "Linear Polarization":
            E_omega = np.cos(alpha_inc_rad) * vec_theta_inc + np.sin(alpha_inc_rad) * vec_phi_inc
        elif current_input_pol_mode == "Left-handed Circular (LCP)":
            E_omega = (1/np.sqrt(2)) * (vec_theta_inc + 1j * vec_phi_inc)
        elif current_input_pol_mode == "Right-handed Circular (RCP)":
            E_omega = (1/np.sqrt(2)) * (vec_theta_inc - 1j * vec_phi_inc)
        else:
            E_omega = vec_theta_inc
        Ex, Ey, Ez = E_omega[0], E_omega[1], E_omega[2]
        E_voigt = np.array([
            Ex*Ex, Ey*Ey, Ez*Ez,
            2*Ey*Ez, 2*Ez*Ex, 2*Ex*Ey
        ], dtype=np.complex128)
        P_induced_2w = np.dot(d_matrix, E_voigt)
        theta_out_scan = np.linspace(0, np.pi, 91)
        phi_out_scan = np.linspace(0, 2 * np.pi, 181)
        theta_out_grid, phi_out_grid = np.meshgrid(theta_out_scan, phi_out_scan)
        R_shg_grid = np.zeros_like(theta_out_grid, dtype=float)
        for i, th_o in enumerate(theta_out_scan):
            for j, ph_o in enumerate(phi_out_scan):
                k_shg_hat = np.array([np.sin(th_o)*np.cos(ph_o), np.sin(th_o)*np.sin(ph_o), np.cos(th_o)])
                P_dot_k_shg = np.dot(P_induced_2w, k_shg_hat)
                P_effective_rad = P_induced_2w - P_dot_k_shg * k_shg_hat
                intensity_at_angle = np.linalg.norm(P_effective_rad)**2
                R_shg_grid[j, i] = intensity_at_angle
        if not hasattr(self, 'shg_3d_plot_window_instance') or not self.shg_3d_plot_window_instance.isVisible():
            self.shg_3d_plot_window_instance = SHG3DPlotWindow(self)
        self.shg_3d_plot_window_instance.plot_data(R_shg_grid, theta_out_grid, phi_out_grid, plot_type="radiation_pattern")
        self.shg_3d_plot_window_instance.show()
        self.shg_3d_plot_window_instance.activateWindow()
        self.shg_3d_plot_window_instance.raise_()
    def _plot_3d_theta_alpha_scan(self):
        print("Starting 3D Theta-Alpha Scan...")
        d_matrix = self._calculate_lab_frame_d_matrix()
        if d_matrix is None: return
        phi_inc_fixed_rad = np.deg2rad(self.alpha_scan_phi_inc_spinbox.value())
        detection_mode_idx = self.detection_combo.currentIndex()
        print(f"Inside _plot_3d_theta_alpha_scan: detection_mode_idx = {detection_mode_idx} (0=Total, 1=Para, 2=Perp), Selected text: {self.detection_combo.currentText()}")
        theta_inc_scan = np.linspace(0, np.pi, 91)
        alpha_inc_scan = np.linspace(0, 2 * np.pi, 73)
        theta_inc_grid, alpha_inc_grid = np.meshgrid(theta_inc_scan, alpha_inc_scan)
        intensity_grid = np.zeros_like(theta_inc_grid)
        for i_th, th_i in enumerate(theta_inc_scan):
            vec_theta_inc_basis = np.array([np.cos(th_i) * np.cos(phi_inc_fixed_rad),
                                            np.cos(th_i) * np.sin(phi_inc_fixed_rad),
                                            -np.sin(th_i)])
            vec_phi_inc_basis = np.array([-np.sin(phi_inc_fixed_rad), np.cos(phi_inc_fixed_rad), 0.0])
            for i_al, al_i in enumerate(alpha_inc_scan):
                E_omega = np.cos(al_i) * vec_theta_inc_basis + np.sin(al_i) * vec_phi_inc_basis
                Ex, Ey, Ez = E_omega[0], E_omega[1], E_omega[2]
                E_voigt = np.array([Ex*Ex, Ey*Ey, Ez*Ez, 2*Ey*Ez, 2*Ex*Ez, 2*Ex*Ey], dtype=np.complex128)
                P_i = np.dot(d_matrix, E_voigt)
                current_intensity = 0.0
                if detection_mode_idx == 0:
                    current_intensity = np.linalg.norm(P_i)**2
                else:
                    E_omega_xy_projection = np.array([Ex, Ey, 0.0])
                    E_omega_xy_norm = np.linalg.norm(E_omega_xy_projection)
                    E_par_dir_xy = np.array([1.0,0.0,0.0]) if E_omega_xy_norm < 1e-9 else E_omega_xy_projection / E_omega_xy_norm
                    P_xy_complex = np.array([P_i[0], P_i[1], 0.0])
                    if detection_mode_idx == 1:
                        P_par_comp = np.dot(P_xy_complex, E_par_dir_xy)
                        current_intensity = np.abs(P_par_comp)**2
                    elif detection_mode_idx == 2:
                        E_perp_dir_xy = np.array([-E_par_dir_xy[1], E_par_dir_xy[0], 0.0])
                        P_perp_comp = np.dot(P_xy_complex, E_perp_dir_xy)
                        current_intensity = np.abs(P_perp_comp)**2
                intensity_grid[i_al, i_th] = current_intensity
        if not hasattr(self, 'shg_3d_plot_window_instance') or not self.shg_3d_plot_window_instance.isVisible():
            self.shg_3d_plot_window_instance = SHG3DPlotWindow(self)
        self.shg_3d_plot_window_instance.plot_data(intensity_grid, theta_inc_grid, alpha_inc_grid, plot_type="theta_alpha_scan")
        self.shg_3d_plot_window_instance.show()
        self.shg_3d_plot_window_instance.activateWindow()
        self.shg_3d_plot_window_instance.raise_()
    def open_manual_input(self):
        if not hasattr(self, 'manual_window_instance') or not self.manual_window_instance.isVisible():
            self.manual_window_instance = ManualInputWindow(self)
            self.manual_window_instance.show()
            self.hide()
        else:
            self.manual_window_instance.activateWindow()
            self.manual_window_instance.raise_()
            self.hide()
    def load_common_crystal(self):
        selected_crystal_name = self.common_crystal_combo.currentText()
        if selected_crystal_name == "Custom":
            return
        if selected_crystal_name in COMMON_CRYSTALS:
            crystal_data = COMMON_CRYSTALS[selected_crystal_name]
            point_group_name = crystal_data["point_group"]
            coeffs = crystal_data["coeffs"]
            try:
                idx = self.group_combo.findText(point_group_name)
                if idx != -1:
                    self.group_combo.blockSignals(True)
                    self.group_combo.setCurrentIndex(idx)
                    self.group_combo.blockSignals(False)
                    self.update_point_group(idx, called_by_common_crystal_load=True)
                else:
                    print(f"Error: Preset crystal's point group '{point_group_name}' not found in the list.")
                    return
            except Exception as e:
                print(f"Error setting point group: {e}")
                return
            current_independent_components = get_components_for_group(self.selected_group)
            for comp_name in current_independent_components:
                value = coeffs.get(comp_name, 0.0)
                self.component_values[comp_name] = value
                if comp_name in self.component_widgets:
                    slider = self.component_widgets[comp_name]['slider']
                    label = self.component_widgets[comp_name]['label']
                    slider_val = int(round(value * 100))
                    slider_val = max(slider.minimum(), min(slider.maximum(), slider_val))
                    slider.setValue(slider_val)
                    label.setText(f"{value:.2f}")
            self.euler_phi_spinbox.setValue(0.0)
            self.euler_theta_spinbox.setValue(0.0)
            self.euler_psi_spinbox.setValue(0.0)
            self.tensor_slider.setValue(10)
            self.plot()
    def save_current_plot_image(self):
        if not hasattr(self.canvas, 'fig') or self.canvas.fig is None:
            print("No image to save.")
            return
        try:
            point_group = self.selected_group.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "").replace("/", "")
            scan_mode_text = self.scan_mode_combo.currentText()
            detection_mode = self.detection_combo.currentText().split(" ")[0]
            filename_parts = ["SHG", point_group, detection_mode]
            if "Incident Angle Scan" in scan_mode_text:
                filename_parts.append("ThetaScan")
                phi_val = self.phi_slider.value()
                filename_parts.append(f"phi{phi_val}")
                input_pol_mode = self.input_polarization_combo.currentText()
                if "Linear" in input_pol_mode:
                    alpha_val = self.alpha_slider.value()
                    filename_parts.append(f"alpha{alpha_val}")
                elif "LCP" in input_pol_mode:
                    filename_parts.append("LCP")
                elif "RCP" in input_pol_mode:
                    filename_parts.append("RCP")
            elif "Polarization Angle Scan (α vs I)" in scan_mode_text:
                filename_parts.append("AlphaScan_vs_I")
                theta_inc = self.fixed_theta_spinbox.value()
                phi_inc = self.alpha_scan_phi_inc_spinbox.value()
                filename_parts.append(f"th{theta_inc:.0f}_ph{phi_inc:.0f}")
            elif "Polarization Angle Scan (α-polar)" in scan_mode_text:
                filename_parts.append("AlphaScan_polar")
                theta_inc = self.fixed_theta_spinbox.value()
                phi_inc = self.alpha_scan_phi_inc_spinbox.value()
                filename_parts.append(f"th{theta_inc:.0f}_ph{phi_inc:.0f}")
            elif "3D" in scan_mode_text:
                filename_parts.append("3DScan")
            clean_parts = []
            for part in filename_parts:
                part_str = str(part)
                part_str = part_str.replace(" ", "_").replace("/", "-").replace("\\", "-")
                part_str = part_str.replace(":", "-").replace("*", "").replace("?", "").replace("\"", "'").replace("<", "").replace(">", "").replace("|", "")
                part_str = part_str.replace(" ", "_").replace("(", "").replace(")", "")
                part_str = part_str.replace("<sub>", "").replace("</sub>", "")
                clean_parts.append(part_str)
            default_filename = "_".join(clean_parts) if clean_parts else "SHG_plot"
        except Exception as e:
            print(f"Error generating default filename: {e}")
            default_filename = "SHG_plot"
        file_filters = "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;PDF File (*.pdf);;SVG Vector Graphic (*.svg);;All Files (*)"
        filePath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            default_filename,
            file_filters
        )
        if filePath:
            try:
                original_facecolor = self.canvas.fig.get_facecolor()
                self.canvas.fig.set_facecolor('white')
                self.canvas.fig.savefig(filePath, dpi=300, bbox_inches='tight')
                self.canvas.fig.set_facecolor(original_facecolor)
                print(f"Image saved to: {filePath}")
            except Exception as e:
                print(f"Error saving image: {e}")
    def handle_show_symbolic_expressions(self):
        try:
            orientation = dict(phi_c=self.euler_phi_spinbox.value(), theta_c=self.euler_theta_spinbox.value(), psi_c=self.euler_psi_spinbox.value())
            expressions = calculate_symbolic_shg_expressions(self.selected_group, orientation)
            dialog = SymbolicExpressionsDialog(expressions, self)
            dialog.exec_()
        except Exception as error:
            QMessageBox.critical(self, "Symbolic expressions", str(error))

class SHG3DPlotWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D SHG Visualization")
        self.setGeometry(150, 150, 800, 600)

        self.canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.ax = self.canvas.figure.add_subplot(111, projection='3d')
        self.setCentralWidget(self.canvas)

    def plot_data(self, R_data, angle1_grid, angle2_grid, plot_type="radiation_pattern"):
        self.ax.clear()

        # Plot the surface
        surf = self.ax.plot_surface(angle1_grid, angle2_grid, R_data, cmap=cm.viridis,
                                    linewidth=0, antialiased=False, alpha=0.7)

        self.canvas.figure.colorbar(surf, shrink=0.5, aspect=5, ax=self.ax, label='SHG Intensity (a.u.)')

        if plot_type == "radiation_pattern":
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('3D SHG Radiation Pattern')
        elif plot_type == "theta_alpha_scan":
            self.ax.set_xlabel('Incident Angle θ (deg)')
            self.ax.set_ylabel('Polarization Angle α (deg)')
            self.ax.set_zlabel('SHG Intensity (a.u.)')
            self.ax.set_title('3D Scan of SHG Intensity vs. θ and α')

        self.ax.view_init(elev=20., azim=-35)
        self.canvas.draw()

class SymbolicExpressionsDialog(QDialog):
    def __init__(self, expressions, parent=None):
        super().__init__(parent)
        self.expressions = expressions
        self.setWindowTitle("Symbolic SHG Tensor Expressions")
        self.setMinimumSize(600, 400)
        self.layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["Default", "LaTeX", "Unicode"])
        self.format_combo.currentTextChanged.connect(self.update_displayed_expressions)
        self.layout.addWidget(self.format_combo)

        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.layout.addWidget(self.copy_button)

        self.text_outputs = {}
        for key in self.expressions.keys():
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(QFont("Courier New", 10))
            self.tab_widget.addTab(text_edit, key.replace("_", " ").title())
            self.text_outputs[key] = text_edit

        self.update_displayed_expressions()

    def _format_single_expression(self, expression_value, format_str):
        # Check if it's a SymPy expression, Array, or any kind of Matrix
        if hasattr(expression_value, '_sympy_'):
            if format_str == "LaTeX":
                # For LaTeX, we'll need to wrap it appropriately for display
                return f"$${sympy.latex(expression_value)}$$"
            elif format_str == "Unicode":
                return sympy.pretty(expression_value)
        # Default to string representation if not a SymPy object or for "Default" format
        return str(expression_value)


    def update_displayed_expressions(self):
        current_format = self.format_combo.currentText()

        for key, text_edit_widget in self.text_outputs.items():
            expression = self.expressions[key]

            # Using a custom formatter for SymPy objects
            if hasattr(expression, 'shape') and len(expression.shape) == 2: # It's a matrix
                rows, cols = expression.shape
                formatted_str = ""
                if current_format == "LaTeX":
                    # Use sympy's latex printer for the whole matrix
                    formatted_str = f"$${sympy.latex(expression)}$$"
                else: # Unicode or Default
                    # Pretty print for console-like output
                    formatted_str = sympy.pretty(expression) if current_format == "Unicode" else str(expression)
                text_edit_widget.setPlainText(formatted_str)
            else:
                # Handle non-matrix expressions or other objects
                text_edit_widget.setPlainText(self._format_single_expression(expression, current_format))

    def copy_to_clipboard(self):
        current_tab_index = self.tab_widget.currentIndex()
        if current_tab_index != -1:
            key = list(self.expressions.keys())[current_tab_index]
            text_to_copy = self.text_outputs[key].toPlainText()
            QApplication.clipboard().setText(text_to_copy)
            QMessageBox.information(self, "Copied", "The content has been copied to the clipboard.")

if __name__ == '__main__':
    from run_workbench import main
    main()
