import sys
import numpy as np
import matplotlib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QSlider, QPushButton, QComboBox, 
                            QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox, QDial, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D # 导入3D绘图模块

from visualization import plot_polarization_intensity
from point_groups import (point_group_components, create_tensor_with_relations, 
                        get_all_point_groups, get_components_for_group, str_to_indices)

# --- 欧拉角旋转辅助函数 ---
def get_rotation_matrix(phi_c, theta_c, psi_c):
    """ 计算ZYZ欧拉角对应的旋转矩阵 (将晶体坐标系矢量转换为实验室坐标系矢量) """
    # 角度转弧度
    phi_rad = np.deg2rad(phi_c)
    theta_rad = np.deg2rad(theta_c)
    psi_rad = np.deg2rad(psi_c)

    # 第一个Z轴旋转 (phi_c)
    Rz_phi = np.array([
        [np.cos(phi_rad), -np.sin(phi_rad), 0],
        [np.sin(phi_rad), np.cos(phi_rad),  0],
        [0,               0,                1]
    ])

    # Y'轴旋转 (theta_c)
    Ry_theta = np.array([
        [np.cos(theta_rad),  0, np.sin(theta_rad)],
        [0,                  1, 0               ],
        [-np.sin(theta_rad), 0, np.cos(theta_rad)]
    ])

    # 第二个Z''轴旋转 (psi_c)
    Rz_psi = np.array([
        [np.cos(psi_rad), -np.sin(psi_rad), 0],
        [np.sin(psi_rad), np.cos(psi_rad),  0],
        [0,               0,                1]
    ])
    
    # 整体旋转矩阵 R = Rz_psi * Ry_theta * Rz_phi
    # 注意：这里定义的是将晶体坐标系中的矢量 R_crystal 变换到实验室坐标系 R_lab = R * R_crystal
    # 因此，张量变换公式 d'_pqr (lab) = R_pi R_qj R_rk d_ijk (crystal) 中的 R 就是这个 R
    R = np.dot(Rz_psi, np.dot(Ry_theta, Rz_phi))
    return R

def rotate_tensor(tensor_crystal, R):
    """ 使用旋转矩阵R将晶体坐标系下的三阶张量tensor_crystal变换到实验室坐标系 """
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
# --- 结束 欧拉角旋转辅助函数 ---

# 检查LaTeX是否可用 - 更安全的方法
def is_latex_available():
    try:
        # 简单检查是否安装了latex命令
        import shutil
        return shutil.which('latex') is not None
    except:
        return False

# 配置matplotlib支持中文显示
# 请确保以下列表中的至少一种字体在您的系统上可用
# 微软雅黑 (Microsoft YaHei), 黑体 (SimHei), 宋体 (SimSun), 或 Arial
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['lines.linewidth'] = 2

# 禁用LaTeX，使用普通文本渲染
use_latex = False  # 直接设置为False，避免LaTeX相关问题

# 创建一个自定义的QSlider类，支持双击事件
class ClickableSlider(QSlider):
    doubleClicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)  # 默认为水平滑块
        self.installEventFilter(self)
        
    def eventFilter(self, obj, event):
        if obj is self and event.type() == QEvent.MouseButtonDblClick:
            self.doubleClicked.emit()
            return True
        return super().eventFilter(obj, event)

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, polar=True)
        super().__init__(self.fig)

class ManualInputWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("非线性极化张量手动输入")
        self.setGeometry(150, 150, 1200, 800)
        
        # 主布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        
        # 左侧控制面板
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.main_layout.addWidget(self.left_widget, 1)
        
        # 张量输入区域
        self.tensor_box = QGroupBox("非线性极化张量输入")
        self.tensor_layout = QVBoxLayout()
        self.tensor_box.setLayout(self.tensor_layout)
        
        # 说明标签
        self.instruction_label = QLabel("请输入3x6矩阵的d<sub>ij</sub>分量，或选择预设晶体")
        self.tensor_layout.addWidget(self.instruction_label)
        
        # 预设晶体选择
        self.preset_box = QGroupBox("预设晶体")
        self.preset_layout = QVBoxLayout()
        self.preset_box.setLayout(self.preset_layout)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["自定义", "MoS2 (2H)", "BaTiO3", "LiNbO3", "KDP"])
        self.preset_combo.currentIndexChanged.connect(self.load_preset)
        self.preset_layout.addWidget(self.preset_combo)
        
        self.tensor_layout.addWidget(self.preset_box)
        
        # 张量输入网格
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        
        # 创建3x6矩阵的输入框
        self.tensor_inputs = {}
        
        # 添加列标签
        for j in range(6):
            label = QLabel(f"{j+1}")
            label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(label, 0, j+1)
        
        # 添加行标签和输入框
        for i in range(3):
            # 行标签
            row_label = QLabel(f"d<sub>{i+1}j</sub>")
            row_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.grid_layout.addWidget(row_label, i+1, 0)
            
            # 每行6个输入框
            for j in range(6):
                input_field = QDoubleSpinBox()
                input_field.setRange(-100, 100)
                input_field.setDecimals(2)
                input_field.setSingleStep(0.1)
                input_field.setValue(0.0)
                input_field.setFixedWidth(80)
                self.grid_layout.addWidget(input_field, i+1, j+1)
                
                # 保存引用 - 使用(行,列)索引
                self.tensor_inputs[(i, j)] = input_field
        
        # 添加滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.grid_widget)
        self.tensor_layout.addWidget(scroll_area)
        
        self.left_layout.addWidget(self.tensor_box)
        
        # 按钮区域
        self.buttons_widget = QWidget()
        self.buttons_layout = QHBoxLayout(self.buttons_widget)
        
        self.plot_button = QPushButton("绘制极化强度极图")
        self.plot_button.clicked.connect(self.plot)
        self.buttons_layout.addWidget(self.plot_button)
        
        self.back_button = QPushButton("返回滑块模式")
        self.back_button.clicked.connect(self.go_back)
        self.buttons_layout.addWidget(self.back_button)
        
        self.left_layout.addWidget(self.buttons_widget)
        
        # 右侧绘图区域
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.main_layout.addWidget(self.plot_widget, 2)
        
        self.canvas = PlotCanvas(self, width=8, height=8)
        self.plot_layout.addWidget(self.canvas)
        
        # 初始化张量和绘图参数
        self.phi = 0.0
        self.tensor = np.zeros((3, 3, 3))
        self.alpha = 0.0 # Initialize alpha for ManualInputWindow
        self.update_input_polarization_controls_manual() # Set initial visibility
        
        # 方位角控制
        self.phi_control = QWidget()
        self.phi_layout = QHBoxLayout(self.phi_control)
        
        self.phi_label = QLabel("方位角 φ (度):")
        self.phi_label.setToolTip("入射光传播方向在 XY 平面内的投影与 X 轴正方向的夹角 (0° 到 359°)。\n这定义了入射光传播方向的方位。")
        
        self.phi_slider = QSlider(Qt.Horizontal)
        self.phi_slider.setMinimum(0)
        self.phi_slider.setMaximum(359)
        self.phi_slider.setValue(0)
        self.phi_slider.valueChanged.connect(self.update_phi)
        self.phi_layout.addWidget(self.phi_slider)
        
        self.phi_display = QLabel("0°")
        self.phi_layout.addWidget(self.phi_display)
        
        self.left_layout.addWidget(self.phi_control)
        
        # 输入光偏振模式选择 (ManualInputWindow)
        self.input_polarization_control_widget = QWidget()
        self.input_polarization_layout = QHBoxLayout(self.input_polarization_control_widget)

        self.input_polarization_label_manual = QLabel("输入光偏振:")
        self.input_polarization_layout.addWidget(self.input_polarization_label_manual)

        self.input_polarization_combo_manual = QComboBox()
        self.input_polarization_combo_manual.addItems(["默认 (θ-偏振)", "线偏振", "左旋圆偏振 (LCP)", "右旋圆偏振 (RCP)"])
        self.input_polarization_combo_manual.currentIndexChanged.connect(self.update_input_polarization_controls_manual)
        self.input_polarization_layout.addWidget(self.input_polarization_combo_manual)
        self.left_layout.addWidget(self.input_polarization_control_widget)

        # 线偏振角度 alpha 控制 (ManualInputWindow - 初始隐藏)
        self.alpha_control_widget_manual = QWidget()
        self.alpha_layout_manual = QHBoxLayout(self.alpha_control_widget_manual)
        self.alpha_label_manual = QLabel("偏振角 α (度):")
        self.alpha_layout_manual.addWidget(self.alpha_label_manual)
        self.alpha_layout_manual.setContentsMargins(0,0,0,0) # 移除不必要的边距

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
        
        # 检测偏振选择
        self.detection_widget = QWidget()
        self.detection_layout = QHBoxLayout(self.detection_widget)
        
        self.detection_label = QLabel("检测偏振:")
        self.detection_layout.addWidget(self.detection_label)
        
        self.detection_combo = QComboBox()
        self.detection_combo.addItems(["总强度 |P|²", "平行模式 (∥)", "垂直模式 (⊥)"])
        self.detection_combo.currentIndexChanged.connect(self.plot)
        self.detection_layout.addWidget(self.detection_combo)
        
        self.left_layout.addWidget(self.detection_widget)
    
    def update_phi(self):
        """更新方位角"""
        self.phi = self.phi_slider.value()
        self.phi_display.setText(f"{self.phi}°")
        self.plot()
    
    def load_preset(self, index):
        """加载预设晶体数据"""
        if index == 0:  # 自定义
            return
        
        # 重置所有输入框
        for input_field in self.tensor_inputs.values():
            input_field.setValue(0.0)
        
        if index == 1:  # MoS2 (2H)
            # 2H-MoS2的非零分量 (D3h点群)
            # https://doi.org/10.1103/PhysRevB.87.161403
            # d_31 = d_32 = 1.0
            self.tensor_inputs[(2, 0)].setValue(1.0)  # d_31
            self.tensor_inputs[(2, 1)].setValue(1.0)  # d_32
            self.tensor_inputs[(0, 4)].setValue(1.0)  # d_15
            self.tensor_inputs[(0, 4)].setValue(1.0)  # d_15
            self.tensor_inputs[(1, 3)].setValue(1.0)  # d_24
            self.tensor_inputs[(1, 3)].setValue(1.0)  # d_24
            
        elif index == 2:  # BaTiO3
            # BaTiO3的非零分量 (4mm点群)
            # 简化值，实际值应查阅文献
            self.tensor_inputs[(2, 2)].setValue(10.0)  # d_33
            self.tensor_inputs[(2, 0)].setValue(3.0)   # d_31
            self.tensor_inputs[(2, 1)].setValue(3.0)   # d_32
            self.tensor_inputs[(0, 4)].setValue(3.0)   # d_15
            self.tensor_inputs[(1, 3)].setValue(3.0)   # d_24
            
        elif index == 3:  # LiNbO3
            # LiNbO3的非零分量 (3m点群)
            # 简化值，实际值应查阅文献
            self.tensor_inputs[(2, 2)].setValue(20.0)  # d_33
            self.tensor_inputs[(2, 0)].setValue(5.0)   # d_31
            self.tensor_inputs[(2, 1)].setValue(5.0)   # d_32
            self.tensor_inputs[(0, 4)].setValue(4.0)   # d_15
            self.tensor_inputs[(1, 3)].setValue(4.0)   # d_24
            self.tensor_inputs[(1, 0)].setValue(-4.0)  # d_22 = -d_21
            self.tensor_inputs[(0, 5)].setValue(4.0)   # d_16
            
        elif index == 4:  # KDP
            # KDP的非零分量 (42m点群)
            # 简化值，实际值应查阅文献
            self.tensor_inputs[(0, 3)].setValue(1.0)   # d_14
            self.tensor_inputs[(1, 4)].setValue(1.0)   # d_25
            self.tensor_inputs[(2, 5)].setValue(1.0)   # d_36
        
        self.plot()
    
    def plot(self):
        """绘制极化强度极图"""
        # 初始化d矩阵 (3x6)
        d_matrix = np.zeros((3, 6))
        
        # 从输入框获取张量值
        for (i, j), input_field in self.tensor_inputs.items():
            d_matrix[i, j] = input_field.value()
        
        # 清除当前图形
        self.canvas.axes.clear()
        
        # 绘制新图
        phi_rad = self.phi * np.pi/180  # 转换为弧度
        theta_range = np.linspace(0, 2*np.pi, 360)
        
        # 计算极化强度
        P = []
        for theta in theta_range:
            # 计算入射电场方向 k_hat (球坐标系到笛卡尔)
            k_hat = np.array([
                np.sin(theta)*np.cos(phi_rad),
                np.sin(theta)*np.sin(phi_rad),
                np.cos(theta)
            ])

            # 根据输入偏振模式计算实际电场 E_omega (ManualInputWindow)
            input_pol_mode = self.input_polarization_combo_manual.currentText()
            E_omega = np.zeros(3)
            alpha_rad = np.deg2rad(self.alpha)

            if input_pol_mode == "线偏振":
                vec_theta = np.array([np.cos(theta) * np.cos(phi_rad),
                                      np.cos(theta) * np.sin(phi_rad),
                                      -np.sin(theta)])
                vec_phi = np.array([-np.sin(phi_rad),
                                    np.cos(phi_rad),
                                    0.0])
                E_omega = np.cos(alpha_rad) * vec_theta + np.sin(alpha_rad) * vec_phi
            
            elif input_pol_mode == "左旋圆偏振 (LCP)":
                vec_theta = np.array([np.cos(theta) * np.cos(phi_rad),
                                      np.cos(theta) * np.sin(phi_rad),
                                      -np.sin(theta)])
                vec_phi = np.array([-np.sin(phi_rad),
                                    np.cos(phi_rad),
                                    0.0])
                E_omega = (1/np.sqrt(2)) * (vec_theta + 1j * vec_phi) # Complex E_omega

            elif input_pol_mode == "右旋圆偏振 (RCP)":
                vec_theta = np.array([np.cos(theta) * np.cos(phi_rad),
                                      np.cos(theta) * np.sin(phi_rad),
                                      -np.sin(theta)])
                vec_phi = np.array([-np.sin(phi_rad),
                                    np.cos(phi_rad),
                                    0.0])
                E_omega = (1/np.sqrt(2)) * (vec_theta - 1j * vec_phi) # Complex E_omega

            else: # "默认" 模式 - p-polarized
                vec_theta = np.array([np.cos(theta) * np.cos(phi_rad),
                                      np.cos(theta) * np.sin(phi_rad),
                                      -np.sin(theta)])
                E_omega = vec_theta # Real E_omega
                # 当模式为"默认 (θ-偏振)"时，与 alpha=0 的线偏振行为一致
                E_omega = vec_theta
            
            # 构建6阶向量 [Ex², Ey², Ez², 2EyEz, 2ExEz, 2ExEy]
            Ex, Ey, Ez = E_omega[0], E_omega[1], E_omega[2] # These can be complex
            E_voigt = np.array([
                Ex*Ex, Ey*Ey, Ez*Ez, # E_i*E_j terms
                2*Ey*Ez, 2*Ex*Ez, 2*Ex*Ey
            ], dtype=np.complex128) # Ensure E_voigt is complex
            
            # 使用矩阵乘法计算 P_i = d_il * E_voigt_l
            P_i = np.dot(d_matrix, E_voigt)
            
            # 获取检测偏振选择
            detection_pol = self.detection_combo.currentIndex()
            
            if detection_pol == 0:  # 总强度 |P|²
                intensity = np.linalg.norm(P_i)**2
            else:
                # 平行和垂直模式检偏 - 只考虑xy平面
                
                # 获取入射基频光 E_omega 在xy平面的投影方向
                # E_omega 已经是归一化的入射电场矢量
                E_omega_xy_projection = np.array([Ex, Ey, 0.0]) # Ex, Ey 是 E_omega 的分量
                E_omega_xy_norm = np.linalg.norm(E_omega_xy_projection)
                
                # 如果E_omega_xy_projection几乎为零向量（电场几乎垂直于xy平面），使用单位x向量作为参考
                if E_omega_xy_norm < 1e-9: # 使用一个较小的阈值
                    E_incident_parallel_direction_xy = np.array([1.0, 0.0, 0.0])
                else:
                    # 归一化得到xy平面内的方向向量
                    E_incident_parallel_direction_xy = E_omega_xy_projection / E_omega_xy_norm
                
                # 计算二次谐波 P_i 在xy平面内的分量
                P_xy = np.array([P_i[0], P_i[1], 0.0])
                
                if detection_pol == 1:  # 平行模式 (∥) - SHG在xy平面内平行于入射光E_omega在xy平面投影的分量
                    # 计算P_xy在E_incident_parallel_direction_xy方向的投影的平方
                    P_parallel_component_xy = np.dot(P_xy, E_incident_parallel_direction_xy)
                    intensity = P_parallel_component_xy**2
                elif detection_pol == 2:  # 垂直模式 (⊥) - SHG在xy平面内垂直于入射光E_omega在xy平面投影的分量
                    # 在xy平面内，与E_incident_parallel_direction_xy垂直的单位向量
                    # 逆时针旋转90度
                    E_incident_perpendicular_direction_xy = np.array([-E_incident_parallel_direction_xy[1], E_incident_parallel_direction_xy[0], 0.0])
                    
                    # 计算P_xy在垂直方向的投影的平方
                    P_perpendicular_component_xy = np.dot(P_xy, E_incident_perpendicular_direction_xy)
                    intensity = P_perpendicular_component_xy**2
            
            P.append(intensity)
        
        # 绘制极图
        self.canvas.axes.plot(theta_range, P, lw=2, color='purple')
        
        # 获取预设名称
        preset_name = self.preset_combo.currentText()
        
        # 获取检测偏振的文本描述
        detection_text = self.detection_combo.currentText()

        # 获取输入偏振模式 (ManualInputWindow)
        input_pol_mode = self.input_polarization_combo_manual.currentText()
        input_pol_text = f"输入: {input_pol_mode}"
        if input_pol_mode == "线偏振":
            input_pol_text += f" (α={self.alpha}°)"
        
        # 更新标题，包含检测偏振信息
        title_text = f"{preset_name} - {detection_text}\n{input_pol_text}, φ={self.phi:.1f}°"
            
        self.canvas.axes.set_title(title_text, pad=20)
        self.canvas.axes.grid(True, linestyle='--', alpha=0.5)
        self.canvas.draw()
    
    def go_back(self):
        """关闭当前窗口，返回主窗口"""
        self.parent().show()
        self.close()

    def update_input_polarization_controls_manual(self):
        """根据选择的输入偏振模式，显示或隐藏alpha控制 (ManualInputWindow)"""
        mode = self.input_polarization_combo_manual.currentText()
        is_linear_polarization = (mode == "线偏振")
        self.alpha_control_widget_manual.setVisible(is_linear_polarization)

        if is_linear_polarization:
            self.alpha_label_manual.setToolTip("线偏振光的电场振动方向与 θ-方向单位矢量之间的夹角。\nθ-方向矢量大致对应p偏振方向(在过Z轴和传播方向的平面内，且垂直于传播方向)。\nφ-方向矢量大致对应s偏振方向(垂直于θ-方向和传播方向)。\nα=0° 表示电场沿 θ-方向； α=90° 表示电场沿 φ-方向。")
        else:
            self.alpha_label_manual.setToolTip("")

        self.plot() # Re-plot when mode changes

    def update_alpha_display_manual(self, update_plot=True):
        """更新alpha角度显示 (ManualInputWindow)"""
        try:
            self.alpha = self.alpha_slider_manual.value()
            self.alpha_display_manual.setText(f"{self.alpha}°")
            if update_plot:
                self.plot()
        except Exception as e:
            print(f"更新alpha角度显示时出错 (ManualInputWindow): {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("非线性极化模拟器")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化关键属性
        self.phi = 0.0
        self.component_values = {}
        self.component_widgets = {}
        self.base_tensor = None

        # 主布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        
        # 左侧控制面板 - 将其放入QScrollArea
        self.controls_scroll_area = QScrollArea()
        self.controls_scroll_area.setWidgetResizable(True) # 关键：允许内部控件调整大小

        self.controls_widget = QWidget() # 这是原来的controls_widget
        self.controls_layout = QVBoxLayout(self.controls_widget) # controls_widget使用QVBoxLayout
        self.controls_scroll_area.setWidget(self.controls_widget) # 将controls_widget放入scroll_area
        self.main_layout.addWidget(self.controls_scroll_area, 1) # 将scroll_area添加到主布局
        
        # 点群选择
        self.group_box = QGroupBox("晶体点群选择")
        self.group_layout = QVBoxLayout()
        self.group_box.setLayout(self.group_layout)
        
        self.group_combo = QComboBox()
        self.group_combo.addItems(get_all_point_groups())
        self.group_combo.currentIndexChanged.connect(self.update_point_group)
        self.group_layout.addWidget(self.group_combo)
        
        # 分量显示区
        self.components_label = QLabel("非零分量:")
        self.group_layout.addWidget(self.components_label)
        self.components_list = QLabel()
        self.group_layout.addWidget(self.components_list)
        
        self.controls_layout.addWidget(self.group_box)
        
        # 参数调整区
        self.params_box = QGroupBox("参数调整")
        self.params_layout = QGridLayout()
        self.params_box.setLayout(self.params_layout)
        
        # 扫描模式选择
        self.scan_mode_label = QLabel("扫描模式:")
        self.params_layout.addWidget(self.scan_mode_label, 0, 0)
        self.scan_mode_combo = QComboBox()
        self.scan_mode_combo.addItems(["默认扫描 (θ-极图)", "偏振角扫描 (α vs I)", "偏振角扫描 (α-强度极图)"])
        self.scan_mode_combo.currentIndexChanged.connect(self.update_scan_mode_controls)
        self.params_layout.addWidget(self.scan_mode_combo, 0, 1, 1, 2)

        # 固定Theta角输入 (仅在偏振角扫描模式下可见/可用)
        self.fixed_theta_label = QLabel("固定入射角 θ (度):") # 天顶角
        self.params_layout.addWidget(self.fixed_theta_label, 1, 0)
        self.fixed_theta_spinbox = QDoubleSpinBox()
        self.fixed_theta_spinbox.setRange(0, 180) # 0 (垂直) 到 180
        self.fixed_theta_spinbox.setValue(0.0) # 默认垂直入射
        self.fixed_theta_spinbox.setSuffix("°")
        self.fixed_theta_spinbox.valueChanged.connect(self.plot) # 值改变时重绘
        self.params_layout.addWidget(self.fixed_theta_spinbox, 1, 1, 1, 2)

        # 晶体朝向欧拉角控制
        self.euler_phi_label = QLabel("晶体 φ<sub>c</sub> (Z旋转, 度):")
        self.params_layout.addWidget(self.euler_phi_label, 2, 0)
        self.euler_phi_spinbox = QDoubleSpinBox()
        self.euler_phi_spinbox.setRange(0, 360)
        self.euler_phi_spinbox.setValue(0.0)
        self.euler_phi_spinbox.setSuffix("°")
        self.euler_phi_spinbox.valueChanged.connect(self.plot)
        self.params_layout.addWidget(self.euler_phi_spinbox, 2, 1, 1, 2)

        self.euler_theta_label = QLabel("晶体 θ<sub>c</sub> (Y'旋转, 度):")
        self.params_layout.addWidget(self.euler_theta_label, 3, 0)
        self.euler_theta_spinbox = QDoubleSpinBox()
        self.euler_theta_spinbox.setRange(0, 180) # 通常0-180度足够
        self.euler_theta_spinbox.setValue(0.0)
        self.euler_theta_spinbox.setSuffix("°")
        self.euler_theta_spinbox.valueChanged.connect(self.plot)
        self.params_layout.addWidget(self.euler_theta_spinbox, 3, 1, 1, 2)

        self.euler_psi_label = QLabel("晶体 ψ<sub>c</sub> (Z''旋转, 度):")
        self.params_layout.addWidget(self.euler_psi_label, 4, 0)
        self.euler_psi_spinbox = QDoubleSpinBox()
        self.euler_psi_spinbox.setRange(0, 360)
        self.euler_psi_spinbox.setValue(0.0)
        self.euler_psi_spinbox.setSuffix("°")
        self.euler_psi_spinbox.valueChanged.connect(self.plot)
        self.params_layout.addWidget(self.euler_psi_spinbox, 4, 1, 1, 2)
        
        # 方位角控制
        self.phi_label = QLabel("光束方位角 φ (度):")
        self.phi_label.setToolTip("入射光传播方向在 XY 平面内的投影与 X 轴正方向的夹角 (0° 到 359°)。\n这定义了入射光传播方向的方位。")
        self.params_layout.addWidget(self.phi_label, 5, 0) # 行号调整
        
        self.phi_slider = QSlider(Qt.Horizontal)
        self.phi_slider.setMinimum(0)
        self.phi_slider.setMaximum(359)
        self.phi_slider.setValue(0)
        self.phi_slider.valueChanged.connect(self.update_phi_display)
        self.params_layout.addWidget(self.phi_slider, 5, 1) # 行号调整
        
        self.phi_display = QLabel("0°")
        self.params_layout.addWidget(self.phi_display, 5, 2) # 行号调整
        
        # 张量分量强度调整
        self.tensor_label = QLabel("张量整体强度:") # 标签修改
        self.params_layout.addWidget(self.tensor_label, 6, 0) # 行号调整
        
        self.tensor_slider = QSlider(Qt.Horizontal)
        self.tensor_slider.setMinimum(1)
        self.tensor_slider.setMaximum(20)
        self.tensor_slider.setValue(10)
        self.tensor_slider.valueChanged.connect(self.update_tensor_display)
        self.params_layout.addWidget(self.tensor_slider, 6, 1) # 行号调整
        
        self.tensor_display = QLabel("1.0")
        self.params_layout.addWidget(self.tensor_display, 6, 2) # 行号调整
        
        # 输入光偏振模式选择
        self.input_polarization_label = QLabel("输入光偏振:")
        self.params_layout.addWidget(self.input_polarization_label, 7, 0) # 行号调整

        self.input_polarization_combo = QComboBox()
        self.input_polarization_combo.addItems(["默认 (θ-偏振)", "线偏振", "左旋圆偏振 (LCP)", "右旋圆偏振 (RCP)"])
        self.input_polarization_combo.currentIndexChanged.connect(self.update_input_polarization_controls)
        self.params_layout.addWidget(self.input_polarization_combo, 7, 1, 1, 2) # 行号调整

        # 线偏振角度 alpha 控制 (初始隐藏) - 用一个 QWidget 包裹
        self.alpha_control_widget = QWidget()
        self.alpha_control_layout = QHBoxLayout(self.alpha_control_widget)
        self.alpha_control_layout.setContentsMargins(0,0,0,0)
        self.alpha_label = QLabel("偏振角 α (度):") 
        self.alpha_control_layout.addWidget(self.alpha_label)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(359) 
        self.alpha_slider.setValue(0)
        self.alpha_slider.valueChanged.connect(self.update_alpha_display)
        self.alpha_control_layout.addWidget(self.alpha_slider)
        self.alpha_display = QLabel("0°")
        self.alpha_control_layout.addWidget(self.alpha_display)
        self.params_layout.addWidget(self.alpha_control_widget, 8, 0, 1, 3) # 行号调整
        self.alpha_control_widget.setVisible(False) 
        
        # 检测偏振选择
        self.detection_label = QLabel("检测偏振:")
        self.params_layout.addWidget(self.detection_label, 9, 0) # 行号调整
        
        self.detection_combo = QComboBox()
        self.detection_combo.addItems(["总强度 |P|²", "平行模式 (∥)", "垂直模式 (⊥)"])
        self.detection_combo.currentIndexChanged.connect(self.plot)
        self.params_layout.addWidget(self.detection_combo, 9, 1, 1, 2) # 行号调整
        
        self.controls_layout.addWidget(self.params_box)
        
        # 分量单独调整区域
        self.component_box = QGroupBox("分量单独调整")
        self.component_layout = QVBoxLayout()
        self.component_box.setLayout(self.component_layout)
        
        self.controls_layout.addWidget(self.component_box)
        
        # 按钮区域
        self.buttons_widget = QWidget()
        self.buttons_layout = QHBoxLayout(self.buttons_widget)
        
        self.plot_button = QPushButton("绘制极化强度极图")
        self.plot_button.clicked.connect(self.plot)
        self.buttons_layout.addWidget(self.plot_button)
        
        self.reset_button = QPushButton("重置参数")
        self.reset_button.clicked.connect(self.reset_params)
        self.buttons_layout.addWidget(self.reset_button)
        
        # 添加打开手动输入窗口的按钮
        self.manual_input_button = QPushButton("手动输入模式")
        self.manual_input_button.clicked.connect(self.open_manual_input)
        self.buttons_layout.addWidget(self.manual_input_button)
        
        self.plot_3d_button = QPushButton("绘制3D SHG图案")
        self.plot_3d_button.clicked.connect(self.open_3d_plot_window)
        self.buttons_layout.addWidget(self.plot_3d_button)
        
        self.controls_layout.addWidget(self.buttons_widget)
        
        # 右侧绘图区域
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.main_layout.addWidget(self.plot_widget, 2)
        
        self.canvas = PlotCanvas(self, width=8, height=8)
        self.plot_layout.addWidget(self.canvas)
        
        # 初始化选择的点群并设置初始参数
        self.selected_group = self.group_combo.currentText()
        if self.selected_group:
            self.update_point_group(0)  # 使用索引0进行初始化
            self.reset_params(False)  # 不自动更新图形
        
        self.alpha = 0 # 初始化alpha值
        self.update_input_polarization_controls() # 根据初始模式设置控件可见性
        self.update_scan_mode_controls() # 根据初始扫描模式设置控件可见性

        # 延迟绘图，确保界面完全加载
        QApplication.processEvents()
        self.plot()
        
    def update_point_group(self, index):
        try:
            if self.group_combo.count() == 0:
                return
                
            self.selected_group = self.group_combo.currentText()
            components = get_components_for_group(self.selected_group)
            
            # 更新分量显示
            comp_str = []
            for c in components:
                indices = str_to_indices(c)
                i, j, k = indices
                # 将jk转换为Voigt索引l
                voigt_map = {
                    (0, 0): 1, (1, 1): 2, (2, 2): 3,
                    (1, 2): 4, (2, 1): 4, (0, 2): 5, 
                    (2, 0): 5, (0, 1): 6, (1, 0): 6
                }
                l = voigt_map.get((j, k), 0)
                if l > 0:
                    comp_str.append(f"d<sub>{i+1}{l}</sub>")
                else:
                    comp_str.append(f"d<sub>{i+1}{j+1}{k+1}</sub>")
            
            self.components_list.setText(", ".join(comp_str))
            
            # 更新张量
            self.base_tensor = create_tensor_with_relations(self.selected_group)
            self.component_values = {comp: 1.0 for comp in components}
            
            # 重建分量滑块
            self.clear_component_sliders()
            self.create_component_sliders(components)
            
            # 更新后自动绘图
            self.plot()
        except Exception as e:
            print(f"更新点群时出错: {e}")
        
    def clear_component_sliders(self):
        """清除现有的分量滑块"""
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
                    # 清除子布局中的控件
                    while layout.count():
                        layout_item = layout.takeAt(0)
                        if layout_item.widget():
                            layout_item.widget().deleteLater()
                    # 移除子布局
                    self.component_layout.removeItem(layout)
            self.component_widgets = {}
        except Exception as e:
            print(f"清除分量滑块时出错: {e}")
        
    def create_component_sliders(self, components):
        """为每个分量创建滑块控件"""
        try:
            # 如果组件太多，使用网格布局
            if len(components) > 6:
                grid_layout = QGridLayout()
                self.component_layout.addLayout(grid_layout)
                
                row, col = 0, 0
                max_cols = 2  # 每行最多2个滑块
                
                for comp in components:
                    widget = QWidget()
                    layout = QVBoxLayout(widget)  # 使用垂直布局
                    
                    # 添加标签 - 转换为dij表示法
                    indices = str_to_indices(comp)
                    i, j, k = indices
                    # 将jk转换为Voigt索引l (11->1, 22->2, 33->3, 23/32->4, 13/31->5, 12/21->6)
                    voigt_map = {
                        (0, 0): 1, (1, 1): 2, (2, 2): 3,
                        (1, 2): 4, (2, 1): 4, (0, 2): 5, 
                        (2, 0): 5, (0, 1): 6, (1, 0): 6
                    }
                    l = voigt_map.get((j, k), 0)
                    if l > 0:
                        label = QLabel(f"d<sub>{i+1}{l}</sub>:")
                    else:
                        label = QLabel(f"d<sub>{i+1}{j+1}{k+1}</sub>:")
                        
                    label.setAlignment(Qt.AlignCenter)  # 居中对齐
                    layout.addWidget(label)
                    
                    # 水平布局包含滑块和数值
                    slider_layout = QHBoxLayout()
                    
                    # 使用自定义QSlider，提供滑块控制
                    slider = ClickableSlider()
                    slider.setMinimum(-1000)        # -10.0
                    slider.setMaximum(1000)         # 10.0，精度为0.01
                    slider.setValue(100)            # 默认值1.0
                    slider.setTickPosition(QSlider.TicksBelow)  # 在滑块下方显示刻度
                    slider.setTickInterval(200)     # 每隔2.0显示一个刻度
                    slider.setMinimumWidth(150)     # 设置最小宽度
                    slider.setProperty("component", comp)
                    slider.valueChanged.connect(self.update_component_value_slider)
                    
                    # 使用闭包传递参数
                    def create_reset_handler(component=comp):
                        return lambda: self.reset_component(component)
                    
                    slider.doubleClicked.connect(create_reset_handler())
                    
                    # 添加数值显示标签
                    value_label = QLabel("1.00")
                    value_label.setAlignment(Qt.AlignCenter)  # 居中对齐
                    value_label.setMinimumWidth(50)  # 确保标签有足够的宽度显示
                    
                    slider_layout.addWidget(slider)
                    slider_layout.addWidget(value_label)
                    
                    layout.addLayout(slider_layout)
                    
                    self.component_widgets[comp] = {
                        'slider': slider,
                        'label': value_label
                    }
                    
                    grid_layout.addWidget(widget, row, col)
                    
                    # 更新行列位置
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
            else:
                # 组件较少时使用列表布局
                for comp in components:
                    widget = QWidget()
                    layout = QHBoxLayout(widget)
                    
                    label = QLabel(f"χ{comp}:")
                    label.setMinimumWidth(40)  # 设置标签最小宽度
                    layout.addWidget(label)
                    
                    # 使用自定义QSlider，提供滑块控制
                    slider = ClickableSlider()
                    slider.setMinimum(-1000)        # -10.0
                    slider.setMaximum(1000)         # 10.0，精度为0.01
                    slider.setValue(100)            # 默认值1.0
                    slider.setTickPosition(QSlider.TicksBelow)  # 在滑块下方显示刻度
                    slider.setTickInterval(200)     # 每隔2.0显示一个刻度
                    slider.setMinimumWidth(180)     # 设置最小宽度
                    slider.setProperty("component", comp)
                    slider.valueChanged.connect(self.update_component_value_slider)
                    
                    # 使用闭包传递参数
                    def create_reset_handler(component=comp):
                        return lambda: self.reset_component(component)
                    
                    slider.doubleClicked.connect(create_reset_handler())
                    
                    # 添加数值显示标签
                    value_label = QLabel("1.00")
                    value_label.setMinimumWidth(50)  # 设置最小宽度
                    value_label.setAlignment(Qt.AlignCenter)  # 居中对齐
                    
                    layout.addWidget(slider)
                    layout.addWidget(value_label)
                    
                    self.component_widgets[comp] = {
                        'slider': slider,
                        'label': value_label
                    }
                    
                    self.component_layout.addWidget(widget)
        except Exception as e:
            print(f"创建分量滑块时出错: {e}")
            
    def update_component_value_slider(self):
        """更新单个分量值（从滑块）"""
        try:
            sender = self.sender()
            if not sender:
                return
                
            comp = sender.property("component")
            if not comp:
                return
                
            raw_value = sender.value()
            
            # 将-1000到1000的值转换为-10.0到10.0
            value = raw_value / 100.0
            
            # 安全检查：确保组件存在于字典中
            if comp in self.component_widgets and 'label' in self.component_widgets[comp]:
                # 更新显示标签
                self.component_widgets[comp]['label'].setText(f"{value:.2f}")
                
                # 更新组件值
                self.component_values[comp] = value
                
                # 实时更新图形
                self.plot()
        except Exception as e:
            print(f"更新组件值时出错: {e}")
        
    def update_phi_display(self, update_plot=True):
        """更新φ角度显示"""
        try:
            self.phi = self.phi_slider.value()
            self.phi_display.setText(f"{self.phi}°")
            # 当角度改变时自动更新图形
            if update_plot:
                self.plot()
        except Exception as e:
            print(f"更新角度显示时出错: {e}")
        
    def update_tensor_display(self, update_plot=True):
        """更新张量强度显示"""
        try:
            value = self.tensor_slider.value() / 10.0
            self.tensor_display.setText(f"{value:.1f}")
            # 当强度改变时自动更新图形
            if update_plot:
                self.plot()
        except Exception as e:
            print(f"更新张量强度显示时出错: {e}")
        
    def reset_params(self, update_plot=True):
        """重置所有参数"""
        try:
            if not hasattr(self, 'phi_slider') or not hasattr(self, 'tensor_slider'):
                return
                
            self.phi_slider.setValue(0)
            self.tensor_slider.setValue(10)
            
            # 重置所有分量旋钮
            for comp, widgets in self.component_widgets.items():
                if 'slider' in widgets:
                    widgets['slider'].setValue(100)
                if 'label' in widgets:
                    widgets['label'].setText("1.00")
                self.component_values[comp] = 1.0
                
            # 更新显示
            self.update_phi_display(False)  # 不自动更新图形
            self.update_tensor_display(False)  # 不自动更新图形
            
            # 重置后自动更新图形
            if update_plot:
                self.plot()
        except Exception as e:
            print(f"重置参数时出错: {e}")
            
    def reset_component(self, component):
        """重置单个分量的值为1.0"""
        try:
            if component in self.component_widgets and 'slider' in self.component_widgets[component]:
                self.component_widgets[component]['slider'].setValue(100)  # 对应1.0
                if 'label' in self.component_widgets[component]:
                    self.component_widgets[component]['label'].setText("1.00")
                self.component_values[component] = 1.0
                self.plot()  # 更新图形
        except Exception as e:
            print(f"重置分量 {component} 时出错: {e}")
            
    def plot(self):
        """绘制极化强度极图或笛卡尔图，根据扫描模式决定"""
        # 检查是否已初始化完成
        if not hasattr(self, 'base_tensor') or self.base_tensor is None:
            return
        
        if not hasattr(self, 'phi'):
            self.phi = 0.0
        
        if not hasattr(self, 'alpha'): #确保alpha已初始化
            self.alpha = 0.0

        current_scan_mode = self.scan_mode_combo.currentText()

        try:
            # 应用分量独立系数
            tensor_crystal_scaled = self.base_tensor.copy()
            for comp, value in self.component_values.items():
                try:
                    indices = str_to_indices(comp)
                    tensor_crystal_scaled[indices] = tensor_crystal_scaled[indices] * value
                except Exception as e:
                    print(f"处理分量 {comp} 时出错: {e}")
            
            # 应用全局系数
            global_scale = self.tensor_slider.value() / 10.0
            tensor_crystal_scaled = tensor_crystal_scaled * global_scale

            # 获取欧拉角并计算旋转矩阵
            phi_c = self.euler_phi_spinbox.value()
            theta_c = self.euler_theta_spinbox.value()
            psi_c = self.euler_psi_spinbox.value()
            
            # 只有当欧拉角不都为零时才进行旋转，以避免不必要的计算
            # 并确保张量在旋转前是实数或复数，以保持类型一致性
            # self.base_tensor 应该是实数张量，所以 tensor_crystal_scaled 也是
            if not (phi_c == 0 and theta_c == 0 and psi_c == 0):
                R = get_rotation_matrix(phi_c, theta_c, psi_c)
                tensor_lab_frame = rotate_tensor(tensor_crystal_scaled, R)
            else:
                tensor_lab_frame = tensor_crystal_scaled # 无旋转，直接使用
            
            # 将3x3x3张量 (此时已在实验室坐标系) 转换为3x6矩阵（Voigt表示法）
            d_matrix = np.zeros((3, 6), dtype=np.complex128 if np.iscomplexobj(tensor_lab_frame) else np.float64)
            # 转换规则：jk -> l：11->1, 22->2, 33->3, 23/32->4, 13/31->5, 12/21->6
            voigt_map = {
                (0, 0): 0, (1, 1): 1, (2, 2): 2,
                (1, 2): 3, (2, 1): 3, (0, 2): 4, 
                (2, 0): 4, (0, 1): 5, (1, 0): 5
            }
            
            # 填充d矩阵
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        if (j, k) in voigt_map:
                            l_idx = voigt_map[(j, k)] # renamed l to l_idx
                            # d_matrix[i, l_idx] += tensor[i, j_idx, k_idx]
                            d_matrix[i, l_idx] += tensor_lab_frame[i, j, k] # 使用旋转后的张量
            
            # 清除当前图形
            self.canvas.axes.clear()

            if current_scan_mode == "偏振角扫描 (α vs I)":
                self.canvas.axes.remove() # 移除旧的极坐标轴
                self.axes = self.canvas.fig.add_subplot(111) # 添加笛卡尔轴
                self.canvas.axes = self.axes # 更新引用

                fixed_theta_rad = np.deg2rad(self.fixed_theta_spinbox.value())
                phi_rad_fixed = np.deg2rad(self.phi_slider.value()) # phi from slider
                alpha_scan_rad = np.linspace(0, 2 * np.pi, 360)
                intensities_vs_alpha = []

                # 定义固定的k_hat, vec_theta_fixed, vec_phi_fixed
                k_hat_fixed = np.array([
                    np.sin(fixed_theta_rad) * np.cos(phi_rad_fixed),
                    np.sin(fixed_theta_rad) * np.sin(phi_rad_fixed),
                    np.cos(fixed_theta_rad)
                ])
                vec_theta_fixed = np.array([
                    np.cos(fixed_theta_rad) * np.cos(phi_rad_fixed),
                    np.cos(fixed_theta_rad) * np.sin(phi_rad_fixed),
                    -np.sin(fixed_theta_rad)
                ])
                vec_phi_fixed = np.array([
                    -np.sin(phi_rad_fixed),
                    np.cos(phi_rad_fixed),
                    0.0
                ])

                for alpha_val_rad in alpha_scan_rad:
                    # E_omega for linear polarization, alpha is the scan variable
                    E_omega = np.cos(alpha_val_rad) * vec_theta_fixed + np.sin(alpha_val_rad) * vec_phi_fixed
                    
                    Ex, Ey, Ez = E_omega[0], E_omega[1], E_omega[2]
                    E_voigt = np.array([Ex*Ex, Ey*Ey, Ez*Ez, 2*Ey*Ez, 2*Ex*Ez, 2*Ex*Ey], dtype=np.complex128)
                    P_i = np.dot(d_matrix, E_voigt)
                    current_intensity = 0.0
                    detection_pol = self.detection_combo.currentIndex()
                    if detection_pol == 0: current_intensity = np.linalg.norm(P_i)**2
                    else:
                        E_omega_xy_projection = np.array([Ex, Ey, 0.0])
                        E_omega_xy_norm = np.linalg.norm(E_omega_xy_projection)
                        E_incident_parallel_direction_xy = np.array([1.0,0.0,0.0]) if E_omega_xy_norm < 1e-9 else E_omega_xy_projection / E_omega_xy_norm
                        P_xy_complex = np.array([P_i[0], P_i[1], 0.0])
                        if detection_pol == 1:
                            P_parallel_component = np.dot(P_xy_complex, E_incident_parallel_direction_xy)
                            current_intensity = np.abs(P_parallel_component)**2
                        elif detection_pol == 2:
                            E_incident_perpendicular_direction_xy = np.array([-E_incident_parallel_direction_xy[1], E_incident_parallel_direction_xy[0], 0.0])
                            P_perpendicular_component = np.dot(P_xy_complex, E_incident_perpendicular_direction_xy)
                            current_intensity = np.abs(P_perpendicular_component)**2
                    intensities_vs_alpha.append(current_intensity)
                
                self.canvas.axes.plot(np.rad2deg(alpha_scan_rad), intensities_vs_alpha, lw=2, color='blue')
                self.canvas.axes.set_xlabel("偏振角 α (度)")
                self.canvas.axes.set_ylabel("SHG 强度 (任意单位)")
                self.canvas.axes.grid(True)

            elif current_scan_mode == "偏振角扫描 (α-强度极图)":
                # 确保是极坐标轴
                if not isinstance(self.canvas.axes, matplotlib.projections.polar.PolarAxes):
                    if self.canvas.axes: self.canvas.axes.remove()
                    self.axes = self.canvas.fig.add_subplot(111, polar=True) # 改为极坐标轴
                    self.canvas.axes = self.axes
                
                fixed_theta_rad = np.deg2rad(self.fixed_theta_spinbox.value())
                phi_rad_fixed = np.deg2rad(self.phi_slider.value()) 
                alpha_scan_rad = np.linspace(0, 2 * np.pi, 360) # 角度用弧度
                intensities_vs_alpha = []

                # 定义固定的k_hat, vec_theta_fixed, vec_phi_fixed (与笛卡尔α扫描模式相同)
                k_hat_fixed = np.array([
                    np.sin(fixed_theta_rad) * np.cos(phi_rad_fixed),
                    np.sin(fixed_theta_rad) * np.sin(phi_rad_fixed),
                    np.cos(fixed_theta_rad)
                ])
                vec_theta_fixed = np.array([
                    np.cos(fixed_theta_rad) * np.cos(phi_rad_fixed),
                    np.cos(fixed_theta_rad) * np.sin(phi_rad_fixed),
                    -np.sin(fixed_theta_rad)
                ])
                vec_phi_fixed = np.array([
                    -np.sin(phi_rad_fixed),
                    np.cos(phi_rad_fixed),
                    0.0
                ])

                # 计算强度 (与笛卡尔α扫描模式相同)
                for alpha_val_rad in alpha_scan_rad:
                    E_omega = np.cos(alpha_val_rad) * vec_theta_fixed + np.sin(alpha_val_rad) * vec_phi_fixed
                    Ex, Ey, Ez = E_omega[0], E_omega[1], E_omega[2]
                    E_voigt = np.array([Ex*Ex, Ey*Ey, Ez*Ez, 2*Ey*Ez, 2*Ex*Ez, 2*Ex*Ey], dtype=np.complex128)
                    P_i = np.dot(d_matrix, E_voigt)
                    current_intensity = 0.0
                    detection_pol = self.detection_combo.currentIndex()
                    if detection_pol == 0: current_intensity = np.linalg.norm(P_i)**2
                    else:
                        E_omega_xy_projection = np.array([Ex, Ey, 0.0])
                        E_omega_xy_norm = np.linalg.norm(E_omega_xy_projection)
                        E_incident_parallel_direction_xy = np.array([1.0,0.0,0.0]) if E_omega_xy_norm < 1e-9 else E_omega_xy_projection / E_omega_xy_norm
                        P_xy_complex = np.array([P_i[0], P_i[1], 0.0])
                        if detection_pol == 1:
                            P_parallel_component = np.dot(P_xy_complex, E_incident_parallel_direction_xy)
                            current_intensity = np.abs(P_parallel_component)**2
                        elif detection_pol == 2:
                            E_incident_perpendicular_direction_xy = np.array([-E_incident_parallel_direction_xy[1], E_incident_parallel_direction_xy[0], 0.0])
                            P_perpendicular_component = np.dot(P_xy_complex, E_incident_perpendicular_direction_xy)
                            current_intensity = np.abs(P_perpendicular_component)**2
                    intensities_vs_alpha.append(current_intensity)
                
                # 使用极坐标绘图
                self.canvas.axes.plot(alpha_scan_rad, intensities_vs_alpha, lw=2, color='green')
                self.canvas.axes.grid(True)

            else: # "默认扫描 (θ-极图)"
                # Reconfigure axes for Polar plot if necessary
                if not isinstance(self.canvas.axes, matplotlib.projections.polar.PolarAxes):
                    if self.canvas.axes: self.canvas.axes.remove()
                    self.axes = self.canvas.fig.add_subplot(111, polar=True)
                    self.canvas.axes = self.axes
                
                phi_rad_current = np.deg2rad(self.phi_slider.value())
                theta_scan_rad = np.linspace(0, 2 * np.pi, 360)
                P_intensities = []
                current_input_pol_mode = self.input_polarization_combo.currentText()
                alpha_for_incident_rad = np.deg2rad(self.alpha_slider.value())

                for theta_val_rad in theta_scan_rad:
                    vec_theta = np.array([np.cos(theta_val_rad) * np.cos(phi_rad_current), np.cos(theta_val_rad) * np.sin(phi_rad_current), -np.sin(theta_val_rad)])
                    vec_phi = np.array([-np.sin(phi_rad_current), np.cos(phi_rad_current), 0.0])
                    E_omega = np.zeros(3, dtype=np.complex128)
                    if current_input_pol_mode == "线偏振":
                        E_omega = np.cos(alpha_for_incident_rad) * vec_theta + np.sin(alpha_for_incident_rad) * vec_phi
                    elif current_input_pol_mode == "左旋圆偏振 (LCP)":
                        E_omega = (1/np.sqrt(2)) * (vec_theta + 1j * vec_phi)
                    elif current_input_pol_mode == "右旋圆偏振 (RCP)":
                        E_omega = (1/np.sqrt(2)) * (vec_theta - 1j * vec_phi)
                    else: # "默认 (θ-偏振)"
                        E_omega = vec_theta
                    
                    Ex, Ey, Ez = E_omega[0], E_omega[1], E_omega[2]
                    E_voigt = np.array([Ex*Ex, Ey*Ey, Ez*Ez, 2*Ey*Ez, 2*Ex*Ez, 2*Ex*Ey], dtype=np.complex128)
                    P_i = np.dot(d_matrix, E_voigt)
                    current_intensity = 0.0
                    detection_pol = self.detection_combo.currentIndex()
                    if detection_pol == 0: current_intensity = np.linalg.norm(P_i)**2
                    else:
                        # For parallel/perpendicular detection, reference E_omega_xy_projection
                        # Need to decide how to handle complex Ex, Ey for E_omega_xy_projection's direction
                        # Using real parts for the reference direction of incident light polarization in xy plane.
                        E_omega_xy_projection_ref = np.array([Ex.real if np.iscomplex(Ex) else Ex, Ey.real if np.iscomplex(Ey) else Ey, 0.0])
                        E_omega_xy_norm_ref = np.linalg.norm(E_omega_xy_projection_ref)
                        E_incident_parallel_direction_xy = np.array([1.0,0.0,0.0]) if E_omega_xy_norm_ref < 1e-9 else E_omega_xy_projection_ref / E_omega_xy_norm_ref
                        P_xy_complex = np.array([P_i[0], P_i[1], 0.0])
                        if detection_pol == 1:
                            P_parallel_component = np.dot(P_xy_complex, E_incident_parallel_direction_xy)
                            current_intensity = np.abs(P_parallel_component)**2
                        elif detection_pol == 2:
                            E_incident_perpendicular_direction_xy = np.array([-E_incident_parallel_direction_xy[1], E_incident_parallel_direction_xy[0], 0.0])
                            P_perpendicular_component = np.dot(P_xy_complex, E_incident_perpendicular_direction_xy)
                            current_intensity = np.abs(P_perpendicular_component)**2
                    P_intensities.append(current_intensity)
                self.canvas.axes.plot(theta_scan_rad, P_intensities, lw=2, color='purple')
                self.canvas.axes.grid(True, linestyle='--', alpha=0.5)

            # Common drawing and title setting
            self.update_plot_title()
            self.canvas.draw()

        except Exception as e:
            print(f"绘图时发生错误: {e}")
            import traceback
            traceback.print_exc()

    def update_plot_title(self):
        """Helper function to set the plot title based on current mode and parameters."""
        display_name = self.selected_group
        # ... (name sanitization as before) ...
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
        crystal_orientation_text = f"晶体朝向: φ<sub>c</sub>={phi_c:.1f}°, θ<sub>c</sub>={theta_c:.1f}°, ψ<sub>c</sub>={psi_c:.1f}°"

        if scan_mode == "偏振角扫描 (α vs I)":
            fixed_theta_val = self.fixed_theta_spinbox.value()
            title_text = f"{display_name} - {detection_text}\n偏振角扫描 (笛卡尔) @ 光束θ={fixed_theta_val:.1f}°, 光束φ={self.phi:.1f}°\n{crystal_orientation_text}"
        elif scan_mode == "偏振角扫描 (α-强度极图)":
            fixed_theta_val = self.fixed_theta_spinbox.value()
            title_text = f"{display_name} - {detection_text}\n偏振角扫描 (极坐标) @ 光束θ={fixed_theta_val:.1f}°, 光束φ={self.phi:.1f}°\n{crystal_orientation_text}"
        else:  # 默认扫描 (θ-极图)
            input_pol_mode = self.input_polarization_combo.currentText()
            input_pol_text = f"输入: {input_pol_mode}"
            if input_pol_mode == "线偏振" or input_pol_mode == "默认 (θ-偏振)": 
                if self.input_polarization_combo.isVisible(): 
                    input_pol_text += f" (α={self.alpha}°)"
            title_text = f"{display_name} - {detection_text}\n{input_pol_text}, 光束φ={self.phi:.1f}°\n{crystal_orientation_text}"
        
        if hasattr(self.canvas, 'axes') and self.canvas.axes is not None:
            self.canvas.axes.set_title(title_text, pad=20)
        else:
            print("Warning: canvas.axes not available for title setting.")

    def update_scan_mode_controls(self):
        """根据选择的扫描模式，更新相关控件的可见性和状态"""
        mode = self.scan_mode_combo.currentText()
        is_alpha_scan_mode_cartesian = (mode == "偏振角扫描 (α vs I)")
        is_alpha_scan_mode_polar = (mode == "偏振角扫描 (α-强度极图)")
        is_any_alpha_scan_mode = is_alpha_scan_mode_cartesian or is_alpha_scan_mode_polar

        self.fixed_theta_label.setVisible(is_any_alpha_scan_mode)
        self.fixed_theta_spinbox.setVisible(is_any_alpha_scan_mode)

        self.input_polarization_label.setVisible(not is_any_alpha_scan_mode)
        self.input_polarization_combo.setVisible(not is_any_alpha_scan_mode)
        if not is_any_alpha_scan_mode:
            self.update_input_polarization_controls() # 更新alpha控件的可见性
        else:
            self.alpha_control_widget.setVisible(False) # 在任何alpha扫描模式下隐藏alpha控件

        self.plot() # 模式改变时重绘

    def update_input_polarization_controls(self):
        """根据选择的输入偏振模式，显示或隐藏alpha控制"""
        mode = self.input_polarization_combo.currentText()
        is_linear_polarization = (mode == "线偏振")
        
        self.alpha_control_widget.setVisible(is_linear_polarization)
        
        if is_linear_polarization:
            self.alpha_label.setToolTip("线偏振光的电场振动方向与 θ-方向单位矢量之间的夹角。\nθ-方向矢量大致对应p偏振方向(在过Z轴和传播方向的平面内，且垂直于传播方向)。\nφ-方向矢量大致对应s偏振方向(垂直于θ-方向和传播方向)。\nα=0° 表示电场沿 θ-方向； α=90° 表示电场沿 φ-方向。")
        else:
            self.alpha_label.setToolTip("") # 清除Tooltip
        
        # 如果从其他模式切换到线偏振或从线偏振切换到其他模式，可能需要重新绘图
        self.plot()

    def update_alpha_display(self, update_plot=True):
        """更新alpha角度显示"""
        try:
            self.alpha = self.alpha_slider.value()
            self.alpha_display.setText(f"{self.alpha}°")
            if update_plot:
                self.plot()
        except Exception as e:
            print(f"更新alpha角度显示时出错: {e}")

    def open_3d_plot_window(self):
        """打开3D SHG图案绘制窗口"""
        # 传递必要的参数给3D绘图窗口，例如计算好的d_matrix和E_omega
        # 这里暂时只打开一个空窗口，后续再填充数据和绘图逻辑
        if not hasattr(self, 'shg_3d_plot_window_instance') or not self.shg_3d_plot_window_instance.isVisible():
            self.shg_3d_plot_window_instance = SHG3DPlotWindow(self)
            self.shg_3d_plot_window_instance.show()
        else:
            self.shg_3d_plot_window_instance.activateWindow()
            self.shg_3d_plot_window_instance.raise_()

    def open_manual_input(self):
        """打开手动输入窗口"""
        # 检查是否已经有一个手动输入窗口实例，避免重复创建
        if not hasattr(self, 'manual_window_instance') or not self.manual_window_instance.isVisible():
            self.manual_window_instance = ManualInputWindow(self)
            self.manual_window_instance.show()
            self.hide()  # 隐藏主窗口
        else:
            self.manual_window_instance.activateWindow() # 如果已存在且可见，则激活
            self.manual_window_instance.raise_() # 确保在最前
            self.hide()

class SHG3DPlotWindow(QMainWindow): # 使用QMainWindow以便可以有菜单等，或者QDialog也可以
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D SHG强度分布图")
        self.setGeometry(200, 200, 800, 700)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        self.canvas = FigureCanvas(Figure(figsize=(7, 6), dpi=100))
        layout.addWidget(self.canvas)
        
        # 添加3D坐标轴
        # self.ax = self.canvas.figure.add_subplot(111, projection='3d')
        # 延迟初始化3D轴到实际绘图时，避免无数据时创建

        # TODO: Add controls or display info if needed

    def plot_data(self, R_shg, theta_out_grid, phi_out_grid):
        """
        绘制3D SHG数据。
        R_shg: SHG强度，作为半径，是一个二维数组，维度对应theta_out_grid和phi_out_grid
        theta_out_grid: 出射天顶角网格 (radians)
        phi_out_grid: 出射方位角网格 (radians)
        """
        if not hasattr(self, 'ax') or self.ax is None or not isinstance(self.ax, Axes3D):
            self.canvas.figure.clear() # 清除旧的（可能是2D的）轴
            self.ax = self.canvas.figure.add_subplot(111, projection='3d')
        else:
            self.ax.clear() # 清除之前的3D图形

        # 将球面半径转换为笛卡尔坐标
        X = R_shg * np.sin(theta_out_grid) * np.cos(phi_out_grid)
        Y = R_shg * np.sin(theta_out_grid) * np.sin(phi_out_grid)
        Z = R_shg * np.cos(theta_out_grid)

        # 绘制表面
        self.ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', rstride=1, cstride=1, alpha=0.8)

        self.ax.set_xlabel('X_lab')
        self.ax.set_ylabel('Y_lab')
        self.ax.set_zlabel('Z_lab')
        self.ax.set_title('3D SHG Intensity Pattern')
        
        # 设置一个固定的合理的视角
        self.ax.view_init(elev=20., azim=45)
        # 调整 límites para asegurar que el origen (0,0,0) sea visible y la forma sea completa
        max_R = np.max(R_shg) if R_shg.size > 0 else 1.0
        self.ax.set_xlim([-max_R, max_R])
        self.ax.set_ylim([-max_R, max_R])
        self.ax.set_zlim([-max_R, max_R])
        self.ax.set_aspect('auto') # او 'equal' si es preferible

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())