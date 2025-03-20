import sys
import numpy as np
import matplotlib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QSlider, QPushButton, QComboBox, 
                            QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox, QDial)
from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from visualization import plot_polarization_intensity
from point_groups import (point_group_components, create_tensor_with_relations, 
                        get_all_point_groups, get_components_for_group, str_to_indices)

# 检查LaTeX是否可用 - 更安全的方法
def is_latex_available():
    try:
        # 简单检查是否安装了latex命令
        import shutil
        return shutil.which('latex') is not None
    except:
        return False

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']  # 优先使用更完整的字体
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
        
        # 左侧控制面板
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.main_layout.addWidget(self.controls_widget, 1)
        
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
        
        # 方位角控制
        self.phi_label = QLabel("方位角 φ (度):")
        self.params_layout.addWidget(self.phi_label, 0, 0)
        
        self.phi_slider = QSlider(Qt.Horizontal)
        self.phi_slider.setMinimum(0)
        self.phi_slider.setMaximum(359)
        self.phi_slider.setValue(0)
        self.phi_slider.valueChanged.connect(self.update_phi_display)
        self.params_layout.addWidget(self.phi_slider, 0, 1)
        
        self.phi_display = QLabel("0°")
        self.params_layout.addWidget(self.phi_display, 0, 2)
        
        # 张量分量强度调整
        self.tensor_label = QLabel("张量分量强度:")
        self.params_layout.addWidget(self.tensor_label, 1, 0)
        
        self.tensor_slider = QSlider(Qt.Horizontal)
        self.tensor_slider.setMinimum(1)
        self.tensor_slider.setMaximum(20)
        self.tensor_slider.setValue(10)
        self.tensor_slider.valueChanged.connect(self.update_tensor_display)
        self.params_layout.addWidget(self.tensor_slider, 1, 1)
        
        self.tensor_display = QLabel("1.0")
        self.params_layout.addWidget(self.tensor_display, 1, 2)
        
        # 检测偏振选择
        self.detection_label = QLabel("检测偏振:")
        self.params_layout.addWidget(self.detection_label, 2, 0)
        
        self.detection_combo = QComboBox()
        self.detection_combo.addItems(["总强度 |P|²", "平行模式 (∥)", "垂直模式 (⊥)"])
        self.detection_combo.currentIndexChanged.connect(self.plot)
        self.params_layout.addWidget(self.detection_combo, 2, 1, 1, 2)
        
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
            comp_str = ", ".join([f"χ<sub>{c}</sub>" for c in components])
            self.components_list.setText(comp_str)
            
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
                    
                    # 添加标签
                    label = QLabel(f"χ{comp}:")
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
        """绘制极化强度极图"""
        # 检查是否已初始化完成
        if not hasattr(self, 'base_tensor') or self.base_tensor is None:
            return
        
        if not hasattr(self, 'phi'):
            self.phi = 0.0
            
        try:
            # 应用分量独立系数
            tensor = self.base_tensor.copy()
            for comp, value in self.component_values.items():
                try:
                    indices = str_to_indices(comp)
                    tensor[indices] = tensor[indices] * value
                except Exception as e:
                    print(f"处理分量 {comp} 时出错: {e}")
            
            # 应用全局系数
            global_scale = self.tensor_slider.value() / 10.0
            tensor = tensor * global_scale
            
            # 清除当前图形
            self.canvas.axes.clear()
            
            # 绘制新图
            phi_rad = self.phi * np.pi/180  # 转换为弧度
            theta_range = np.linspace(0, 2*np.pi, 360)
            
            # 计算极化强度
            P = []
            for theta in theta_range:
                # 计算入射电场（球坐标系）
                E = (np.sin(theta)*np.cos(phi_rad),
                     np.sin(theta)*np.sin(phi_rad),
                     np.cos(theta))
                
                # 张量缩并计算 P_i = χ_ijk E_j E_k 
                P_i = np.einsum('ijk,j,k->i', tensor, E, E)
                
                # 获取检测偏振选择
                detection_pol = self.detection_combo.currentIndex()
                
                if detection_pol == 0:  # 总强度 |P|²
                    intensity = np.linalg.norm(P_i)**2
                else:
                    # 平行和垂直模式检偏 - 只考虑xy平面
                    # 获取E在xy平面的投影
                    E_xy = np.array([E[0], E[1], 0.0])
                    E_xy_norm = np.linalg.norm(E_xy)
                    
                    # 如果E_xy几乎为零向量（电场几乎垂直于xy平面），使用单位x向量作为参考
                    if E_xy_norm < 1e-10:
                        E_xy_direction = np.array([1.0, 0.0, 0.0])
                    else:
                        # 归一化得到xy平面内的方向向量
                        E_xy_direction = E_xy / E_xy_norm
                    
                    # 计算P在xy平面内的分量
                    P_xy = np.array([P_i[0], P_i[1], 0.0])
                    
                    if detection_pol == 1:  # 平行模式 (∥) - xy平面内平行于E_xy的分量
                        # 计算P在E_xy方向的投影
                        P_parallel_xy = np.dot(P_xy, E_xy_direction)
                        intensity = P_parallel_xy**2
                    elif detection_pol == 2:  # 垂直模式 (⊥) - xy平面内垂直于E_xy的分量
                        # 在xy平面内，与E_xy垂直的单位向量
                        if E_xy_norm < 1e-10:
                            E_perp_direction = np.array([0.0, 1.0, 0.0])  # 如果E在z方向，选y作为垂直方向
                        else:
                            # 在xy平面内，逆时针旋转90度
                            E_perp_direction = np.array([-E_xy_direction[1], E_xy_direction[0], 0.0])
                        
                        # 计算P在垂直方向的投影
                        P_perp_xy = np.dot(P_xy, E_perp_direction)
                        intensity = P_perp_xy**2
                
                P.append(intensity)
            
            # 绘制极图
            self.canvas.axes.plot(theta_range, P, lw=2, color='purple')
            
            # 处理点群名称，使用普通文本格式
            # 移除特殊Unicode下标字符，使用普通字符代替
            display_name = self.selected_group
            display_name = display_name.replace("₁", "1").replace("₂", "2").replace("₃", "3")
            display_name = display_name.replace("₄", "4").replace("₆", "6")
            display_name = display_name.replace("ᵥ", "v").replace("ₕ", "h").replace("ₘ", "m")
            display_name = display_name.replace("ᵤ", "u").replace("ₐ", "a").replace("ᵢ", "i")
            display_name = display_name.replace("̄", "")  # 移除上划线符号
            
            # 获取检测偏振的文本描述
            detection_text = self.detection_combo.currentText()
            
            # 更新标题，包含检测偏振信息
            title_text = f"{display_name}\n非线性极化强度 - {detection_text}\nφ={self.phi:.1f}°"
                
            self.canvas.axes.set_title(title_text, pad=20)
            self.canvas.axes.grid(True, linestyle='--', alpha=0.5)
            self.canvas.draw()
        except Exception as e:
            print(f"绘图时发生错误: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())