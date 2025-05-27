# SHG 非线性极化模拟器 (SHGSimulator)

## 简介

SHGSimulator 是一个用 Python 和 PyQt5 开发的图形用户界面应用程序，用于模拟和可视化材料的二次谐波产生 (SHG) 非线性极化效应。用户可以选择不同的晶体点群、调整张量分量、设置入射光参数，并观察产生的SHG强度极图。

## 主要特性

*   **图形用户界面**：基于 PyQt5 构建，操作直观。
*   **点群选择**：支持多种常见晶体点群，自动确定非零张量分量。
*   **常见晶体预设**：内置LiNbO3, KDP等常见晶体的参数，方便快速加载。
*   **参数可调**：
    *   独立调整非零张量分量的相对强度。
    *   调整晶体欧拉角 (φc, θc, ψc) 以改变晶体朝向。
    *   选择不同的扫描模式：
        *   入射角扫描 (θ-极图)
        *   偏振角扫描 (α vs SHG强度，笛卡尔图)
        *   偏振角扫描 (α-强度极图)
        *   3D θinc-αinc 扫描
    *   对于入射角扫描：可设置光束方位角 (φ) 和入射光偏振态 (默认θ偏振, 线偏振及角度α, 左旋/右旋圆偏振)。
    *   对于偏振角扫描：可设置固定的入射天顶角 (θinc) 和入射方位角 (φinc)。
*   **检测偏振分析**：可分析总SHG强度，或平行/垂直于入射光偏振方向的SHG分量。
*   **实时可视化**：使用 Matplotlib 动态绘制SHG强度图样。
*   **手动张量输入模式**：允许用户直接输入 3x6 的 dij 张量矩阵。
*   **3D SHG 图样显示**：可根据当前参数绘制3D的SHG辐射方向图或 (θinc, αinc) 强度扫描图。
*   **内置Logo**：界面中包含IPE课题组Logo。
*   **可打包为EXE**：使用 PyInstaller 将程序打包为单文件可执行程序，方便在Windows上分发和运行。

## 运行环境与依赖

*   Python 3.9+ (推荐 Python 3.11)
*   PyQt5
*   NumPy
*   Matplotlib
*   PyInstaller (用于打包)

**安装依赖 (示例使用 pip):**
```bash
pip install PyQt5 numpy matplotlib
```

## 如何运行

### 1. 从源代码运行

1.  克隆或下载本仓库。
2.  确保已安装上述依赖库。
3.  在项目根目录下打开终端，运行：
    ```bash
    python src/main.py
    ```

### 2. 运行已打包的 EXE (Windows)

1.  从 `dist` 文件夹中找到 `SHGSimulator.exe`。
2.  直接双击运行即可。程序启动可能需要几秒钟。

## 如何打包 (开发者)

如果你修改了源代码并希望重新打包为EXE：

1.  确保已安装 PyInstaller:
    ```bash
    pip install pyinstaller
    ```
2.  在项目根目录下，确保 `IPE_logo.png` 文件存在。
3.  运行打包命令：
    ```bash
    pyinstaller SHGSimulator.spec
    ```
    或者，如果第一次打包或不使用 `.spec` 文件：
    ```bash
    pyinstaller --onefile --windowed --name SHGSimulator --add-data "data;data" --add-data "IPE_logo.png;." src/main.py
    ```
    使用 `.spec` 文件 (`SHGSimulator.spec`) 是推荐的方式，因为它包含了更详细的配置。

## 项目结构

```
shg_simulation/
├── src/
│   ├── main.py               # 主程序和GUI逻辑
│   ├── point_groups.py       # 点群数据和张量处理
│   ├── visualization.py      # (目前可能部分功能已整合到main.py)
│   └── point_group_data.json # 点群对称性数据 (在data文件夹内)
├── data/
│   └── point_group_data.json # 点群对称性数据
├── dist/                     # (打包后生成) 包含EXE文件
├── build/                    # (打包后生成) 临时构建文件
├── IPE_logo.png              # Logo图片
├── SHGSimulator.spec         # PyInstaller 配置文件
├── LICENSE                   # MIT 许可证文件
└── README.md                 # 本文件
```

## 技术栈

*   **Python 3**: 主要编程语言。
*   **PyQt5**: 用于构建图形用户界面。
*   **NumPy**: 用于高效的数值计算和张量操作。
*   **Matplotlib**: 用于数据可视化和绘图。

## 未来可能的改进

*   **更精确的材料数据库**: 引入包含波长依赖的、更广泛的非线性材料的精确 d<sub>ij</sub> 系数值。
*   **相位匹配条件**: 考虑相位匹配对SHG效率的影响。
*   **多层膜结构**: 支持多层薄膜样品中的SHG模拟。
*   **聚焦光束**: 模拟高斯光束等聚焦光束下的SHG。
*   **输出数据导出**: 允许用户将计算结果或图表数据导出为文本或图像文件。
*   **更高级的3D可视化**: 使用更专业的3D可视化库（如 Mayavi, VisPy）以获得更好的性能和交互性。
*   **单位和物理常数**: 明确物理量的单位，并在计算中引入相关物理常数。

## 贡献

欢迎通过提交 Pull Request 或创建 Issues 来贡献代码或提出建议。

## 许可证

本项目采用 [MIT许可证](LICENSE) (如果计划添加LICENSE文件)。