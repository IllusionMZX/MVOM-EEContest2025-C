# Monocular Vision Measurement and Power Monitoring System

[English](#english) / [中文](#chinese)

---

<img src="https://github.com/IllusionMZX/MVOM-EEContest2025-C/blob/main/IMG/image-1.jpg" alt="System Overview Diagram" style="zoom: 33%;" />

## English <a name="english"></a>

### 1. System Overview

This project presents a target measurement device based on monocular vision, designed to accurately measure the distance (D) from a baseline to a target object and the geometric dimensions (w) of the object's surface, while simultaneously monitoring power consumption.

The system uses a Raspberry Pi as the main controller, equipped with a monocular camera to capture images of the target. After correcting for distortion using perspective transformation, different algorithms are applied based on the target type:
- **Single Geometric Shapes:** Dimensions are calculated through contour analysis and shape recognition.
- **Multiple-Disjoint-Squares:** Parameters are extracted by filtering based on geometric features.
- **Partially Overlapping Squares:** A YOLOv8n-seg instance segmentation model is used to separate the contours.
- **Numbered Squares:** A LeNet-5 model is employed for recognition and matching of the numbers.

The hardware integrates a current detection circuit to monitor the supply current in real-time. The current value is converted to a voltage value and sampled by an STM32 ADC. This data, combined with the 5V supply voltage, is used to calculate power consumption. A touch-enabled display screen facilitates mode switching and interaction with the results.

**Hardware Modules:**
- **Main Controller:** Raspberry Pi 4B (4GB RAM), 32GB TF Card, Raspberry Pi OS (64-bit).
- **Microcontroller:** STM32F103C8T6 minimum system board.
- **Camera:** Hikvision USB Camera.
- **Display:** TJC X5 Series 7-inch IPS Touch Screen.

### 2. Raspberry Pi Deployment

This section outlines the steps to set up the software environment on the Raspberry Pi.

**Prerequisites:**
- Raspberry Pi 4B with Raspberry Pi OS (64-bit) installed.
- Internet connection.

**Setup Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/IllusionMZX/MVOM-EEContest2025-C.git
    cd MVOM-EEContest2025-C
    ```

2.  **Install Python Dependencies:**
    The project requires `python>=3.9`. It is recommended to use a virtual environment.
    ```bash
    sudo apt update
    sudo apt install python3-venv -y
    python3 -m venv venv
    source venv/bin/activate
    ```
    Install the required packages from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should contain:
    ```
    ultralytics
    torch
    onnx
    opencv-python
    onnxruntime
    pyserial
    ```

3.  **Run the Main Application:**
    Connect the USB camera, STM32 board (via USB/UART), and the TJC display to the Raspberry Pi. Then, run the main script:
    ```bash
    python main.py
    ```

### 3. Model Training and ONNX Export

This section describes the process for training the models and exporting them to ONNX format. The training scripts can be found in the `Model-Training-with-OnnxExport` directory.

**Environment Setup:**
- A PC or server with a GPU is recommended for training.
- Python >= 3.9.
- Install necessary libraries: `ultralytics`, `pytorch`, `onnx`, `onnxruntime`, `opencv-python`.

**Models:**

1.  **LeNet-5 for Numbered Squares:**
    - **Description:** This model is trained on the **MNIST handwritten digit dataset** to recognize the numbers on the squares.
    - **Code:** The training and export scripts are located in `Model-Training-with-OnnxExport/Digital-Squares-LeNet`.
    - **Process:** Train the LeNet-5 model using PyTorch and then use `torch.onnx.export()` to convert the trained model to `lenet.onnx`.

2.  **YOLOv8n-seg for Overlapping Squares:**
    - **Description:** This instance segmentation model is trained on a **custom dataset** to detect and separate overlapping squares.
    - **Code:** The training and export scripts are in `Model-Training-with-OnnxExport/Overlapping-Squares-Yolov8n-seg`.
    - **Dataset Style:** The custom dataset consists of images with partially overlapping squares, as shown below.
      ![Custom Dataset Sample](https://github.com/IllusionMZX/MVOM-EEContest2025-C/blob/main/IMG/image-3.jpg)
    - **Process:** Use the `ultralytics` library to train the YOLOv8n-seg model. After training, export the best model to `yolov8n-seg.onnx` using the `model.export(format='onnx')` command.

### 4. Current Detection Circuit Design

The current detection circuit is designed to monitor the system's power consumption. It is built around a high-precision current-sense amplifier, such as the **MAX4372FESA+** or a similar module.

**Design Overview:**
- The sensor is placed in series with the 5V power supply line.
- It measures the voltage drop across a small shunt resistor and outputs an analog voltage proportional to the current.
- This analog signal is fed into an ADC pin on the STM32F103C8T6 for digital conversion.

**Schematic:**
![Current Detection Circuit Schematic](IMG/imgae-2.jpg)

**STM32 Integration:**
- The STM32 project is developed using **STM32CubeIDE**.
- The IDE is used for code editing, peripheral configuration (using the graphical CubeMX tool), and programming the microcontroller.
- The STM32 firmware samples the voltage from the sensor, converts it to a current reading, and transmits the value to the Raspberry Pi via UART.

### 5. Power Consumption Measurement and Display

This module integrates the data from the current detection circuit to calculate and display real-time power consumption.

**Process Flow:**
1.  **Data Acquisition (STM32):** The STM32 continuously reads the analog voltage from the current sensor via its ADC.
2.  **Data Transmission:** The calculated current value (in Amperes) is sent from the STM32 to the Raspberry Pi over a UART serial port.
3.  **Data Reception (Raspberry Pi):** A Python script on the Raspberry Pi uses the `pyserial` library to listen to the serial port and receive the current data.
4.  **Power Calculation:** The script calculates power using the formula: `Power (W) = Voltage (V) × Current (A)`, where the voltage is a constant 5V.
5.  **Display:** The calculated power value, along with the primary measurement results, is sent to the TJC touch screen for display using its specific communication protocol.

---

## 中文 <a name="chinese"></a>

### 1. 系统整体概述

本项目设计了一款基于单目视觉的目标物测量装置，旨在实现对基准线到目标物距离（D）及物面几何图形尺寸（w）的精准测量，并同步监测系统功耗。

系统以树莓派为主控，搭载单目摄像头采集目标物图像。图像经透视变换校正畸变后，针对不同目标类型采用对应算法：
- **单个几何图形：** 通过轮廓分析与形状识别计算尺寸。
- **多个分离正方形：** 通过几何特征筛选提取参数。
- **局部重叠正方形：** 借助 YOLOv8n-seg 实例分割模型实现轮廓分离。
- **数字编号正方形：** 通过 LeNet-5 模型完成数字识别与匹配。

硬件集成了电流检测电路，实时监测供电电流。电流值被转换为电压值后，由STM32的ADC进行采集。该数据结合5V供电电压计算出实时功耗。系统通过一块触控显示屏实现人机交互与模式切换。

**硬件模块:**
- **主控核心：** 树莓派4B (4GB内存版), 32GB TF卡, 树莓派官方64位系统。
- **微控制器：** STM32F103C8T6 最小系统板。
- **摄像头：** 海康威视USB摄像头。
- **显示屏：** 淘晶驰X5系列7寸IPS触摸屏。

### 2. 树莓派部署

本节介绍在树莓派上配置软件环境的步骤。

**环境要求:**
- 已安装官方64位Raspberry Pi OS的树莓派4B。
- 网络连接。

**部署步骤:**

1.  **克隆代码仓库:**
    ```bash
    git clone https://github.com/IllusionMZX/MVOM-EEContest2025-C.git
    cd MVOM-EEContest2025-C
    ```

2.  **安装Python依赖:**
    项目要求 `python>=3.9`。推荐使用虚拟环境以隔离依赖。
    ```bash
    sudo apt update
    sudo apt install python3-venv -y
    python3 -m venv venv
    source venv/bin/activate
    ```
    使用 `requirements.txt` 文件安装所有必要的库：
    ```bash
    pip install -r requirements.txt
    ```
    `requirements.txt` 文件应包含以下内容：
    ```
    ultralytics
    torch
    onnx
    opencv-python
    onnxruntime
    pyserial
    ```

3.  **运行主程序:**
    将USB摄像头、STM32（通过USB/UART）和淘晶驰串口屏连接到树莓派。然后执行主脚本：
    ```bash
    python main.py
    ```

### 3. 模型训练及ONNX导出

本节描述如何训练模型并将其导出为ONNX格式。所有训练脚本位于 `Model-Training-with-OnnxExport` 目录下。

**环境配置:**
- 推荐使用带GPU的PC或服务器进行模型训练。
- Python >= 3.9。
- 安装必要的库: `ultralytics`, `pytorch`, `onnx`, `onnxruntime`, `opencv-python`。

**模型详情:**

1.  **用于数字方块识别的LeNet-5模型:**
    - **描述:** 该模型基于 **MNIST手写数字数据集** 进行训练，用于识别方块上的数字。
    - **代码:** 训练和导出脚本位于 `Model-Training-with-OnnxExport/Digital-Squares-LeNet`。
    - **流程:** 使用PyTorch训练LeNet-5模型，然后调用 `torch.onnx.export()` 函数将训练好的模型转换为 `lenet.onnx`。

2.  **用于重叠方块分割的YOLOv8n-seg模型:**
    - **描述:** 该实例分割模型基于一个 **自定义数据集** 进行训练，用于检测并分离重叠的方块。
    - **代码:** 训练和导出脚本位于 `Model-Training-with-OnnxExport/Overlapping-Squares-Yolov8n-seg`。
    - **数据集样式:** 自定义数据集包含部分重叠的方块图像，样式如下图所示。
      ![自定义数据集样例](IMG/imgae-3.jpg)
    - **流程:** 使用 `ultralytics` 库训练YOLOv8n-seg模型。训练完成后，使用 `model.export(format='onnx')` 命令将最佳模型导出为 `yolov8n-seg.onnx`。

### 4. 电流检测电路设计

电流检测电路用于监测系统的整体功耗，其核心是一个高精度电流检测放大器（例如 **MAX4372FESA+** 或功能类似的模块）。

**设计概述:**
- 传感器串联在5V供电线路上。
- 它通过测量一个微小采样电阻上的压降，输出一个与电流成正比的模拟电压。
- 此模拟信号被送入STM32F103C8T6的ADC引脚进行数字化转换。

**电路原理图:**
![电流检测电路原理图](https://github.com/IllusionMZX/MVOM-EEContest2025-C/blob/main/IMG/image-2.jpg)

**STM32集成:**
- STM32的工程项目使用 **STM32CubeIDE** 进行开发。
- 该IDE用于代码编辑、外设配置（通过图形化的CubeMX工具）以及程序烧录。
- STM32固件对来自传感器的电压进行采样，将其转换为电流读数，并通过UART串口发送给树莓派。

### 5. 功耗测量与显示

该模块整合来自电流检测电路的数据，以计算并实时显示功耗。

**处理流程:**
1.  **数据采集 (STM32):** STM32通过其ADC持续读取电流传感器的模拟电压。
2.  **数据传输:** 计算出的电流值（单位：安培）通过UART串口从STM32发送到树莓派。
3.  **数据接收 (树莓派):** 树莓派上的Python脚本使用 `pyserial` 库监听串口并接收电流数据。
4.  **功耗计算:** 脚本使用公式 `功率 (W) = 电压 (V) × 电流 (A)` 计算功耗，其中电压为恒定的5V。
5.  **数据显示:** 计算出的功耗值，连同核心的测量结果，被一同发送到淘晶驰触摸屏上，并遵循其特定的通信协议进行显示。