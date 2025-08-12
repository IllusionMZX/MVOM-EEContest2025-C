# Monocular Vision Measurement and Power Monitoring System

[English](#english) / [中文](#chinese)

---



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

[Back to Top](#monocular-vision-measurement-and-power-monitoring-system) | [中文 Version](#chinese)

### 2. Raspberry Pi Deployment

This section outlines the steps to set up the software environment on the Raspberry Pi.

**Prerequisites:**
- Raspberry Pi 4B with Raspberry Pi OS (64-bit) installed.
- Internet connection.

**Setup Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
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

[Back to Top](#monocular-vision-measurement-and-power-monitoring-system) | [中文 Version](#chinese)

### 3. Model Training and ONNX Export

This section describes the process for training the custom models (YOLOv8n-seg, LeNet-5) and exporting them to the ONNX (Open Neural Network Exchange) format for optimized inference on the Raspberry Pi.

**Environment Setup:**
- A PC or server with a GPU is recommended for training.
- Python >= 3.9.
- Install necessary libraries: `ultralytics`, `pytorch`, `onnx`, `onnxruntime`, `opencv-python`.

**Training Process (Example for YOLOv8):**

1.  **Prepare Your Dataset:** Organize your images and labels in the format required by YOLOv8.
2.  **Train the Model:** Use the `ultralytics` CLI or Python SDK to start the training process.
    
    ```python
    from ultralytics import YOLO
    
    # Load a pretrained model
    model = YOLO('yolov8n-seg.pt') 
    
    # Train the model
    results = model.train(data='your_dataset.yaml', epochs=100, imgsz=640)
    ```
3.  **Export to ONNX:** After training, export the best-performing model (`best.pt`) to ONNX format.
    ```python
    from ultralytics import YOLO
    
    # Load the trained model
    model = YOLO('path/to/your/best.pt')
    
    # Export the model to ONNX format
    model.export(format='onnx')
    ```
    This will generate a `best.onnx` file, which can then be deployed to the Raspberry Pi for inference. The process for LeNet-5 is similar, involving training with PyTorch and then using `torch.onnx.export()` to convert the model.

[Back to Top](#monocular-vision-measurement-and-power-monitoring-system) | [中文 Version](#chinese)

### 4. Current Detection Circuit Design

The current detection circuit is a crucial hardware component for monitoring the system's power consumption.

**Design Overview:**
- A high-precision current sense amplifier (e.g., INA219 or a similar module) is used.
- The sensor is placed in series with the 5V power supply line of the Raspberry Pi.
- The sensor measures the voltage drop across a small shunt resistor.
- The output of the sensor is an analog voltage signal proportional to the measured current.
- This analog signal is fed into an ADC pin on the STM32F103C8T6 microcontroller for digital conversion.

**STM32 Integration:**
- The STM32 project is developed using **STM32CubeIDE**.
- The IDE is used for code editing, configuration (using CubeMX for pin setup), and programming (flashing) the microcontroller.
- Within the STM32 code, the ADC is configured to continuously sample the voltage from the current sensor.
- The sampled digital value is then converted back to a current reading based on the sensor's specifications.
- This current value is transmitted to the Raspberry Pi, typically via a UART serial connection.

[Back to Top](#monocular-vision-measurement-and-power-monitoring-system) | [中文 Version](#chinese)

### 5. Power Consumption Measurement and Display

This module integrates the data from the current detection circuit to calculate and display real-time power consumption.

**Process Flow:**
1.  **Data Acquisition (STM32):** The STM32 continuously reads the analog voltage from the current sensor via its ADC.
2.  **Data Transmission:** The calculated current value (in Amperes) is sent from the STM32 to the Raspberry Pi over a UART serial port.
3.  **Data Reception (Raspberry Pi):** A Python script on the Raspberry Pi listens to the serial port to receive the current data. The `pyserial` library is used for this purpose.
4.  **Power Calculation:** The Raspberry Pi script calculates the power using the formula:
    `Power (W) = Voltage (V) × Current (A)`
    The voltage is a constant 5V, and the current is the value received from the STM32.
5.  **Display:** The calculated power value, along with the primary measurement results (distance D, width w), is sent to the TJC touch screen for display. The communication protocol specific to the TJC screen is used to update the UI elements.

[Back to Top](#monocular-vision-measurement-and-power-monitoring-system) | [中文 Version](#chinese)

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

[返回顶部](#monocular-vision-measurement-and-power-monitoring-system) | [English Version](#english)

### 2. 树莓派部署

本节介绍在树莓派上配置软件环境的步骤。

**环境要求:**
- 已安装官方64位Raspberry Pi OS的树莓派4B。
- 网络连接。

**部署步骤:**

1.  **克隆代码仓库:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
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

[返回顶部](#monocular-vision-measurement-and-power-monitoring-system) | [English Version](#english)

### 3. 模型训练及ONNX导出

本节描述如何训练自定义模型（YOLOv8n-seg, LeNet-5），并将其导出为ONNX（开放神经网络交换）格式，以便在树莓派上进行优化推理。

**环境配置:**
- 推荐使用带GPU的PC或服务器进行模型训练。
- Python >= 3.9。
- 安装必要的库: `ultralytics`, `pytorch`, `onnx`, `onnxruntime`, `opencv-python`。

**训练流程 (以YOLOv8为例):**

1.  **准备数据集:** 按照YOLOv8要求的数据集格式组织您的图片和标签文件。
2.  **训练模型:** 使用 `ultralytics` 的命令行工具或Python SDK开始训练。
    ```python
    from ultralytics import YOLO
    
    # 加载一个预训练模型
    model = YOLO('yolov8n-seg.pt') 
    
    # 训练模型
    results = model.train(data='your_dataset.yaml', epochs=100, imgsz=640)
    ```
3.  **导出为ONNX:** 训练完成后，将表现最好的模型权重（通常是`best.pt`）导出为ONNX格式。
    ```python
    from ultralytics import YOLO
    
    # 加载已训练好的模型
    model = YOLO('path/to/your/best.pt')
    
    # 将模型导出为ONNX格式
    model.export(format='onnx')
    ```
    该操作会生成一个 `best.onnx` 文件，此文件即可拷贝到树莓派上用于推理。LeNet-5的训练与导出流程类似，通常涉及使用PyTorch进行训练，然后调用 `torch.onnx.export()` 函数完成转换。

[返回顶部](#monocular-vision-measurement-and-power-monitoring-system) | [English Version](#english)

### 4. 电流检测电路设计

电流检测电路是监测系统功耗的关键硬件部分。

**设计概述:**
- 采用高精度电流检测放大器（例如INA219或功能类似的模块）。
- 传感器串联在树莓派的5V供电线路上。
- 传感器通过测量一个微小的采样电阻上的压降来计算电流。
- 传感器的输出是一个与被测电流成正比的模拟电压信号。
- 此模拟信号被送入STM32F103C8T6微控制器的ADC引脚进行数字化转换。

**STM32集成:**
- STM32的工程项目使用 **STM32CubeIDE** 进行开发。
- 该IDE用于代码编辑、引脚和外设配置（通过CubeMX图形化工具）以及程序烧录。
- 在STM32的固件代码中，ADC被配置为连续采样来自电流传感器的电压信号。
- 采样到的数字值根据传感器的规格被转换回电流读数。
- 计算出的电流值通过UART串口通信发送给树莓派。

[返回顶部](#monocular-vision-measurement-and-power-monitoring-system) | [English Version](#english)

### 5. 功耗测量与显示

该模块整合来自电流检测电路的数据，以计算并实时显示功耗。

**处理流程:**
1.  **数据采集 (STM32):** STM32通过其ADC持续读取电流传感器的模拟电压。
2.  **数据传输:** 计算出的电流值（单位：安培）通过UART串口从STM32发送到树莓派。
3.  **数据接收 (树莓派):** 树莓派上的Python脚本监听指定的串口，以接收电流数据。此功能通过 `pyserial` 库实现。
4.  **功耗计算:** 树莓派脚本使用以下公式计算功耗：
    `功率 (W) = 电压 (V) × 电流 (A)`
    其中，电压为恒定的5V，电流为从STM32接收到的值。
5.  **数据显示:** 计算出的功耗值，连同核心的测量结果（距离D、宽度W），一同被发送到淘晶驰触摸屏上进行显示。更新UI界面时，需遵循该屏幕特定的通信协议。

[返回顶部](#monocular-vision-measurement-and-power-monitoring-system) | [English Version](#english)