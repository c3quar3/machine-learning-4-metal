import wave
import threading
import datetime
import os
import tkinter as tk
from tkinter import ttk, messagebox

import serial
import serial.tools.list_ports
import pyaudio

from config import GATHER_DATA_DIR, SR

class DataCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("声音信号采集系统 ")
        self.root.geometry("450x620") 
        
        # --- 变量初始化 ---
        self.serial_port = None
        self.is_recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.auto_stop_id = None  
        self.current_speed = "0"  
        
        # 录音参数
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = SR
        self.SAVE_DIR = GATHER_DATA_DIR #保存文件夹
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

        self.setup_ui()

    def setup_ui(self):
        # ================= 1. 串口连接区 =================
        frame_serial = tk.LabelFrame(self.root, text="1. 硬件连接 (串口)", padx=10, pady=10)
        frame_serial.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_serial, text="选择串口:").grid(row=0, column=0, sticky="w")
        self.port_combo = ttk.Combobox(frame_serial, width=15)
        self.port_combo.grid(row=0, column=1, padx=5)
        self.refresh_ports()

        tk.Button(frame_serial, text="刷新", command=self.refresh_ports).grid(row=0, column=2, padx=5)
        self.btn_connect = tk.Button(frame_serial, text="连接", command=self.toggle_serial, bg="#90EE90")
        self.btn_connect.grid(row=0, column=3, padx=5)

        # ================= 2. 马达转速控制区 =================
        frame_motor = tk.LabelFrame(self.root, text="2. 马达转速设置", padx=10, pady=10)
        frame_motor.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_motor, text="输入转速:").grid(row=0, column=0, sticky="w")
        self.speed_entry = tk.Entry(frame_motor, width=15)
        self.speed_entry.insert(0, "0") 
        self.speed_entry.grid(row=0, column=1, padx=5)

        tk.Button(frame_motor, text="发送指令 / 更新转速", command=self.send_speed).grid(row=0, column=2, padx=5)
        tk.Button(frame_motor, text="紧急停止电机", command=self.stop_motor, fg="red").grid(row=1, column=0, columnspan=3, pady=5)

        # ================= 3. 实验参数设置区 =================
        frame_exp = tk.LabelFrame(self.root, text="3. 实验与文件参数", padx=10, pady=10)
        frame_exp.pack(fill="x", padx=10, pady=5)

        # 材质选择
        tk.Label(frame_exp, text="摩擦球材质:").grid(row=0, column=0, sticky="w", pady=5)
        self.ball_var = tk.StringVar(value="Alball")
        tk.Radiobutton(frame_exp, text="铝球 (Alball)", variable=self.ball_var, value="Alball").grid(row=0, column=1, sticky="w")
        tk.Radiobutton(frame_exp, text="钢球 (Stball)", variable=self.ball_var, value="Stball").grid(row=0, column=2, sticky="w")

        # 录制时间
        tk.Label(frame_exp, text="设定录音时间(秒):").grid(row=1, column=0, sticky="w", pady=5)
        self.time_entry = tk.Entry(frame_exp, width=10)
        self.time_entry.insert(0, "10") 
        self.time_entry.grid(row=1, column=1, sticky="w")

        # ================= 4. 数据采集控制区 =================
        frame_audio = tk.LabelFrame(self.root, text="4. 数据采集", padx=10, pady=10)
        frame_audio.pack(fill="both", expand=True, padx=10, pady=5)

        # 麦克风选择区
        mic_frame = tk.Frame(frame_audio)
        mic_frame.pack(fill="x", pady=5)
        tk.Label(mic_frame, text="选择录音设备:").pack(side="left")
        self.mic_combo = ttk.Combobox(mic_frame, width=30)
        self.mic_combo.pack(side="left", padx=5)
        tk.Button(mic_frame, text="刷新", command=self.refresh_mics).pack(side="left")
        self.refresh_mics() # 初始化时获取麦克风列表

        self.btn_record = tk.Button(frame_audio, text="● 开始定时录音", font=("Arial", 12, "bold"), bg="#ff9999", command=self.start_recording)
        self.btn_record.pack(pady=10, fill="x")

        self.btn_stop_record = tk.Button(frame_audio, text="⏹ 提前结束录音", font=("Arial", 10), command=self.stop_recording, state="disabled")
        self.btn_stop_record.pack(pady=5, fill="x")

        self.status_label = tk.Label(frame_audio, text="状态: 待机中...", fg="blue", font=("Arial", 10))
        self.status_label.pack(pady=5)

    # --- 麦克风选择功能 (新增) ---
    def refresh_mics(self):
        """扫描系统中所有的音频输入设备"""
        mics = []
        info = self.audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        
        for i in range(0, numdevices):
            # 只要 maxInputChannels > 0，就说明它是麦克风（输入设备）
            if (self.audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                try:
                    # 获取设备名称，处理可能出现的中文乱码
                    device_name = self.audio.get_device_info_by_host_api_device_index(0, i).get('name')
                    # 格式： "编号: 设备名"
                    mics.append(f"{i}: {device_name}")
                except Exception:
                    mics.append(f"{i}: 未知设备")
                    
        self.mic_combo['values'] = mics
        if mics:
            self.mic_combo.current(0) # 默认选中第一个
        else:
            self.mic_combo.set("未找到麦克风")

    # --- 串口与马达控制 ---
    def refresh_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)

    def toggle_serial(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.btn_connect.config(text="连接", bg="#90EE90")
            self.status_label.config(text="串口已断开")
        else:
            port = self.port_combo.get()
            if not port:
                messagebox.showerror("错误", "请选择串口")
                return
            try:
                self.serial_port = serial.Serial(port, 9600, timeout=1)
                self.btn_connect.config(text="断开", bg="#ff9999")
                self.status_label.config(text=f"已连接至 {port}")
            except Exception as e:
                messagebox.showerror("串口错误", f"无法打开串口:\n{e}")

    def send_speed(self):
        try:
            # 1. 尝试将输入转换为浮点数 (float) 而不是整数 (int)
            speed_val = float(self.speed_entry.get())
            
            # 2. 智能格式化：如果是像 1500.0 这样的整数，就去掉小数点变成 1500
            if speed_val.is_integer():
                speed_str = str(int(speed_val))
            else:
                speed_str = str(speed_val)
                
            self.current_speed = speed_str # 保存当前转速用于命名
            
            if self.serial_port and self.serial_port.is_open:
                # 组合指令，例如 "V1500.5\n"
                command = f"V{speed_str}\n"
                self.serial_port.write(command.encode('utf-8'))
                self.status_label.config(text=f"已发送转速指令: {speed_str}")
            else:
                messagebox.showwarning("警告", "串口未连接！\n(当前仅记录转速数值用于文件命名)")
        except ValueError:
            # 3. 修改报错提示
            messagebox.showerror("格式错误", "转速必须是有效的数字（可以是整数或小数）！")

    def stop_motor(self):
        self.speed_entry.delete(0, tk.END)
        self.speed_entry.insert(0, "0")
        self.send_speed()

    # --- 录音功能 ---
    def start_recording(self):
        if self.is_recording:
            return

        # 1. 检查时间输入
        try:
            record_time = int(self.time_entry.get())
            if record_time <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("格式错误", "录音时间必须是大于0的整数(秒)！")
            return

        # 2. 生成文件路径
        ball_type = self.ball_var.get()
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        sub_folder = "alumi" if ball_type == "Alball" else "steel"
        target_dir = os.path.join(self.SAVE_DIR, sub_folder)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        self.filename = os.path.join(target_dir, f"{ball_type}_v={self.current_speed}_{timestamp}.wav")

        # 3. ✅ 获取选中的麦克风编号
        selected_mic = self.mic_combo.get()
        device_index = None # 默认不传参数
        if selected_mic and ":" in selected_mic:
            try:
                # 提取字符串前面的编号，例如 "1: USB Microphone" 提取出 1
                device_index = int(selected_mic.split(":")[0])
            except ValueError:
                pass

        # 4. 打开音频流
        try:
            self.stream = self.audio.open(format=self.FORMAT, 
                                          channels=self.CHANNELS,
                                          rate=self.RATE, 
                                          input=True, 
                                          input_device_index=device_index, # ✅ 传入麦克风编号
                                          frames_per_buffer=self.CHUNK)
        except Exception as e:
            messagebox.showerror("麦克风错误", f"无法打开指定的音频设备:\n{e}\n请尝试刷新或选择其他设备。")
            return

        self.is_recording = True
        self.frames = []
        
        # 更新UI
        self.btn_record.config(state="disabled")
        self.btn_stop_record.config(state="normal")
        self.status_label.config(text=f"🔴 录音中... (设定时间: {record_time}秒)", fg="red")

        # 启动后台录音线程
        threading.Thread(target=self._record_thread, daemon=True).start()

        # 设定倒计时自动停止
        self.auto_stop_id = self.root.after(record_time * 1000, self.stop_recording)

    def _record_thread(self):
        while self.is_recording:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self.frames.append(data)
            except Exception:
                break

    def stop_recording(self):
        if not self.is_recording:
            return

        if self.auto_stop_id:
            self.root.after_cancel(self.auto_stop_id)
            self.auto_stop_id = None

        self.is_recording = False
        self.status_label.config(text="正在保存数据...", fg="orange")
        self.root.update()

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        # 保存文件
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        self.btn_record.config(state="normal")
        self.btn_stop_record.config(state="disabled")
        self.status_label.config(text=f" 保存成功: {os.path.basename(self.filename)}", fg="green")

    def on_closing(self):
        if self.is_recording:
            self.stop_recording()
        self.stop_motor() 
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.audio.terminate()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
