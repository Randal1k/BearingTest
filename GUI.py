import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import threading
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

import tool_monitor as tm
from tool_monitor import (
    init_training,
    init_prediction,
    load_file,
    signal_FFT,
    process_signal,
    load_model,
    predict_condition,
    save_model
)

# Global Vars

root = None
notebook = None
current_mode = None
model_type = None
selected_model = None
training_folder = None
test_file = None
viz_file = None
progress = None
status_label = None
results_text = None
fig = None
canvas = None
train_button = None
load_model_button = None
test_button = None
model_combo = None
learning_frame = None
testing_frame = None
is_running = False
predict_function = None


def create_main_window():
    """Create and configure the main window"""
    global root
    root = tk.Tk()
    root.title("Tool Condition Monitor")
    root.geometry("800x600")
    root.minsize(600, 500)

    # Configure style
    style = ttk.Style()
    style.theme_use('clam')

    initialize_variables()
    create_widgets()
    refresh_available_models()


def initialize_variables():
    """Initialize all global variables"""
    global current_mode, model_type, selected_model, training_folder, test_file, viz_file, is_running, predict_function

    current_mode = tk.StringVar(value="learning")
    model_type = tk.StringVar(value="random_forest")
    selected_model = tk.StringVar()
    training_folder = tk.StringVar()
    test_file = tk.StringVar()
    viz_file = tk.StringVar()
    is_running = False
    predict_function = None


def create_widgets():
    """Create all GUI widgets"""
    global notebook

    # Create main notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Create tabs
    create_main_tab()
    create_visualization_tab()
    create_results_tab()


def create_main_tab():
    """Create the main tab with learning and testing modes"""
    global learning_frame, testing_frame, train_button, load_model_button, test_button, model_combo, progress, status_label

    # Main tab
    main_frame = ttk.Frame(notebook)
    notebook.add(main_frame, text="Main")

    # Mode selection frame
    mode_frame = ttk.LabelFrame(main_frame, text="Select Mode", padding="10")
    mode_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Radiobutton(mode_frame, text="Learning Mode",
                    variable=current_mode, value="learning",
                    command=on_mode_change).pack(side=tk.LEFT, padx=10)
    ttk.Radiobutton(mode_frame, text="Testing Mode",
                    variable=current_mode, value="testing",
                    command=on_mode_change).pack(side=tk.LEFT, padx=10)

    # Learning mode frame
    learning_frame = ttk.LabelFrame(main_frame, text="Learning Configuration", padding="10")
    learning_frame.pack(fill=tk.X, padx=10, pady=5)

    # Model type selection
    ttk.Label(learning_frame, text="Model Type:").grid(row=0, column=0, sticky=tk.W, pady=2)
    model_combo_learning = ttk.Combobox(learning_frame, textvariable=model_type,
                                        values=["random_forest", "svm"], state="readonly", width=20)
    model_combo_learning.grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)

    # Training folder selection
    ttk.Label(learning_frame, text="Training Data Folder:").grid(row=1, column=0, sticky=tk.W, pady=2)
    ttk.Entry(learning_frame, textvariable=training_folder, width=40).grid(row=1, column=1, sticky=tk.W, padx=10,
                                                                           pady=2)
    ttk.Button(learning_frame, text="Browse",
               command=browse_training_folder).grid(row=1, column=2, padx=5, pady=2)

    # Start training button
    train_button = ttk.Button(learning_frame, text="Start Training",
                              command=start_training)
    train_button.grid(row=2, column=0, columnspan=3, pady=10)

    # Testing mode frame
    testing_frame = ttk.LabelFrame(main_frame, text="Testing Configuration", padding="10")
    testing_frame.pack(fill=tk.X, padx=10, pady=5)

    # Model selection
    ttk.Label(testing_frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
    model_combo = ttk.Combobox(testing_frame, textvariable=selected_model,
                               state="readonly", width=40)
    model_combo.grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
    ttk.Button(testing_frame, text="Refresh",
               command=refresh_available_models).grid(row=0, column=2, padx=5, pady=2)

    # Test file selection
    ttk.Label(testing_frame, text="Test Data File:").grid(row=1, column=0, sticky=tk.W, pady=2)
    ttk.Entry(testing_frame, textvariable=test_file, width=40).grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)
    ttk.Button(testing_frame, text="Browse",
               command=browse_test_file).grid(row=1, column=2, padx=5, pady=2)

    # Load model and test buttons
    button_frame = ttk.Frame(testing_frame)
    button_frame.grid(row=2, column=0, columnspan=3, pady=10)

    load_model_button = ttk.Button(button_frame, text="Load Model",
                                   command=load_model)
    load_model_button.pack(side=tk.LEFT, padx=5)

    test_button = ttk.Button(button_frame, text="Run Test",
                             command=run_test)
    test_button.pack(side=tk.LEFT, padx=5)

    # Progress bar
    progress_frame = ttk.Frame(main_frame)
    progress_frame.pack(fill=tk.X, padx=10, pady=5)

    progress = ttk.Progressbar(progress_frame, mode='indeterminate')
    progress.pack(fill=tk.X)

    # Status label
    status_label = ttk.Label(main_frame, text="Ready")
    status_label.pack(pady=5)

    # Set initial mode
    on_mode_change()


def create_visualization_tab():
    """Create the visualization tab"""
    global fig, canvas

    # Visualization tab
    viz_frame = ttk.Frame(notebook)
    notebook.add(viz_frame, text="Signal Visualization")

    # File selection for visualization
    file_frame = ttk.LabelFrame(viz_frame, text="Signal File", padding="10")
    file_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Entry(file_frame, textvariable=viz_file, width=50).pack(side=tk.LEFT, padx=5)
    ttk.Button(file_frame, text="Browse", command=browse_viz_file).pack(side=tk.LEFT, padx=5)
    ttk.Button(file_frame, text="Plot Signal", command=plot_signal).pack(side=tk.LEFT, padx=5)

    # Matplotlib canvas
    fig = Figure(figsize=(10, 8))
    canvas = FigureCanvasTkAgg(fig, viz_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)


def create_results_tab():
    """Create the results tab"""
    global results_text

    # Results tab
    results_frame = ttk.Frame(notebook)
    notebook.add(results_frame, text="Results")

    # Results display
    results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD,
                                             height=20, width=80)
    results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Clear button
    ttk.Button(results_frame, text="Clear Results",
               command=clear_results).pack(pady=5)

def on_mode_change():
    """Handle mode change between learning and testing"""
    if current_mode.get() == "learning":
        learning_frame.pack(fill=tk.X, padx=10, pady=5)
        testing_frame.pack_forget()
    else:
        testing_frame.pack(fill=tk.X, padx=10, pady=5)
        learning_frame.pack_forget()

def browse_training_folder():
    """Browse for training data folder"""
    folder = filedialog.askdirectory(title="Select Training Data Folder")
    if folder:
        training_folder.set(folder)

def browse_test_file():
    """Browse for test data file"""
    file = filedialog.askopenfilename(
        title="Select Test Data File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if file:
        test_file.set(file)

def browse_viz_file():
    """Browse for visualization file"""
    file = filedialog.askopenfilename(
        title="Select Signal File for Visualization",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if file:
        viz_file.set(file)


def refresh_available_models():
    """Refresh the list of available models"""
    try:
        # This would call your list_available_models() function
        # models = tm.list_available_models()

        # For demo purposes, add some example models
        models = []
        model_dir = 'res/model'
        if os.path.exists(model_dir):
            for root_dir, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith('.pkl'):
                        relative_path = os.path.relpath(os.path.join(root_dir, file), model_dir)
                        models.append(relative_path)

        model_combo['values'] = models
        if models and not selected_model.get():
            selected_model.set(models[0])

    except Exception as e:
        log_message(f"Error refreshing models: {str(e)}")


def start_training():
    """Start model training in a separate thread"""
    global is_running

    if not training_folder.get():
        messagebox.showerror("Error", "Please select a training data folder")
        return

    if not os.path.exists(training_folder.get()):
        messagebox.showerror("Error", "Training data folder does not exist")
        return

    if is_running:
        messagebox.showwarning("Warning", "Training is already in progress")
        return

    # Start training in separate thread
    is_running = True
    train_button.config(state='disabled')
    progress.start(10)
    status_label.config(text="Training in progress...")

    thread = threading.Thread(target=training_worker)
    thread.daemon = True
    thread.start()


def training_worker():
    """Worker function for training (runs in separate thread)"""
    try:
        log_message("Starting training...")
        log_message(f"Training folder: {training_folder.get()}")
        log_message(f"Model type: {model_type.get()}")

        # Replace with your actual training function call
        tm.init_training(training_folder.get(),model_type.get())

        # Simulate training time
        import time
        time.sleep(3)

        log_message("Training completed successfully!")

    except Exception as e:
        log_message(f"Training error: {str(e)}")

    finally:
        # Update UI in main thread
        root.after(0, training_finished)


def training_finished():
    """Called when training is finished"""
    global is_running

    is_running = False
    train_button.config(state='normal')
    progress.stop()
    status_label.config(text="Training completed")
    refresh_available_models()


def load_model():
    """Load selected model"""
    global predict_function

    if not selected_model.get():
        messagebox.showerror("Error", "Please select a model")
        return

    try:
        status_label.config(text="Loading model...")

        model_path = os.path.join('res', 'model', selected_model.get())
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Replace with your actual model loading
        predict_function = tm.init_prediction(model_path)

        log_message(f"Model loaded: {selected_model.get()}")
        status_label.config(text="Model loaded successfully")

    except Exception as e:
        log_message(f"Error loading model: {str(e)}")
        status_label.config(text="Error loading model")


def run_test():
    """Run test on selected file"""
    if not test_file.get():
        messagebox.showerror("Error", "Please select a test file")
        return

    if not os.path.exists(test_file.get()):
        messagebox.showerror("Error", "Test file does not exist")
        return

    if predict_function is None:
        messagebox.showerror("Error", "Please load a model first")
        return

    try:
        status_label.config(text="Running test...")

        result = predict_function(test_file.get())

        # Replace with your actual prediction function call
        # result = tm.predict_condition(test_file.get())

        display_test_results(result)
        status_label.config(text="Test completed")

    except Exception as e:
        log_message(f"Test error: {str(e)}")
        status_label.config(text="Test failed")


def display_test_results(result):
    """Display test results in results tab"""
    notebook.select(2)  # Switch to results tab

    results_text_content = f"""
=== TEST RESULTS ===
File: {test_file.get()}
Model: {selected_model.get()}

PREDICTION:
Condition: {result['condition'].upper()}
Confidence: {result['confidence']:.2%}

PROBABILITIES:
Normal: {result['probabilities']['normal']:.2%}
Unbalance: {result['probabilities']['unbalance']:.2%}
Misalignment: {result['probabilities']['misalignment']:.2%}
Bearing: {result['probabilities']['bearing']:.2%}

RECOMMENDATIONS:
"""
    for rec in result['recommendations']:
        results_text_content += f"• {rec}\n"

    results_text_content += "\n" + "=" * 50 + "\n\n"

    results_text.insert(tk.END, results_text_content)
    results_text.see(tk.END)


def plot_signal():
    """Plot signal visualization"""
    if not viz_file.get():
        messagebox.showerror("Error", "Please select a signal file")
        return

    if not os.path.exists(viz_file.get()):
        messagebox.showerror("Error", "Signal file does not exist")
        return

    try:
        # Clear previous plots
        fig.clear()

        # Replace with your actual signal loading and processing
        # data = tm.load_file(viz_file.get())

        # Simulate loading data
        data = pd.DataFrame({
            't': np.linspace(0, 1, 1000),
            'x': np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000)) + 0.1 * np.random.randn(1000),
            'y': np.sin(2 * np.pi * 15 * np.linspace(0, 1, 1000)) + 0.1 * np.random.randn(1000),
            'z': np.sin(2 * np.pi * 8 * np.linspace(0, 1, 1000)) + 0.1 * np.random.randn(1000)
        })

        # Create subplots
        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3, 2, 4)
        ax5 = fig.add_subplot(3, 2, 5)
        ax6 = fig.add_subplot(3, 2, 6)

        # Time domain plots
        ax1.plot(data['t'], data['x'])
        ax1.set_title('X-axis Signal')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')

        ax3.plot(data['t'], data['y'])
        ax3.set_title('Y-axis Signal')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')

        ax5.plot(data['t'], data['z'])
        ax5.set_title('Z-axis Signal')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Amplitude')

        # FFT plots (simplified)
        for i, (axis, ax) in enumerate([(data['x'], ax2), (data['y'], ax4), (data['z'], ax6)]):
            fft_vals = np.fft.fft(axis)
            freqs = np.fft.fftfreq(len(axis), 1 / 1000)
            ax.plot(freqs[:len(freqs) // 2], np.abs(fft_vals)[:len(freqs) // 2])
            ax.set_title(f'{"XYZ"[i]}-axis FFT')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude')

        fig.tight_layout()
        canvas.draw()

        # Switch to visualization tab
        notebook.select(1)

    except Exception as e:
        messagebox.showerror("Error", f"Error plotting signal: {str(e)}")


def log_message(message):
    """Log message to results tab"""
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    results_text.insert(tk.END, log_entry)
    results_text.see(tk.END)


def clear_results():
    """Clear results text area"""
    results_text.delete(1.0, tk.END)


def main():
    """Main function to run the application"""
    create_main_window()
    root.mainloop()

if __name__ == "__main__":
    main()