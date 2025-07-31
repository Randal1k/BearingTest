import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk

import threading
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

import tool_monitor as tm
import RUL as rul  # Import the new RUL module

# Set the appearance mode and default color theme
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# Global Vars
root = None
notebook = None
current_mode = None
model_type = None
selected_model = None
selected_rul_model = None  # New RUL model selection
training_folder = None
test_file = None
viz_file = None
viz_file_axis = None
custom_model_name = None
is_running = False
predict_function = None
rul_predict_function = None  # New RUL prediction function

current_file = 0
total_files = 0
start_time = None

learning_frame = None
testing_frame = None
train_button = None
train_rul_button = None  # New RUL training button
load_model_button = None
load_rul_model_button = None  # New RUL model loading button
test_button = None
model_combo = None
rul_model_combo = None  # New RUL model combo
progress_bar = None
progress_detail_label = None
status_label = None
results_text = None
fig = None
canvas = None


def create_main_window():
    """Create and configure the main window"""
    global root
    root = ctk.CTk()
    root.title("Tool Condition Monitor with RUL Prediction")
    root.geometry("1200x900")  # Increased size for RUL features
    root.minsize(1000, 700)

    initialize_variables()
    create_widgets()
    refresh_available_models()
    on_mode_change()


def initialize_variables():
    """Initialize all global variables"""
    global current_mode, model_type, selected_model, selected_rul_model, training_folder, viz_file_axis, test_file, viz_file, is_running, predict_function, rul_predict_function, custom_model_name
    global progress_frame, status_frame

    current_mode = ctk.StringVar(value="learning")
    model_type = ctk.StringVar(value="random_forest")
    viz_file_axis = ctk.StringVar(value="x")
    selected_model = ctk.StringVar()
    selected_rul_model = ctk.StringVar()  # New RUL model variable
    training_folder = ctk.StringVar()
    test_file = ctk.StringVar()
    viz_file = ctk.StringVar()
    custom_model_name = ctk.StringVar()


def create_widgets():
    """Create all GUI widgets"""
    global notebook

    # Create main tabview
    notebook = ctk.CTkTabview(root, width=900, height=700)
    notebook.pack(fill="both", expand=True, padx=20, pady=20)

    # Create tabs
    create_guide_tab()
    create_main_tab()
    create_visualization_tab()
    create_results_tab()


def create_guide_tab():
    """Create the user guide tab"""
    # Guide tab
    guide_tab = notebook.add("📖 User Guide")

    # Create scrollable text widget for the guide
    guide_text = ctk.CTkTextbox(guide_tab, wrap="word", height=500, width=750,
                                font=ctk.CTkFont(size=16))
    guide_text.pack(fill="both", expand=True, padx=15, pady=15)

    # Insert the comprehensive user guide
    try:
        with open("res/guide.md", "r", encoding="utf-8") as f:
            guide_content = f.read()
        guide_text.insert("1.0", guide_content)
    except FileNotFoundError:
        guide_text.insert("1.0", "User guide file not found. Please ensure 'res/guide.md' exists.")


def create_main_tab():
    """Create the main tab with learning and testing modes"""
    global learning_frame, testing_frame, train_button, train_rul_button, load_model_button, load_rul_model_button
    global test_button, model_combo, rul_model_combo, progress_bar, progress_detail_label, status_label
    global progress_frame, status_frame

    # Main tab frame
    main_tab = notebook.add("🔧 Main")

    # Create scrollable frame for main content
    scrollable_frame = ctk.CTkScrollableFrame(main_tab, width=850, height=600)
    scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Mode selection frame
    mode_frame = ctk.CTkFrame(scrollable_frame)
    mode_frame.pack(fill="x", padx=10, pady=10)

    mode_label = ctk.CTkLabel(mode_frame, text="Select Mode", font=ctk.CTkFont(size=14, weight="bold"))
    mode_label.pack(pady=(10, 5))

    mode_button_frame = ctk.CTkFrame(mode_frame, fg_color="transparent")
    mode_button_frame.pack(pady=(0, 10))

    learning_radio = ctk.CTkRadioButton(mode_button_frame, text="Learning Mode",
                                        variable=current_mode, value="learning",
                                        command=on_mode_change, font=ctk.CTkFont(size=11))
    learning_radio.pack(side="left", padx=15, pady=5)

    testing_radio = ctk.CTkRadioButton(mode_button_frame, text="Testing Mode",
                                       variable=current_mode, value="testing",
                                       command=on_mode_change, font=ctk.CTkFont(size=11))
    testing_radio.pack(side="left", padx=15, pady=5)

    # Learning mode frame
    learning_frame = ctk.CTkFrame(scrollable_frame)
    learning_frame.pack(fill="x", padx=10, pady=10)

    learning_label = ctk.CTkLabel(learning_frame, text="Learning Configuration",
                                  font=ctk.CTkFont(size=16, weight="bold"))
    learning_label.pack(pady=(10, 15))

    # Model type selection
    model_type_frame = ctk.CTkFrame(learning_frame)
    model_type_frame.pack(fill="x", padx=20, pady=5)

    ctk.CTkLabel(model_type_frame, text="Model Type:",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10, pady=10)
    model_combo_learning = ctk.CTkComboBox(model_type_frame, variable=model_type,
                                           values=["random_forest", "svm"], width=200)
    model_combo_learning.pack(side="left", padx=10, pady=10)

    # Training folder selection
    folder_frame = ctk.CTkFrame(learning_frame)
    folder_frame.pack(fill="x", padx=20, pady=5)

    ctk.CTkLabel(folder_frame, text="Training Data Folder:",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10, pady=10)
    training_entry = ctk.CTkEntry(folder_frame, textvariable=training_folder, width=300)
    training_entry.pack(side="left", padx=10, pady=10)
    browse_folder_btn = ctk.CTkButton(folder_frame, text="Browse", width=80,
                                      command=browse_training_folder)
    browse_folder_btn.pack(side="left", padx=5, pady=10)

    # Custom model name
    name_frame = ctk.CTkFrame(learning_frame)
    name_frame.pack(fill="x", padx=20, pady=5)

    ctk.CTkLabel(name_frame, text="Model Name (optional):",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10, pady=10)
    custom_name_entry = ctk.CTkEntry(name_frame, textvariable=custom_model_name, width=300)
    custom_name_entry.pack(side="left", padx=10, pady=10)
    ctk.CTkLabel(name_frame, text="Leave empty for auto-naming",
                 text_color="gray").pack(side="left", padx=5, pady=10)

    # Training buttons
    button_frame = ctk.CTkFrame(learning_frame)
    button_frame.pack(pady=20)

    train_button = ctk.CTkButton(button_frame, text="🚀 Train Condition Model", height=40,
                                 font=ctk.CTkFont(size=12, weight="bold"),
                                 command=start_training)
    train_button.pack(side="left", padx=10)

    train_rul_button = ctk.CTkButton(button_frame, text="⏱️ Train RUL Model", height=40,
                                     font=ctk.CTkFont(size=12, weight="bold"),
                                     command=start_rul_training)
    train_rul_button.pack(side="left", padx=10)

    # Testing mode frame
    testing_frame = ctk.CTkFrame(scrollable_frame)
    testing_frame.pack(fill="x", padx=10, pady=10)

    testing_label = ctk.CTkLabel(testing_frame, text="Testing Configuration",
                                 font=ctk.CTkFont(size=16, weight="bold"))
    testing_label.pack(pady=(10, 15))

    # Model selection for condition prediction
    model_select_frame = ctk.CTkFrame(testing_frame)
    model_select_frame.pack(fill="x", padx=20, pady=5)

    ctk.CTkLabel(model_select_frame, text="Condition Model:",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10, pady=10)
    model_combo = ctk.CTkComboBox(model_select_frame, variable=selected_model, width=250)
    model_combo.pack(side="left", padx=10, pady=10)

    load_model_button = ctk.CTkButton(model_select_frame, text="📁 Load", width=80,
                                      command=load_model)
    load_model_button.pack(side="left", padx=5, pady=10)

    # RUL model selection
    rul_model_select_frame = ctk.CTkFrame(testing_frame)
    rul_model_select_frame.pack(fill="x", padx=20, pady=5)

    ctk.CTkLabel(rul_model_select_frame, text="RUL Model:",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10, pady=10)
    rul_model_combo = ctk.CTkComboBox(rul_model_select_frame, variable=selected_rul_model, width=250)
    rul_model_combo.pack(side="left", padx=10, pady=10)

    load_rul_model_button = ctk.CTkButton(rul_model_select_frame, text="📁 Load", width=80,
                                          command=load_rul_model)
    load_rul_model_button.pack(side="left", padx=5, pady=10)

    # Refresh button for models
    refresh_frame = ctk.CTkFrame(testing_frame)
    refresh_frame.pack(fill="x", padx=20, pady=5)

    refresh_btn = ctk.CTkButton(refresh_frame, text="🔄 Refresh Models", width=150,
                                command=refresh_available_models)
    refresh_btn.pack(pady=10)

    # Test file selection
    test_file_frame = ctk.CTkFrame(testing_frame)
    test_file_frame.pack(fill="x", padx=20, pady=5)

    ctk.CTkLabel(test_file_frame, text="Test Data File:",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10, pady=10)
    test_entry = ctk.CTkEntry(test_file_frame, textvariable=test_file, width=300)
    test_entry.pack(side="left", padx=10, pady=10)
    browse_test_btn = ctk.CTkButton(test_file_frame, text="Browse", width=80,
                                    command=browse_test_file)
    browse_test_btn.pack(side="left", padx=5, pady=10)

    # Test button
    test_button_frame = ctk.CTkFrame(testing_frame)
    test_button_frame.pack(pady=20)

    test_button = ctk.CTkButton(test_button_frame, text="🧪 Run Complete Analysis", height=40,
                                font=ctk.CTkFont(size=14, weight="bold"),
                                command=run_test)
    test_button.pack()

    # Progress section
    progress_frame = ctk.CTkFrame(scrollable_frame)
    progress_frame.pack(fill="x", padx=10, pady=10)

    progress_label = ctk.CTkLabel(progress_frame, text="Progress",
                                  font=ctk.CTkFont(size=16, weight="bold"))
    progress_label.pack(pady=(10, 5))

    # Progress bar
    progress_bar = ctk.CTkProgressBar(progress_frame, width=600, height=20)
    progress_bar.pack(padx=20, pady=(5, 0))
    progress_bar.set(0)

    # Status label
    status_frame = ctk.CTkFrame(scrollable_frame)
    status_frame.pack(fill="x", padx=10, pady=0)
    status_label = ctk.CTkLabel(status_frame, text="Ready",
                                font=ctk.CTkFont(size=12, weight="bold"))
    status_label.pack(pady=10)


def create_visualization_tab():
    """Create the visualization tab for signal plotting"""
    global fig, canvas

    # Visualization tab
    viz_tab = notebook.add("📊 Signal Visualization")

    # File selection for visualization
    file_frame = ctk.CTkFrame(viz_tab)
    file_frame.pack(fill="x", padx=10, pady=10)

    file_label = ctk.CTkLabel(file_frame, text="Signal File",
                              font=ctk.CTkFont(size=16, weight="bold"))
    file_label.pack(pady=(10, 5))

    file_select_frame = ctk.CTkFrame(file_frame)
    file_select_frame.pack(pady=(0, 10))

    viz_entry = ctk.CTkEntry(file_select_frame, textvariable=viz_file, width=400)
    viz_entry.pack(side="left", padx=10, pady=10)

    browse_viz_btn = ctk.CTkButton(file_select_frame, text="Browse", width=80,
                                   command=browse_viz_file)
    browse_viz_btn.pack(side="left", padx=5, pady=10)

    plot_btn = ctk.CTkButton(file_select_frame, text="📈 Plot Signal", width=120,
                             font=ctk.CTkFont(size=12, weight="bold"),
                             command=plot_signal)
    plot_btn.pack(side="left", padx=5, pady=10)

    axis_frame = ctk.CTkFrame(viz_tab)
    axis_frame.pack(pady=(0, 10))
    ctk.CTkLabel(axis_frame, text="Select axis:",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10, pady=5)
    axis_combo = ctk.CTkComboBox(axis_frame, variable=viz_file_axis, values=["x", "y", "z"], width=150)
    axis_combo.pack(side="left", padx=10, pady=10)

    # Matplotlib canvas frame
    canvas_frame = ctk.CTkFrame(viz_tab)
    canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # Matplotlib canvas
    fig = Figure(figsize=(12, 8), facecolor='white')
    canvas = FigureCanvasTkAgg(fig, canvas_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)


def create_results_tab():
    """Create the results tab for displaying outputs"""
    global results_text

    # Results tab
    results_tab = notebook.add("📋 Results")

    # Results display
    results_text = ctk.CTkTextbox(results_tab, wrap="word", height=450, width=750,
                                  font=ctk.CTkFont(family="Courier", size=14))
    results_text.pack(fill="both", expand=True, padx=10, pady=10)

    # Clear button
    clear_btn = ctk.CTkButton(results_tab, text="🗑️ Clear Results", height=35,
                              font=ctk.CTkFont(size=12, weight="bold"),
                              command=lambda: results_text.delete("1.0", "end"))
    clear_btn.pack(pady=10)


def on_mode_change():
    """Handle mode change between learning and testing"""

    learning_frame.pack_forget()
    testing_frame.pack_forget()

    if current_mode.get() == "learning":
        learning_frame.pack(fill=tk.X, padx=10, pady=5)
    else:
        testing_frame.pack(fill=tk.X, padx=10, pady=5)

    # Always pack progress frame at the end
    progress_frame.pack(fill="x", padx=10, pady=10)
    status_frame.pack(fill="x", padx=10, pady=5)


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
        condition_models = []
        rul_models = []
        model_dir = 'res/model'

        if os.path.exists(model_dir):
            for root_dir, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith('.pkl'):
                        relative_path = os.path.relpath(os.path.join(root_dir, file), model_dir)
                        if '_rul' in file:
                            rul_models.append(relative_path)
                        else:
                            condition_models.append(relative_path)

        if model_combo:
            model_combo.configure(values=condition_models)
            if condition_models and not selected_model.get():
                selected_model.set(condition_models[0])
                model_combo.set(condition_models[0])

        if rul_model_combo:
            rul_model_combo.configure(values=rul_models)
            if rul_models and not selected_rul_model.get():
                selected_rul_model.set(rul_models[0])
                rul_model_combo.set(rul_models[0])

    except Exception as e:
        log_message(f"Error refreshing models: {str(e)}")


def count_csv_files(folder_path):
    """Count total CSV files in all subfolders"""
    total = 0
    try:
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
                total += len(csv_files)
    except Exception as e:
        log_message(f"Error counting files: {str(e)}")
    return total


def format_time(seconds):
    """Format time in MM:SS format"""
    if seconds < 0:
        return "00:00"
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def update_progress(current, total, elapsed_time=None):
    """Update progress bar and details"""

    def _update():
        if total > 0:
            # Only update the progress bar, no text
            progress_bar.set(current / total)  # CTkProgressBar uses 0-1 scale

    # Schedule the GUI update in the main thread (CRITICAL for thread safety)
    root.after(0, _update)


def start_training():
    """Start condition model training in a separate thread"""
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
    train_button.configure(state='disabled')
    train_rul_button.configure(state='disabled')
    progress_bar.set(0)
    status_label.configure(text="Training condition model...")

    thread = threading.Thread(target=training_worker)
    thread.daemon = True
    thread.start()


def start_rul_training():
    """Start RUL model training in a separate thread"""
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

    # Start RUL training in separate thread
    is_running = True
    train_button.configure(state='disabled')
    train_rul_button.configure(state='disabled')
    progress_bar.set(0)
    status_label.configure(text="Training RUL model...")

    thread = threading.Thread(target=rul_training_worker)
    thread.daemon = True
    thread.start()


def training_worker():
    """Worker function for condition training (runs in separate thread)"""
    try:
        log_message("Starting condition model training...")
        log_message(f"Training folder: {training_folder.get()}")
        log_message(f"Model type: {model_type.get()}")

        # Generate model filename
        if custom_model_name.get().strip():
            # Use custom name but keep it in the model type folder
            custom_name = custom_model_name.get().strip()
            # Remove .pkl extension if user added it
            if custom_name.endswith('.pkl'):
                custom_name = custom_name[:-4]
            model_filename = f"{model_type.get()}/{custom_name}.pkl"
            log_message(f"Using custom model name: {custom_name}.pkl in {model_type.get()}/ folder")
        else:
            # Auto-generate name with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            model_filename = f"{model_type.get()}/tool_monitor_{model_type.get()}_{timestamp}.pkl"
            log_message(f"Auto-generated model name: tool_monitor_{model_type.get()}_{timestamp}.pkl")

        # Count total files for progress tracking
        total_files = count_csv_files(training_folder.get())
        log_message(f"Found {total_files} files to process")

        # Replace with your actual training function call
        feature_list, labels = tm.load_data_with_progress(
            training_folder.get(),
            update_progress  # Pass the progress callback
        )
        X, y, feature_names = tm.prepare_training_data(feature_list, labels)
        tm.train_model(X, y, model_type.get())
        tm.save_model(model_filename, feature_list, True)

        log_message("Condition model training completed successfully!")
        log_message(f"Model saved as: {model_filename}")

    except Exception as e:
        log_message(f"Training error: {str(e)}")

    finally:
        # Update UI in main thread
        root.after(0, training_finished)


def rul_training_worker():
    """Worker function for RUL training (runs in separate thread)"""
    try:
        log_message("Starting RUL model training...")
        log_message(f"Training folder: {training_folder.get()}")
        log_message(f"Model type: {model_type.get()}")

        # Generate RUL model filename
        if custom_model_name.get().strip():
            custom_name = custom_model_name.get().strip()
            if custom_name.endswith('.pkl'):
                custom_name = custom_name[:-4]
            rul_model_filename = f"{model_type.get()}/{custom_name}_rul.pkl"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            rul_model_filename = f"{model_type.get()}/tool_rul_{model_type.get()}_{timestamp}.pkl"

        log_message(f"RUL model will be saved as: {rul_model_filename}")

        # Train RUL model
        saved_filepath = rul.init_rul_training(
            training_folder.get(),
            model_type.get(),
            update_progress
        )

        log_message("RUL model training completed successfully!")
        log_message(f"RUL model saved as: {saved_filepath}")

    except Exception as e:
        log_message(f"RUL training error: {str(e)}")

    finally:
        # Update UI in main thread
        root.after(0, training_finished)


def training_finished():
    """Called when training is finished"""
    global is_running

    is_running = False
    train_button.configure(state='normal')
    train_rul_button.configure(state='normal')
    progress_bar.set(1.0)
    status_label.configure(text="Training completed")
    refresh_available_models()


def load_model():
    """Load selected condition model"""
    global predict_function

    if not selected_model.get():
        messagebox.showerror("Error", "Please select a condition model")
        return

    try:
        status_label.configure(text="Loading condition model...")

        model_path = os.path.join('res', 'model', selected_model.get())
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Replace with your actual model loading
        predict_function = tm.init_prediction(model_path)

        log_message(f"Condition model loaded: {selected_model.get()}")
        status_label.configure(text="Condition model loaded successfully")

    except Exception as e:
        log_message(f"Error loading condition model: {str(e)}")
        status_label.configure(text="Error loading condition model")


def load_rul_model():
    """Load selected RUL model"""
    global rul_predict_function

    if not selected_rul_model.get():
        messagebox.showerror("Error", "Please select a RUL model")
        return

    try:
        status_label.configure(text="Loading RUL model...")

        rul_model_path = os.path.join('res', 'model', selected_rul_model.get())
        if not os.path.exists(rul_model_path):
            raise FileNotFoundError(f"RUL model file not found: {rul_model_path}")

        # Load RUL model
        rul_predict_function = rul.init_rul_prediction(rul_model_path)

        log_message(f"RUL model loaded: {selected_rul_model.get()}")
        status_label.configure(text="RUL model loaded successfully")

    except Exception as e:
        log_message(f"Error loading RUL model: {str(e)}")
        status_label.configure(text="Error loading RUL model")


def run_test():
    """Run complete analysis (condition + RUL)"""
    if not test_file.get():
        messagebox.showerror("Error", "Please select a test file")
        return

    if not os.path.exists(test_file.get()):
        messagebox.showerror("Error", "Test file does not exist")
        return

    if predict_function is None:
        messagebox.showerror("Error", "Please load a condition model first")
        return

    try:
        status_label.configure(text="Running analysis...")

        # Run condition prediction
        condition_result = predict_function(test_file.get())

        # Run RUL prediction if model is loaded
        rul_result = None
        if rul_predict_function is not None:
            rul_result = rul_predict_function(test_file.get(), condition_result)

        display_complete_results(condition_result, rul_result)
        status_label.configure(text="Analysis completed")

    except Exception as e:
        log_message(f"Test error: {str(e)}")
        status_label.configure(text="Test failed")


def display_complete_results(condition_result, rul_result=None):
    """Display complete test results including RUL in results tab"""
    notebook.set("📋 Results")  # Switch to results tab

    results_content = f"""
=== COMPLETE TOOL ANALYSIS ===
File: {test_file.get()}
Condition Model: {selected_model.get()}
Analysis Time: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

🔧 CONDITION ANALYSIS:
Current Condition: {condition_result['condition'].upper()}
Confidence: {condition_result['confidence']:.2%}

Detailed Probabilities:
• Normal: {condition_result['probabilities']['normal']:.2%}
• Unbalance: {condition_result['probabilities']['unbalance']:.2%}
• Misalignment: {condition_result['probabilities']['misalignment']:.2%}
• Bearing Issue: {condition_result['probabilities']['bearing']:.2%}
"""

    if rul_result:
        results_content += f"""
⏱️ REMAINING USEFUL LIFE (RUL):
RUL Model: {selected_rul_model.get()}
Estimated Remaining Time: {rul_result['rul_days']:.1f} days ({rul_result['rul_hours']:.0f} hours)
Estimated Failure Date: {rul_result['estimated_failure_date']}
Confidence Interval: {rul_result['confidence_interval_days'][0]:.1f} - {rul_result['confidence_interval_days'][1]:.1f} days
Prediction Reliability: {rul_result['reliability_level']}
Wear index: {rul_result['wear_index']}

📋 MAINTENANCE RECOMMENDATIONS:
"""
        for rec in rul_result['recommendations']:
            results_content += f"• {rec}\n"
    else:
        results_content += f"""
⏱️ RUL PREDICTION: Not available (no RUL model loaded)

📋 CONDITION-BASED RECOMMENDATIONS:
"""
        for rec in condition_result['recommendations']:
            results_content += f"• {rec}\n"

    results_content += "\n" + "=" * 60 + "\n\n"

    results_text.insert("end", results_content)
    results_text.see("end")


def plot_signal():
    """Plot only 3x3 feature comparison plots"""
    if not viz_file.get():
        messagebox.showerror("Error", "Please select a feature file")
        return

    if not os.path.exists(viz_file.get()):
        messagebox.showerror("Error", "Feature file does not exist")
        return

    try:
        fig.clear()
        # Set white background for plots
        fig.patch.set_facecolor('white')

        # === LOAD FEATURES FROM FILE ===
        feature = tm.load_features(viz_file.get())
        print(feature)

        # Check if number of rows is divisible by 4 classes
        total_rows = feature.shape[0]
        if total_rows % 4 != 0:
            messagebox.showerror("Error", "Feature file does not have rows divisible by 4.")
            return

        segment_size = total_rows // 4  # e.g. 400/4 = 100

        # Plot feature charts only for selected axis
        tm.plot_axis_features_from_file(fig, viz_file_axis.get(), feature, segment_size)

        # Set white background for all subplots
        for ax in fig.get_axes():
            ax.set_facecolor('white')

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        canvas.draw()
        notebook.set("📊 Signal Visualization")

    except Exception as e:
        messagebox.showerror("Error", f"Error plotting features: {str(e)}")


def log_message(message):
    """Log message to results tab"""
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    results_text.insert("end", log_entry)
    results_text.see("end")


def clear_results():
    """Clear results text area"""
    results_text.delete("1.0", "end")


def main():
    """Main function to run the application"""
    create_main_window()
    root.mainloop()


if __name__ == "__main__":
    main()