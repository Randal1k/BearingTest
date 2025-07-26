import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from ttkthemes import ThemedTk
import customtkinter as ctk

import threading
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

import tool_monitor as tm

# Global Vars

root = None
notebook = None
current_mode = None
model_type = None
selected_model = None
training_folder = None
test_file = None
viz_file = None
custom_model_name = None
is_running = False
predict_function = None

current_file = 0
total_files = 0
start_time = None

learning_frame = None
testing_frame = None
train_button = None
load_model_button = None
test_button = None
model_combo = None
progress_bar = None
progress_detail_label = None
status_label = None
results_text = None
fig = None
canvas = None



def create_main_window():
    """Create and configure the main window"""
    global root
    root = tk.Tk()

    # ctk.set_appearance_mode("dark")
    # ctk.set_default_color_theme("blue")
    #
    # root = ctk.CTk()

    root.title("Tool Condition Monitor")
    root.geometry("800x600")
    root.minsize(600, 500)

    #Configure style
    style = ttk.Style()
    style.theme_use('default')
    print(style.theme_names())


    initialize_variables()
    create_widgets()
    refresh_available_models()
    on_mode_change()

def initialize_variables():
    """Initialize all global variables"""
    global current_mode, model_type, selected_model, training_folder, test_file, viz_file, is_running, predict_function, custom_model_name

    current_mode = tk.StringVar(value="learning")
    model_type = tk.StringVar(value="random_forest")
    selected_model = tk.StringVar()
    training_folder = tk.StringVar()
    test_file = tk.StringVar()
    viz_file = tk.StringVar()
    custom_model_name = tk.StringVar()
    global progress_frame, status_frame

def create_widgets():
    """Create all GUI widgets"""
    global notebook

    # Create main notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Create tabs
    create_guide_tab()
    create_main_tab()
    create_visualization_tab()
    create_results_tab()


def create_guide_tab():
    """Create the user guide tab"""
    # Guide tab
    guide_frame = ttk.Frame(notebook)
    notebook.add(guide_frame, text="📖 User Guide")

    # Create scrollable text widget for the guide
    guide_text = scrolledtext.ScrolledText(guide_frame, wrap=tk.WORD,
                                           height=25, width=80, font=('TkDefaultFont', 10))
    guide_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

    # Insert the comprehensive user guide
    with open("res/guide.md", "r", encoding="utf-8") as f:
        guide_content = f.read()

    guide_text.insert(1.0, guide_content)
    guide_text.config(state='disabled')  # Make it read-only


def create_main_tab():
    """Create the main tab with learning and testing modes"""
    global learning_frame, testing_frame, train_button, load_model_button
    global test_button, model_combo, progress_bar, progress_detail_label, status_label
    global progress_frame, status_frame

    # Main tab frame
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

    # Custom model name
    ttk.Label(learning_frame, text="Model Name (optional):").grid(row=2, column=0, sticky=tk.W, pady=2)
    custom_name_entry = ttk.Entry(learning_frame, textvariable=custom_model_name, width=40)
    custom_name_entry.grid(row=2, column=1, sticky=tk.W, padx=10, pady=2)
    ttk.Label(learning_frame, text="Leave empty for auto-naming", font=('TkDefaultFont', 8),
              foreground='gray').grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)

    # Start training button
    train_button = ttk.Button(learning_frame, text="Start Training",
                              command=start_training)
    train_button.grid(row=3, column=0, columnspan=3, pady=10)

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

    load_model_button = ttk.Button(button_frame, text="Load Model", command=load_model)
    load_model_button.pack(side=tk.LEFT, padx=5)

    test_button = ttk.Button(button_frame, text="Run Test", command=run_test)
    test_button.pack(side=tk.LEFT, padx=5)

    # Progress section - Fixed position to prevent jumping
    progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
    progress_frame.pack(fill=tk.X, padx=10, pady=5)
    progress_frame.pack_propagate(False)  # Prevent size changes
    progress_frame.config(height=100)  # Fixed height

    # Progress bar (determinate for file processing)
    progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
    progress_bar.pack(fill=tk.X, pady=(0, 5))

    # Progress details label (using monospace font for alignment)
    progress_detail_label = ttk.Label(progress_frame, text="Ready to start training...",
                                      font=('Courier', 9))
    progress_detail_label.pack(fill=tk.X)

    # Status label - Fixed position
    status_frame = ttk.Frame(main_frame)
    status_frame.pack(fill=tk.X, padx=10, pady=2)
    status_label = ttk.Label(status_frame, text="Ready")
    status_label.pack()


def create_visualization_tab():
    """Create the visualization tab for signal plotting"""
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
    global fig, canvas
    fig = Figure(figsize=(10, 8))
    canvas = FigureCanvasTkAgg(fig, viz_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)


def create_results_tab():
    """Create the results tab for displaying outputs"""
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
               command=lambda: results_text.delete(1.0, tk.END)).pack(pady=5)


def on_mode_change():
    """Handle mode change between learning and testing"""
    learning_frame.pack_forget()
    testing_frame.pack_forget()

    if current_mode.get() == "learning":
        learning_frame.pack(fill=tk.X, padx=10, pady=5)
    else:
        testing_frame.pack(fill=tk.X, padx=10, pady=5)

        # Zawsze pakuj te ramki na końcu (żeby nie przeskakiwały)
    progress_frame.pack(fill=tk.X, padx=10, pady=5)
    status_frame.pack(fill=tk.X, padx=10, pady=2)


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
        models = []
        model_dir = 'res/model'
        if os.path.exists(model_dir):
            for root_dir, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith('.pkl'):
                        relative_path = os.path.relpath(os.path.join(root_dir, file), model_dir)
                        models.append(relative_path)

        if model_combo:
            model_combo['values'] = models
            if models and not selected_model.get():
                selected_model.set(models[0])

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
    """
    Update progress bar and details

    DETAILED EXPLANATION OF PROGRESS BAR:

    1. THREAD SAFETY: This function can be called from worker threads, but GUI updates
       must happen in the main thread. We use root.after(0, _update) to schedule
       the actual GUI update in the main thread.

    2. PROGRESS CALCULATION:
       - percentage = (current / total) * 100  # Calculate completion percentage
       - progress_bar['value'] = percentage    # Update the visual progress bar

    3. VISUAL PROGRESS BAR:
       - Uses Unicode block characters (█) to create a visual bar
       - int(percentage/10) determines how many blocks to show (0-10 blocks)
       - Remaining spaces are filled with '  ' (two spaces per missing block)

    4. TIMING CALCULATIONS:
       - speed = current / elapsed_time        # Files processed per second
       - remaining_files = total - current     # How many files left
       - eta = remaining_files / speed         # Estimated time to completion

    5. FORMAT STRING:
       Processing files:  61%|██████    | 2444/4000 [02:48<01:58, 13.18it/s]

       Parts breakdown:
       - "Processing files:" - Static label
       - "61%" - Current percentage
       - "|██████    |" - Visual progress bar (6 blocks filled, 4 empty)
       - "2444/4000" - Current file / Total files
       - "[02:48<01:58" - [elapsed_time<estimated_remaining_time]
       - "13.18it/s]" - Processing speed (iterations/files per second)
    """

    def _update():
        if total > 0:
            # Calculate percentage and update progress bar
            percentage = (current / total) * 100
            progress_bar['value'] = percentage

            # Calculate timing and speed information
            if elapsed_time and elapsed_time > 0:
                speed = current / elapsed_time  # Files per second
                remaining_files = total - current
                eta = remaining_files / speed if speed > 0 else 0

                # Create visual progress bar using Unicode blocks
                filled_blocks = int(percentage / 10)  # 0-10 blocks
                empty_blocks = 10 - filled_blocks
                visual_bar = '█' * filled_blocks + '  ' * empty_blocks

                # Format the complete progress string (similar to tqdm)
                progress_text = (f"Processing files: {percentage:3.0f}%|{visual_bar}| "
                                 f"{current}/{total} [{format_time(elapsed_time)}<"
                                 f"{format_time(eta)}, {speed:.2f}it/s]")
            else:
                # Simplified version without timing
                filled_blocks = int(percentage / 10)
                empty_blocks = 10 - filled_blocks
                visual_bar = '█' * filled_blocks + '  ' * empty_blocks
                progress_text = f"Processing files: {percentage:3.0f}%|{visual_bar}| {current}/{total}"

            # Update the progress detail label
            progress_detail_label.config(text=progress_text)

    # Schedule the GUI update in the main thread (CRITICAL for thread safety)
    root.after(0, _update)

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
    progress_bar.config(mode='determinate')
    progress_bar['value'] = 0
    progress_detail_label.config(text="Initializing...")
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
        tm.save_model(model_filename, True)

        log_message("Training completed successfully!")
        log_message(f"Model saved as: {model_filename}")

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
    progress_bar['value'] = 100
    progress_detail_label.config(text="Training completed")
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
    """Plot only 3x3 feature comparison plots"""
    if not viz_file.get():
        messagebox.showerror("Error", "Please select a feature file")
        return

    if not os.path.exists(viz_file.get()):
        messagebox.showerror("Error", "Feature file does not exist")
        return

    try:
        fig.clear()

        # === WCZYTAJ CECHY Z PLIKU ===
        feature = tm.load_features(viz_file.get())
        print(feature)

        # Sprawdź czy liczba wierszy dzieli się na 4 klasy
        total_rows = feature.shape[0]
        if total_rows % 4 != 0:
            messagebox.showerror("Error", "Feature file does not have rows divisible by 4.")
            return

        segment_size = total_rows // 4  # np. 400/4 = 100

        # Rysuj wykresy cech tylko dla osi X
        tm.plot_axis_features_from_file(fig,'x', feature, segment_size)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        canvas.draw()
        notebook.select(2)

    except Exception as e:
        messagebox.showerror("Error", f"Error plotting features: {str(e)}")

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