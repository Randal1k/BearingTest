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
from PIL import Image, ImageTk

import tool_monitor as tm
import RUL as rul  # Import the new RUL module
from config_loader import CONFIG, get_gui_params, get_config_value

# Set the appearance mode and default color theme
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# Global Vars
root = None
notebook = None
current_mode = None
model_type = None
selected_model = None
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
load_model_button = None
test_button = None
model_combo = None
progress_bar = None
progress_detail_label = None
status_label = None
results_text = None
fig = None
canvas = None

progress_value = 0.0

def create_main_window():
    """Create and configure the main window"""
    global root

    gui_params = get_gui_params()
    window_config = gui_params['window']

    root = ctk.CTk()
    root.title(window_config['title'])
    root.geometry(window_config['geometry'])
    root.minsize(window_config['min_size'][0], window_config['min_size'][1])

    # Set appearance from config
    ctk.set_appearance_mode(gui_params['appearance_mode'])
    ctk.set_default_color_theme(gui_params['color_theme'])

    initialize_variables()
    create_widgets()
    refresh_available_models()
    on_mode_change()

    root.bind("<<UpdateProgress>>", handle_progress_update)

def handle_progress_update(event=None):
    global progress_value
    if progress_bar:
        progress_bar.set(progress_value)

def initialize_variables():
    """Initialize all global variables"""
    global current_mode, model_type, selected_model, training_folder, viz_file_axis, test_file, viz_file, is_running, predict_function, rul_predict_function, custom_model_name
    global progress_frame, status_frame

    current_mode = ctk.StringVar(value="learning")
    model_type = ctk.StringVar(value="random_forest")
    viz_file_axis = ctk.StringVar(value="x")
    selected_model = ctk.StringVar()
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
    create_theory_tab()


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
    global learning_frame, testing_frame, train_button, load_model_button
    global test_button, model_combo, progress_bar, progress_detail_label, status_label
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

    learning_label = ctk.CTkLabel(learning_frame, text="Integrated Training Configuration",
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

    # Training button (single integrated training)
    button_frame = ctk.CTkFrame(learning_frame)
    button_frame.pack(pady=20)

    train_button = ctk.CTkButton(button_frame, text="🚀 Train Integrated Model", height=40,
                                 font=ctk.CTkFont(size=14, weight="bold"),
                                 command=start_integrated_training)
    train_button.pack()

    # Info label
    info_label = ctk.CTkLabel(button_frame,
                              text="This will train both condition and RUL models together",
                              text_color="gray", font=ctk.CTkFont(size=10))
    info_label.pack(pady=(5, 0))

    # Testing mode frame
    testing_frame = ctk.CTkFrame(scrollable_frame)
    testing_frame.pack(fill="x", padx=10, pady=10)

    testing_label = ctk.CTkLabel(testing_frame, text="Integrated Testing Configuration",
                                 font=ctk.CTkFont(size=16, weight="bold"))
    testing_label.pack(pady=(10, 15))

    # Model selection for integrated prediction
    model_select_frame = ctk.CTkFrame(testing_frame)
    model_select_frame.pack(fill="x", padx=20, pady=5)

    ctk.CTkLabel(model_select_frame, text="Integrated Model:",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10, pady=10)
    model_combo = ctk.CTkComboBox(model_select_frame, variable=selected_model, width=300)
    model_combo.pack(side="left", padx=10, pady=10)

    load_model_button = ctk.CTkButton(model_select_frame, text="📁 Load Integrated Model", width=150,
                                      command=load_integrated_model)
    load_model_button.pack(side="left", padx=5, pady=10)

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
                                command=run_integrated_test)
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
    """Refresh the list of available integrated models"""
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
            model_combo.configure(values=models)
            if models and not selected_model.get():
                selected_model.set(models[0])
                model_combo.set(models[0])

    except Exception as e:
        log_message(f"Error refreshing models: {str(e)}")

def create_theory_tab():
    """Create the theory tab for explanations and visuals"""
    theory_tab = notebook.add("📘 Theory")

    # Scrollable frame
    scrollable = ctk.CTkScrollableFrame(theory_tab, width=800, height=600)
    scrollable.pack(fill="both", expand=True, padx=20, pady=20)

    font_header = ctk.CTkFont(size=18, weight="bold")
    font_text = ctk.CTkFont(size=13)

    # Header
    ctk.CTkLabel(scrollable, text="🧠 Teoria programu", font=font_header).pack(anchor="w", pady=(10, 5))

    # Random Forest
    ctk.CTkLabel(scrollable, text="Random Forest", font=font_text).pack(anchor="w", pady=(10, 2))
    ctk.CTkLabel(scrollable,
        text="Random Forest to metoda uczenia zespołowego oparta na wielu drzewach decyzyjnych. Każde drzewo jest trenowane na losowym podzbiorze danych i losowym zestawie cech, \n"
             "co zapewnia różnorodność w modelu. Wynik końcowy jest uzyskiwany poprzez głosowanie większościowe (dla klasyfikacji) lub uśrednianie (dla regresji). Dzięki temu model\n"
             " jest odporny na przeuczenie i dobrze radzi sobie z danymi o wysokiej zmienności. Sprawdza się w zadaniach detekcji anomalii i oceny stanu technicznego urządzeń. \n"
             "Działa szybko i efektywnie nawet przy dużych zbiorach danych.",
        font=font_text, justify="left").pack(anchor="w")

    # SVM
    ctk.CTkLabel(scrollable, text="Support Vector Machine (SVM):", font=font_text).pack(anchor="w", pady=(10, 2))
    ctk.CTkLabel(scrollable,
        text="SVM to metoda uczenia maszynowego, która znajduje optymalną granicę decyzyjną (hiperpłaszczyznę), oddzielającą klasy danych z maksymalnym marginesem.\n"
             " Działa dobrze w przestrzeniach o wysokiej wymiarowości i potrafi modelować nieliniowe zależności dzięki funkcjom jądra (kernelom). SVM jest skuteczny\n"
             " przy niewielkiej liczbie próbek i dobrze sprawdza się w detekcji stanu anormalnego urządzeń. Model ten jest szczególnie użyteczny, gdy klasy są wyraźnie\n"
             " oddzielne. Działa również w wersji regresyjnej jako SVR (Support Vector Regression).",
        font=font_text, justify="left").pack(anchor="w")

    # Tool Condition Monitoring
    ctk.CTkLabel(scrollable, text="Proces przewidywania kondycji:", font=font_text).pack(anchor="w", pady=(10, 2))
    ctk.CTkLabel(scrollable,
        text="Proces przewidywania kondycji zaczyna się od zbierania danych pomiarowych z czujników (np. wibracje, temperatura, dźwięk). Następnie dane te są \n"
             "przetwarzane i opisywane za pomocą cech statystycznych (np. średnia, RMS). Kolejnym krokiem jest trening modelu predykcyjnego na danych historycznych, \n"
             "gdzie znany jest stan urządzenia. Po wytrenowaniu, model otrzymuje nowe dane i określa aktualną kondycję urządzenia (np. normalna, zużyta, awaryjna). \n"
             "Przewidywanie kondycji może być klasyfikacją (stan) lub regresją (wskaźnik zużycia). W niektórych przypadkach stosuje się również techniki prognozowania szeregów czasowych.",
        font=font_text, justify="left").pack(anchor="w")

    # RUL Prediction
    ctk.CTkLabel(scrollable, text="Przewidywanie RUL (Remaining Useful Life)", font=font_text).pack(anchor="w", pady=(10, 2))
    ctk.CTkLabel(scrollable,
        text="RUL to przewidywanie pozostałego czasu pracy urządzenia przed wystąpieniem awarii. Model uczony jest na danych historycznych urządzeń podobnego typu, \n"
             "gdzie znany jest czas awarii. Często wykorzystuje się metody regresji, sieci neuronowe lub modele sekwencyjne (np. LSTM). Model przetwarza dane cech \n"
             "z czujników i estymuje ile czasu (lub cykli) pozostało do końca życia technicznego. Predykcja RUL jest kluczowa w utrzymaniu predykcyjnym (predictive maintenance). \n"
             "Wynik pozwala zaplanować naprawy lub wymiany z wyprzedzeniem.",
        font=font_text, justify="left").pack(anchor="w")

    # Image Section (example placeholder)
    ctk.CTkLabel(scrollable, text="Diagram przetwarzania sygnału", font=font_text).pack(anchor="w", pady=(20, 5))
    try:
        img = Image.open("res/images/img.png")  # <- Wstaw swój obrazek tutaj
        img = img.resize((600, 350))
        img_tk = ImageTk.PhotoImage(img)
        label_img = tk.Label(scrollable, image=img_tk, bg="#2b2b2b")  # tkinter label for image
        label_img.image = img_tk
        label_img.pack(pady=10)
    except Exception as e:
        ctk.CTkLabel(scrollable, text=f"[Image not loaded: {e}]", font=font_text, text_color="gray").pack(anchor="w")

    # Footer
    ctk.CTkLabel(scrollable,
        text=" ",
        font=font_text, justify="left", text_color="gray").pack(anchor="w", pady=(20, 10))

    # === Feature explanations below canvas ===

    font_title = ctk.CTkFont(size=15, weight="bold")
    font_text = ctk.CTkFont(size=13)

    # List of features and corresponding image files
    features_info = [
        ("Mean (średnia):", "Określa przeciętną wartość sygnału. Wzrost może wskazywać na zmianę stanu pracy lub powolne zużycie.", "mean.png"),
        ("Standard Deviation (odchylenie standardowe):","Mierzy rozproszenie sygnału. Wysokie odchylenie sugeruje niestabilność lub wibracje.", "std.png"),
        ("RMS (Root Mean Square): ","Reprezentuje efektywną wartość sygnału. Jest czułe na wzrost amplitudy – przydatne do wykrywania zużycia mechanicznego.", "rms.png"),
        ("Peak to Peak:","Różnica między maksymalnym a minimalnym wychyleniem. Ujawnia skoki i gwałtowne zmiany, charakterystyczne dla defektów.", "p2p.png"),
        ("Impulse Factor:","Stosunek wartości maksymalnej do średniej z modułów sygnału. Wzrost może sugerować wystąpienie krótkich, ostrych impulsów (np. uszkodzenie łożyska).", "if.png"),
        ("Skewness (skośność):","Wskazuje asymetrię rozkładu danych. Zmiana może sugerować przesunięcie charakterystyki pracy urządzenia.", "skew.png"),
        ("Kurtosis (kurtoza): ","Informuje o „szczytowości” rozkładu. Wysoka kurtoza często wskazuje na impulsy (np. uderzenia lub pęknięcia).", "kurtosis.png"),
        ("Crest Factor: ","Stosunek wartości szczytowej do RMS. Używany do wykrywania anomalii – wysoka wartość wskazuje na obecność szczytów.", "crest.png"),
        ("Shape Factor: ","Stosunek RMS do wartości średniej. Służy do oceny kształtu sygnału – zmiany mogą wskazywać na nienaturalne zakłócenia w pracy maszyny.", "shape.png"),
    ]

    image_refs = []

    for name, desc, img_file in features_info:
        label = ctk.CTkLabel(scrollable, text=f"• {name}", font=font_title, anchor="w", justify="left")
        label.pack(anchor="w", padx=10, pady=(6, 0))

        desc_label = ctk.CTkLabel(scrollable, text=desc, font=font_text, text_color="white", wraplength=720,
                                  justify="left")
        desc_label.pack(anchor="w", padx=30)

        try:
            img_path = os.path.join("res/images/features", img_file)
            img = Image.open(img_path)
            img= img.resize((100,50))
            img_tk = ImageTk.PhotoImage(img)
            label_img = tk.Label(scrollable, image=img_tk, bg="#2b2b2b")
            label_img.image = img_tk
            label_img.pack(pady=10)
        except Exception as e:
            fallback = ctk.CTkLabel(scrollable, text=f"[Nie można załadować wzoru: {img_file}]",
                                    text_color="gray", font=font_text)
            fallback.pack(anchor="w", padx=30, pady=(0, 10))

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


def update_progress(current, total, elapsed_time=None):
    """Update progress bar and details"""

    def _update():
        if total > 0:
            progress_bar.set(current / total)

    # Schedule the GUI update in the main thread
    root.after(0, _update)


def start_integrated_training():
    """Start integrated training (both condition and RUL models) in a separate thread"""
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

    # Start integrated training in separate thread
    is_running = True
    train_button.configure(state='disabled')
    progress_bar.set(0)
    status_label.configure(text="Training integrated model (condition + RUL)...")

    thread = threading.Thread(target=integrated_training_worker)
    thread.daemon = True
    thread.start()


def integrated_training_worker():
    """Worker function for integrated training (runs in separate thread)"""
    import time
    global is_running

    try:
        start_time = time.time()
        update_status("Loading data...")
        progress_value = 0.05
        root.event_generate("<<UpdateProgress>>", when="tail")

        log_message("=== STARTING INTEGRATED TRAINING ===")
        log_message(f"Training folder: {training_folder.get()}")
        log_message(f"Model type: {model_type.get()}")

        # Ustaw nazwę pliku
        if custom_model_name.get().strip():
            custom_name = custom_model_name.get().strip()
            if custom_name.endswith('.pkl'):
                custom_name = custom_name[:-4]
            model_filename = f"{model_type.get()}/{custom_name}.pkl"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            model_filename = f"{model_type.get()}/tool_monitor_integrated_{timestamp}.pkl"

        feature_list, labels = tm.load_data_with_progress(
            training_folder.get(),
            lambda curr, total, elapsed: root.after(
                0, lambda: progress_bar.set(min(0.05 + (curr / total) * 0.25, 0.3))
            )
        )

        if not feature_list:
            log_message("❌ Error: No training data loaded!")
            return

        log_message(f"✅ Loaded {len(feature_list)} samples")

        # Trening condition modelu
        update_status("Training condition model...")
        progress_value = 0.35
        root.event_generate("<<UpdateProgress>>", when="tail")

        X_condition, y_condition, _ = tm.prepare_training_data(feature_list, labels)
        condition_model = tm.train_model(X_condition, y_condition, model_type.get())

        log_message("✅ Condition model training completed.")
        root.after(0, lambda: progress_bar.set(0.55))

        # Trening RUL modelu
        update_status("Training RUL model...")
        rul_targets = rul.generate_synthetic_rul_data(feature_list, labels)
        X_rul, y_rul, _ = rul.prepare_rul_training_data(feature_list, labels, rul_targets)

        def rul_progress_callback(curr, total):
            global progress_value
            ratio = curr / total
            progress_value = 0.6 + ratio * 0.35
            root.event_generate("<<UpdateProgress>>", when="tail")
            update_status(f"Training RUL... ({int(progress_value * 100)}%)")

        rul.train_rul_model(X_rul, y_rul, model_type.get(), rul_progress_callback)

        update_status("Saving model...")

        save_filepath = save_integrated_model(model_filename, feature_list)
        log_message("Integrated model training completed successfully!")
        log_message(f"Model saved as: {save_filepath}")

        progress_value = 1.0
        root.event_generate("<<UpdateProgress>>", when="tail")

        training_finished()
        elapsed = time.time() - start_time
        update_status(f"✅ Done in {elapsed:.1f}s")

    except Exception as e:
        update_status(f"❌ Error: {str(e)}")


def update_status(msg):
    root.after(0, lambda: status_label.configure(text=msg))

def save_integrated_model(filepath, feature_list):
    """Save integrated model containing both condition and RUL models"""
    if tm.model is None or rul.rul_model is None:
        raise ValueError("Both condition and RUL models must be trained before saving.")

    # Ensure proper directory structure
    if not filepath.startswith('res/model/'):
        if '/' in filepath:
            filepath = os.path.join('res/model', filepath)
        else:
            filepath = os.path.join('res/model', filepath)

    # Create directory
    model_dir = os.path.dirname(filepath)
    os.makedirs(model_dir, exist_ok=True)

    # Generate unique filename
    original_filepath = filepath
    counter = 1
    while os.path.exists(filepath):
        base_name = os.path.splitext(original_filepath)[0]
        extension = os.path.splitext(original_filepath)[1]
        filepath = f"{base_name}_{counter:03d}{extension}"
        counter += 1

    # Prepare integrated model data
    integrated_model_data = {
        'condition_model': tm.model,
        'condition_scaler': tm.scaler,
        'condition_feature_names': tm.feature_names,
        'condition_metadata': tm.training_metadata,
        'rul_model': rul.rul_model,
        'rul_scaler': rul.rul_scaler,
        'rul_feature_names': rul.rul_feature_names,
        'rul_metadata': rul.rul_training_metadata,
        'integration_timestamp': pd.Timestamp.now().isoformat(),
        'model_type': model_type.get()
    }

    # Save integrated model
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(integrated_model_data, f)

    log_message(f"Integrated model saved to {filepath}")

    # Save features
    tm.save_features(feature_list, filepath)

    return filepath


def load_integrated_model():
    """Load selected integrated model"""
    global predict_function, rul_predict_function

    if not selected_model.get():
        messagebox.showerror("Error", "Please select an integrated model")
        return

    try:
        status_label.configure(text="Loading integrated model...")

        model_path = os.path.join('res', 'model', selected_model.get())
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load integrated model
        import pickle
        with open(model_path, 'rb') as f:
            integrated_data = pickle.load(f)

        # Check if it's an integrated model
        if 'condition_model' in integrated_data and 'rul_model' in integrated_data:
            # Load condition model components
            tm.model = integrated_data['condition_model']
            tm.scaler = integrated_data['condition_scaler']
            tm.feature_names = integrated_data['condition_feature_names']
            tm.training_metadata = integrated_data['condition_metadata']

            # Load RUL model components
            rul.rul_model = integrated_data['rul_model']
            rul.rul_scaler = integrated_data['rul_scaler']
            rul.rul_feature_names = integrated_data['rul_feature_names']
            rul.rul_training_metadata = integrated_data['rul_metadata']

            # Set up prediction functions
            predict_function = tm.predict_condition
            rul_predict_function = rul.predict_rul

            log_message(f"Integrated model loaded: {selected_model.get()}")
            log_message(f"Condition model type: {integrated_data.get('model_type', 'Unknown')}")
            log_message(f"Integration date: {integrated_data.get('integration_timestamp', 'Unknown')}")

        else:
            # Fallback: try to load as regular condition model
            tm.model = integrated_data.get('model')
            tm.scaler = integrated_data.get('scaler')
            tm.feature_names = integrated_data.get('feature_names', [])
            tm.training_metadata = integrated_data.get('training_metadata', {})

            predict_function = tm.predict_condition
            rul_predict_function = None

            log_message(f"Regular condition model loaded: {selected_model.get()}")
            log_message("Note: No RUL prediction available with this model")

        status_label.configure(text="Integrated model loaded successfully")

    except Exception as e:
        log_message(f"Error loading integrated model: {str(e)}")
        status_label.configure(text="Error loading integrated model")


def run_integrated_test():
    """Run integrated analysis (condition + RUL)"""
    if not test_file.get():
        messagebox.showerror("Error", "Please select a test file")
        return

    if not os.path.exists(test_file.get()):
        messagebox.showerror("Error", "Test file does not exist")
        return

    if predict_function is None:
        messagebox.showerror("Error", "Please load an integrated model first")
        return

    try:
        status_label.configure(text="Running integrated analysis...")

        # Run condition prediction
        condition_result = predict_function(test_file.get())

        # Run RUL prediction if available
        rul_result = None
        if rul_predict_function is not None:
            rul_result = rul_predict_function(test_file.get(), condition_result)

        display_integrated_results(condition_result, rul_result)
        status_label.configure(text="Integrated analysis completed")

    except Exception as e:
        log_message(f"Integrated test error: {str(e)}")
        status_label.configure(text="Integrated test failed")


def display_integrated_results(condition_result, rul_result=None):
    """Display complete integrated test results"""
    notebook.set("📋 Results")  # Switch to results tab

    results_content = f"""
=== INTEGRATED TOOL ANALYSIS ===
File: {test_file.get()}
Model: {selected_model.get()}
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
Estimated Remaining Time: {rul_result['rul_days']:.1f} days ({rul_result['rul_hours']:.0f} hours)
Estimated Failure Date: {rul_result['estimated_failure_date']}
Confidence Interval: {rul_result['confidence_interval_days'][0]:.1f} - {rul_result['confidence_interval_days'][1]:.1f} days
Prediction Reliability: {rul_result['reliability_level']}
Wear Index: {rul_result['wear_index']:.4f}

📋 INTEGRATED MAINTENANCE RECOMMENDATIONS:
"""
        for rec in rul_result['recommendations']:
            results_content += f"• {rec}\n"
    else:
        results_content += f"""
⏱️ RUL PREDICTION: Not available with this model

📋 CONDITION-BASED RECOMMENDATIONS:
"""
        for rec in condition_result['recommendations']:
            results_content += f"• {rec}\n"

    results_content += "\n" + "=" * 60 + "\n\n"

    results_text.insert("end", results_content)
    results_text.see("end")


def training_finished():
    """Called when training is finished"""
    global is_running

    is_running = False
    train_button.configure(state='normal')
    progress_bar.set(1.0)
    status_label.configure(text="Integrated training completed")
    refresh_available_models()


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
        plot_config = get_config_value(CONFIG, 'gui.plots')
        fig.patch.set_facecolor(plot_config['background_color'])

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