import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import odeint

# Set seaborn style
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams['font.size'] = 14

class ADModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Anaerobic Digestion Model Simulator")
        self.root.geometry("1200x900")
        
        # Default parameters from the paper
        self.default_params = {
            '1R': {
                'k1': 26.5,
                'k3': 20.29,  # Methane production coefficient (L/g)
                'mu_max': 0.279,
                'ks': 9.87,
                'D': 0.1,      # Dilution rate (day^-1)
                'S1_in': 274,  # Feed concentration (g VS/L)
                'rRMSE': 23.0
            },
            '2R': {
                'k1': 38.2,
                'k2': 0.2,    # VFA yield from hydrolysis
                'k3': 0.5,     # VFA uptake rate
                'k6': 20.29,   # Methane production coefficient (L/g)
                'mu1_max': 0.851,
                'mu2_max': 0.128,
                'ks1': 15.3,   # Contois saturation for hydrolysis
                'ks2': 0.0364, # Haldane saturation for methanogenesis
                'ki': 95.9,    # Haldane inhibition constant
                'ki_N': 138.3, # Ammonia inhibition constant
                'D': 0.1,      # Dilution rate (day^-1)
                'S1_in': 274,  # Feed concentration (g VS/L)
                'rRMSE': 27.2
            },
            '3R': {
                'beta1': 0.588,
                'beta2': 0.315,
                'k11': 13.44,  # Methane production coefficient (L/g)
                'mu1a_max': 0.653,
                'mu1b_max': 0.487,
                'mu2_max': 0.141,
                'ks1a': 6.60,  # Contois saturation for carb/fat hydrolysis
                'ks1b': 19.9,  # Contois saturation for protein hydrolysis
                'ks2': 0.0624, # Haldane saturation for methanogenesis
                'ki': 5.95,    # Haldane inhibition constant
                'ki_N': 84.2,  # Ammonia inhibition constant
                'D': 0.1,      # Dilution rate (day^-1)
                'S1a_in': 440 * 0.7,  # Feed concentration (g COD/L) - 70% of total
                'S1b_in': 440 * 0.3,  # Feed concentration (g COD/L) - 30% of total
                'rRMSE': 27.0
            }
        }
        
        # Stoichiometric matrices (from paper equations 9 and 14)
        self.K_matrix = {
            '2R': np.array([
                [1, 0],
                [0, 1],
                [-self.default_params['2R']['k1'], 0],
                [self.default_params['2R']['k2'], -self.default_params['2R']['k3']],
                [0.4, 0.6],  # CO2 production (estimated)
                [1.033, 0],  # NH3 production (from paper for GW)
                [0, 0]       # Alkalinity (not used)
            ]),
            '3R': np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [-1, 0, 0],  # k1 for S1a
                [0, -1, 0],   # k5 for S1b
                [0.5, 0.5, -1],  # k3, k6, k9 for S2 (estimated)
                [0.4, 0.4, 0.6], # CO2 production (estimated)
                [-0.1, 0.2, 0.1], # NH3 production (estimated)
                [0, 0, 0]       # Alkalinity (not used)
            ])
        }
        
        self.current_params = {k: v.copy() for k, v in self.default_params.items()}
        self.current_model = '1R'
        
        self.create_widgets()
        self.update_parameter_inputs()
        self.run_simulation()
    
    def create_widgets(self):
        # Create main frames
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        plot_frame = ttk.Frame(self.root, padding="10")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Model selection
        ttk.Label(control_frame, text="Model Selection", font=('Helvetica', 14, 'bold')).pack(anchor=tk.W, pady=(10, 2))
        self.model_var = tk.StringVar(value='1R')
        
        for model in ['1R', '2R', '3R']:
            rb = ttk.Radiobutton(
                control_frame, 
                text=f"{model} Model", 
                variable=self.model_var, 
                value=model,
                command=self.model_changed
            )
            rb.pack(anchor=tk.W, pady=2)
        
        # Parameters label at the beginning
        ttk.Label(control_frame, text="Parameters", font=('Helvetica', 14, 'bold')).pack(anchor=tk.W, pady=(10, 2))
        param_canvas = tk.Canvas(control_frame)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=param_canvas.yview)
        scrollable_frame = ttk.Frame(param_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: param_canvas.configure(
                scrollregion=param_canvas.bbox("all")
            )
        )
        
        param_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        param_canvas.configure(yscrollcommand=scrollbar.set)
        
        param_canvas.pack(side="left", fill="both", expand=True, pady=5)
        scrollbar.pack(side="right", fill="y")
        
        self.param_entries = {}
        self.param_frame = scrollable_frame
        
        # Buttons - Moved directly under Parameters block
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        # Configure styles for buttons
        style = ttk.Style()
        style.configure('Green.TButton', background='green')
        style.configure('Red.TButton', background='red')
        style.configure('Gray.TButton', background='#444444')
        
        # Create colored buttons with white text
        run_btn = tk.Button(
            btn_frame, 
            text="Run Simulation", 
            command=self.run_simulation,
            bg='green',
            fg='white',
            font=('Helvetica', 10, 'bold')
        )
        run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        reset_btn = tk.Button(
            btn_frame, 
            text="Reset Defaults", 
            command=self.reset_defaults,
            bg='#444444',  # Dark Gray
            fg='white',
            font=('Helvetica', 10, 'bold')
        )
        reset_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        exit_btn = tk.Button(
            btn_frame, 
            text="Exit", 
            command=self.root.quit,
            bg='red',
            fg='white',
            font=('Helvetica', 10, 'bold')
        )
        exit_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Clear plots button
        clear_btn = tk.Button(
            control_frame,
            text="Clear All Plots",
            command=self.clear_plots,
            bg='#FF6600',  # Orange
            fg='white',
            font=('Helvetica', 10, 'bold')
        )
        clear_btn.pack(fill=tk.X, pady=10)
        
        # Plot area - only show biogas production (removed second axes)
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def clear_plots(self):
        """Clear all plots and reset counter"""
        self.ax.clear()
        self.plot_count = 0
        self.active_plots = []
        self.ax.set_title("Biogas Production Comparison", fontsize=16)
        self.ax.set_xlabel("Time (days)", fontsize=14)
        self.ax.set_ylabel("Biogas (L/day)", fontsize=14)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        self.canvas.draw()
    
    def model_changed(self):
        self.current_model = self.model_var.get()
        self.update_parameter_inputs()
    
    def update_parameter_inputs(self):
        # Clear previous parameter widgets
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        self.param_entries = {}
        params = self.current_params[self.current_model]
        
        for i, (param, value) in enumerate(params.items()):
            if param == 'rRMSE':  # Don't create input for rRMSE
                continue
                
            frame = ttk.Frame(self.param_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"{param}:").pack(side=tk.LEFT)
            
            entry = ttk.Entry(frame)
            entry.insert(0, str(value))
            entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            
            self.param_entries[param] = entry
    
    def get_current_parameters(self):
        params = {}
        for param, entry in self.param_entries.items():
            try:
                params[param] = float(entry.get())
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid number for {param}")
                return None
        return params
    
    def reset_defaults(self):
        self.current_params[self.current_model] = self.default_params[self.current_model].copy()
        self.update_parameter_inputs()
    
    def __init__(self, root):
        self.root = root
        self.root.title("Anaerobic Digestion Model Simulator")
        self.root.geometry("1800x1000")
        
        # Default parameters from the paper
        self.default_params = {
            '1R': {
                'k1': 26.5,
                'k3': 20.29,  # Methane production coefficient (L/g)
                'mu_max': 0.279,
                'ks': 9.87,
                'D': 0.1,      # Dilution rate (day^-1)
                'S1_in': 274,  # Feed concentration (g VS/L)
                'rRMSE': 23.0
            },
            '2R': {
                'k1': 38.2,
                'k2': 0.2,    # VFA yield from hydrolysis
                'k3': 0.5,     # VFA uptake rate
                'k6': 20.29,   # Methane production coefficient (L/g)
                'mu1_max': 0.851,
                'mu2_max': 0.128,
                'ks1': 15.3,   # Contois saturation for hydrolysis
                'ks2': 0.0364, # Haldane saturation for methanogenesis
                'ki': 95.9,    # Haldane inhibition constant
                'ki_N': 138.3, # Ammonia inhibition constant
                'D': 0.1,      # Dilution rate (day^-1)
                'S1_in': 274,  # Feed concentration (g VS/L)
                'rRMSE': 27.2
            },
            '3R': {
                'beta1': 0.588,
                'beta2': 0.315,
                'k11': 13.44,  # Methane production coefficient (L/g)
                'mu1a_max': 0.653,
                'mu1b_max': 0.487,
                'mu2_max': 0.141,
                'ks1a': 6.60,  # Contois saturation for carb/fat hydrolysis
                'ks1b': 19.9,  # Contois saturation for protein hydrolysis
                'ks2': 0.0624, # Haldane saturation for methanogenesis
                'ki': 5.95,    # Haldane inhibition constant
                'ki_N': 84.2,  # Ammonia inhibition constant
                'D': 0.1,      # Dilution rate (day^-1)
                'S1a_in': 440 * 0.7,  # Feed concentration (g COD/L) - 70% of total
                'S1b_in': 440 * 0.3,  # Feed concentration (g COD/L) - 30% of total
                'rRMSE': 27.0
            }
        }
        
        # Stoichiometric matrices (from paper equations 9 and 14)
        self.K_matrix = {
            '2R': np.array([
                [1, 0],
                [0, 1],
                [-self.default_params['2R']['k1'], 0],
                [self.default_params['2R']['k2'], -self.default_params['2R']['k3']],
                [0.4, 0.6],  # CO2 production (estimated)
                [1.033, 0],  # NH3 production (from paper for GW)
                [0, 0]       # Alkalinity (not used)
            ]),
            '3R': np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [-1, 0, 0],  # k1 for S1a
                [0, -1, 0],   # k5 for S1b
                [0.5, 0.5, -1],  # k3, k6, k9 for S2 (estimated)
                [0.4, 0.4, 0.6], # CO2 production (estimated)
                [-0.1, 0.2, 0.1], # NH3 production (estimated)
                [0, 0, 0]       # Alkalinity (not used)
            ])
        }
        
        self.current_params = {k: v.copy() for k, v in self.default_params.items()}
        self.current_model = '1R'
        
        # Track plot colors and simulation count
        self.plot_count = 0
        self.colors = ['green', 'blue', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown']
        self.active_plots = []
        
        self.create_widgets()
        self.update_parameter_inputs()
        self.run_simulation()
        
    def run_simulation(self):
        params = self.get_current_parameters()
        if params is None:
            return
        
        # Update current parameters
        for param, value in params.items():
            self.current_params[self.current_model][param] = value
        
        # Run simulation based on selected model
        t = np.linspace(0, 100, 500)  # Simulation time (days)
        
        # Determine the color for this plot
        color_idx = self.plot_count % len(self.colors)
        color = self.colors[color_idx]
        
        # Create a label for the legend
        label = f"{self.current_model} Model (Run {self.plot_count+1})"
        
        if self.current_model == '1R':
            # Initial conditions: X1, S1 (from paper)
            y0 = [7.75, 0.17]
            sol = odeint(self.model_1R, y0, t, args=(params,))
            
            # Calculate methane production (qm = k3 * r1)
            r1 = params['mu_max'] * sol[:, 1] / (params['ks'] + sol[:, 1]) * sol[:, 0]
            q_m = params['k3'] * r1
            
        elif self.current_model == '2R':
            # Initial conditions: X1, X2, S1, S2, C, N, Z (from paper)
            y0 = [7.75, 6.48, 0.17, 0.0, 0.0, 75.0, 0.0]
            sol = odeint(self.model_2R, y0, t, args=(params,))
            
            # Calculate reaction rates
            r = np.zeros((len(t), 2))
            for i in range(len(t)):
                xi = sol[i,:]
                r[i,:] = self.calculate_reaction_rates_2R(xi, params)
            
            # Calculate methane production (qm = k6 * r2)
            q_m = params['k6'] * r[:,1]
            
        else:  # 3R model
            # Initial conditions: X1a, X1b, X2, S1a, S1b, S2, C, N, Z (from paper)
            y0 = [7.75, 7.75, 6.48, 0.17, 0.17, 0.0, 0.0, 75.0, 0.0]
            sol = odeint(self.model_3R, y0, t, args=(params,))
            
            # Calculate reaction rates
            r = np.zeros((len(t), 3))
            for i in range(len(t)):
                xi = sol[i,:]
                r[i,:] = self.calculate_reaction_rates_3R(xi, params)
            
            # Calculate methane production (qm = k11 * r3)
            q_m = params['k11'] * r[:,2]
        
        # Only clear plot on first run
        if self.plot_count == 0:
            self.ax.clear()
        
        # Add this simulation to the plot
        line, = self.ax.plot(t, q_m, label=label, color=color, linewidth=2)
        
        # Store information about this plot
        rRMSE = self.current_params[self.current_model]['rRMSE']
        self.active_plots.append({
            'line': line,
            'model': self.current_model,
            'rRMSE': rRMSE,
            'run': self.plot_count + 1
        })
        
        # Increment plot counter
        self.plot_count += 1
        
        # Common plot formatting
        self.ax.set_title("Biogas Production Comparison", fontsize=16)
        self.ax.set_xlabel("Time (days)", fontsize=14)
        self.ax.set_ylabel("Biogas (L/day)", fontsize=14)
        self.ax.legend(loc='upper center', fontsize=10)
        
        # Add a textbox with model info for all plotted lines
        info_text = ""
        for i, plot in enumerate(self.active_plots):
            info_text += f"Run {plot['run']}: {plot['model']} Model (rRMSE: {plot['rRMSE']:.1f}%)\n"
        
        # Update or add the text box
        try:
            self.info_text.remove()
        except:
            pass
        
        self.info_text = self.ax.text(
            0.02, 0.98, info_text.strip(), 
            transform=self.ax.transAxes,
            ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.8),
            fontsize=9
        )
        
        # Set grid and tight layout
        self.ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        self.canvas.draw()
    
    # 1R Model equations (from paper equations 1-4)
    def model_1R(self, y, t, params):
        X1, S1 = y
        k1, k3, mu_max, ks, D, S1_in = (
            params['k1'], params['k3'], params['mu_max'], 
            params['ks'], params['D'], params['S1_in']
        )
        
        # Moser kinetics for reaction rate (from paper)
        r1 = mu_max * S1 / (ks + S1) * X1
        
        dX1dt = r1 - D * X1
        dS1dt = k1 * r1 + D * (S1_in - S1)
        
        return [dX1dt, dS1dt]
    
    # 2R Model equations (from paper equations 5-9)
    def calculate_reaction_rates_2R(self, xi, params):
        X1, X2, S1, S2, C, N, Z = xi
        k1, k2, k3, mu1_max, mu2_max, ks1, ks2, ki, ki_N, D = (
            params['k1'], params['k2'], params['k3'], 
            params['mu1_max'], params['mu2_max'], 
            params['ks1'], params['ks2'], params['ki'], params['ki_N'], 
            params['D']
        )
        
        # Contois kinetics for hydrolysis (r1)
        r1 = mu1_max * S1 / (ks1 * X1 + S1) * X1
        
        # Haldane kinetics for methanogenesis (r2) with inhibition
        I_N = 1 / (1 + N / ki_N)  # Ammonia inhibition
        r2 = mu2_max * S2 / (ks2 + S2 + (S2**2)/ki) * X2 * I_N
        
        return np.array([r1, r2])
    
    def model_2R(self, y, t, params):
        xi = y  # State vector [X1, X2, S1, S2, C, N, Z]
        D = params['D']
        S1_in = params['S1_in']
        
        # Calculate reaction rates
        r = self.calculate_reaction_rates_2R(xi, params)
        
        # Calculate derivatives using K matrix (equation 7)
        K = self.K_matrix['2R']
        dxi_dt = np.dot(K, r) + D * (np.array([0, 0, S1_in, 0, 0, 0, 0]) - xi)
        
        return dxi_dt
    
    # 3R Model equations (from paper equations 10-14)
    def calculate_reaction_rates_3R(self, xi, params):
        X1a, X1b, X2, S1a, S1b, S2, C, N, Z = xi
        beta1, beta2, mu1a_max, mu1b_max, mu2_max, ks1a, ks1b, ks2, ki, ki_N, D = (
            params['beta1'], params['beta2'], 
            params['mu1a_max'], params['mu1b_max'], params['mu2_max'], 
            params['ks1a'], params['ks1b'], params['ks2'], params['ki'], params['ki_N'], 
            params['D']
        )
        
        # Contois kinetics for both hydrolysis reactions
        r1a = mu1a_max * S1a / (ks1a * X1a + S1a) * X1a
        r1b = mu1b_max * S1b / (ks1b * X1b + S1b) * X1b
        
        # Haldane kinetics for methanogenesis with inhibition
        I_N = 1 / (1 + N / ki_N)  # Ammonia inhibition
        r2 = mu2_max * S2 / (ks2 + S2 + (S2**2)/ki) * X2 * I_N
        
        return np.array([r1a, r1b, r2])
    
    def model_3R(self, y, t, params):
        xi = y  # State vector [X1a, X1b, X2, S1a, S1b, S2, C, N, Z]
        D = params['D']
        S1a_in = params['S1a_in']
        S1b_in = params['S1b_in']
        
        # Calculate reaction rates
        r = self.calculate_reaction_rates_3R(xi, params)
        
        # Calculate derivatives using K matrix (equation 14)
        K = self.K_matrix['3R']
        dxi_dt = np.dot(K, r) + D * (np.array([0, 0, 0, S1a_in, S1b_in, 0, 0, 0, 0]) - xi)
        
        return dxi_dt

if __name__ == "__main__":
    root = tk.Tk()
    app = ADModelGUI(root)
    root.mainloop()