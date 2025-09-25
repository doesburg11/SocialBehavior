# Heatmap Visualization for Altruism Grid Search
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load results
df = pd.read_csv('altruism/grid_search_results.csv')


def plot_heatmap_embedded(
                    df,
                    x,
                    y,
                    fixed,
                    value='coexist_prob',
                    aggfunc='mean',
                    cmap='viridis',
                    canvas=None,
                    fig=None
                ):
    dff = df.copy()
    for k, v in fixed.items():
        dff = dff[dff[k] == v]
    dff = dff.sort_values([y, x])
    pivot = dff.pivot_table(index=y, columns=x, values=value, aggfunc=aggfunc)
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig.clf()
        ax = fig.add_subplot(111)
    sns.heatmap(
        pivot,
        annot=False,
        fmt='.2f',
        cmap=cmap,
        cbar_kws={'label': 'Coexistence probability'},
        vmin=0,
        vmax=1,
        ax=ax
    )
    ax.set_title(f'Coexistence probability\nFixed: {fixed}')
    ax.set_xlabel(x.replace('_', ' '))
    ax.set_ylabel(y.replace('_', ' '))
    fig.tight_layout()
    if canvas is not None:
        canvas.draw()
    return fig


# Get all unique harshness values from the data
harshness_options = sorted(df['harshness'].unique())


def on_harshness_change():
    selected_harshness = float(harshness_var.get())
    fixed = {
        'disease': 0.26,
        'harshness': selected_harshness,
    }
    plot_heatmap_embedded(
        df,
        'benefit_from_altruism',
        'cost_of_altruism',
        fixed, value='coexist_prob',
        cmap='viridis',
        canvas=canvas,
        fig=fig
    )


root = tk.Tk()
root.title("Altruism Grid Search Heatmap")

frame = ttk.Frame(root, padding=10)
frame.pack(side=tk.LEFT, fill=tk.Y)

ttk.Label(
    frame, text="Select harshness:",
    font=("Helvetica", 18, "bold")
).pack(anchor=tk.W, pady=5)
harshness_var = tk.StringVar(value=str(harshness_options[0]))


radio_style = ttk.Style()
radio_style.configure(
    'Big.TRadiobutton',
    font=('Helvetica', 16),
    indicatorsize=20,
    padding=10
)

for h in harshness_options:
    ttk.Radiobutton(
        frame,
        text=f"{h}",
        variable=harshness_var,
        value=str(h),
        command=on_harshness_change,
        style='Big.TRadiobutton'
    ).pack(anchor=tk.W, pady=2)

# Matplotlib figure and canvas
fig = plt.Figure(figsize=(8, 6))
canvas_frame = ttk.Frame(root)
canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Initial plot
on_harshness_change()

root.mainloop()
