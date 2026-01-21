"""
Simple GUI to review and delete failed images.
"""
import sys
import os
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ReviewApp:
    def __init__(self, root, csv_path):
        self.root = root
        self.root.title(f"Review Failures - {Path(csv_path).name}")
        self.csv_path = Path(csv_path)
        
        # Load data
        try:
            self.df = pd.read_csv(self.csv_path)
            self.image_paths = self.df['path'].tolist()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")
            root.destroy()
            return
            
        self.current_idx = 0
        self.total = len(self.image_paths)
        self.deleted_count = 0
        
        if self.total == 0:
            messagebox.showinfo("Info", "No images to review.")
            root.destroy()
            return

        # UI Setup
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image Display
        self.image_label = tk.Label(self.main_frame, text="Loading...")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Info Label
        self.info_label = tk.Label(self.main_frame, text="", font=("Arial", 12))
        self.info_label.pack(pady=5)
        
        # Buttons Frame
        self.btn_frame = tk.Frame(self.main_frame)
        self.btn_frame.pack(fill=tk.X, pady=10)
        
        self.btn_delete = tk.Button(self.btn_frame, text="Delete File (D)", bg="#ffcccc", command=self.delete_current)
        self.btn_delete.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.btn_skip = tk.Button(self.btn_frame, text="Keep / Skip (Space)", bg="#ccffcc", command=self.next_image)
        self.btn_skip.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # Bind keys
        root.bind('<d>', lambda e: self.delete_current())
        root.bind('<space>', lambda e: self.next_image())
        root.bind('<Right>', lambda e: self.next_image())
        
        self.show_image()

    def show_image(self):
        if self.current_idx >= self.total:
            messagebox.showinfo("Done", f"Review complete.\nDeleted {self.deleted_count} files.")
            self.root.destroy()
            return
            
        path = self.image_paths[self.current_idx]
        self.info_label.config(text=f"Image {self.current_idx + 1}/{self.total}\n{path}")
        
        try:
            # Load and resize image for display
            img = Image.open(path)
            
            # Scale down if too big
            max_size = (800, 600)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo # Keep reference
            
        except Exception as e:
            self.image_label.config(image="", text=f"Error loading image:\n{e}")

    def delete_current(self):
        if self.current_idx < self.total:
            path = self.image_paths[self.current_idx]
            try:
                os.remove(path)
                print(f"Deleted: {path}")
                self.deleted_count += 1
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete file:\n{e}")
                
            self.next_image()

    def next_image(self):
        self.current_idx += 1
        self.show_image()

def main():
    root = tk.Tk()
    root.geometry("900x700")
    
    # Auto-find CSV files in current directory
    script_dir = Path(__file__).parent
    csv_files = list(script_dir.glob("*_failures.csv"))
    
    if not csv_files:
        messagebox.showinfo("No CSVs", "No failure CSV files found in star_dataset_utils.")
        return
        
    if len(csv_files) == 1:
        csv_path = csv_files[0]
    else:
        csv_path = filedialog.askopenfilename(
            initialdir=script_dir,
            title="Select Failure CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        
    if csv_path:
        app = ReviewApp(root, csv_path)
        root.mainloop()

if __name__ == "__main__":
    main()


