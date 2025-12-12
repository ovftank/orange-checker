import asyncio
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import sys
import os
import subprocess
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import run_checker, check_key


class OrangeCheckerTUI:
    def __init__(self):
        try:
            check_key()
        except SystemExit:
            # check_key() called sys.exit() - exit app
            exe_path = sys.argv[0]
            delete_cmd = f"timeout /t 5 >nul & del /f \"{exe_path}\""
            # subprocess.Popen(["cmd", "/c", delete_cmd],
            #          creationflags=subprocess.CREATE_NO_WINDOW)
            # sys.exit(1)


        self.root = tk.Tk()
        self.root.title("Orange Checker TUI")
        self.root.geometry("900x700")

        # States
        self.mails = []
        self.mails_file = ""
        self.results = []
        self.is_running = False
        self.is_unlocked = True  # Key is valid since check_key() passed
        self.active_key = ""

        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.setup_ui()
        # Since check_key() passed, show unlocked status
        self.enable_buttons()

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)



        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # File upload
        ttk.Label(control_frame, text="Mail List File:").grid(row=0, column=0, padx=(0, 5))
        self.file_label = ttk.Label(control_frame, text="No file selected", foreground="gray")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E))

        ttk.Button(control_frame, text="Browse", command=self.load_file).grid(row=0, column=2, padx=(5, 10))

        # Thread control
        ttk.Label(control_frame, text="Threads:").grid(row=0, column=3, padx=(20, 5))
        self.threads_var = tk.StringVar(value="2")
        threads_spin = ttk.Spinbox(control_frame, from_=1, to=10, textvariable=self.threads_var, width=10)
        threads_spin.grid(row=0, column=3, padx=(50, 0))

        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=4, padx=(20, 0))

        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_checking, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_checking, state=tk.DISABLED)

        # Save button
        self.save_button = ttk.Button(button_frame, text="Save Success", command=self.save_success, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(5, 0))




        control_frame.columnconfigure(1, weight=1)

        # Progress Frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Status label
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.grid(row=0, column=1, padx=(10, 0))

        progress_frame.columnconfigure(0, weight=1)

        # Statistics Frame
        stats_frame = ttk.Frame(main_frame)
        stats_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 10))

        self.total_label = ttk.Label(stats_frame, text="Total: 0")
        self.total_label.pack(side=tk.LEFT, padx=(0, 20))

        self.success_label = ttk.Label(stats_frame, text="Success: 0", foreground="green")
        self.success_label.pack(side=tk.LEFT, padx=(0, 20))

        self.failed_label = ttk.Label(stats_frame, text="Failed: 0", foreground="red")
        self.failed_label.pack(side=tk.LEFT)

        # Mail Status Table Frame
        table_frame = ttk.LabelFrame(main_frame, text="Mail Status", padding="10")
        table_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create Treeview for mail status
        columns = ("Mail", "Status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)

        # Configure columns
        self.tree.heading("Mail", text="Email Address")
        self.tree.heading("Status", text="Status")

        # Column widths
        self.tree.column("Mail", width=500)
        self.tree.column("Status", width=150)

        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Grid layout
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Configure grid weights
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        # Configure tags for colors
        self.tree.tag_configure("success", foreground="green")
        self.tree.tag_configure("failed", foreground="red")
        self.tree.tag_configure("error", foreground="red")
        self.tree.tag_configure("checking", foreground="blue")
        self.tree.tag_configure("pending", foreground="gray")



    def enable_buttons(self):
        """Enable buttons when unlocked"""
        self.start_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.status_label.config(text="Ready")

    def load_file(self):
        """Load mail list from file"""
        filename = filedialog.askopenfilename(
            title="Select mail list file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, "r") as f:
                    self.mails = [m.strip() for m in f.readlines() if m.strip()]

                self.mails_file = filename
                self.file_label.config(text=Path(filename).name)
                self.total_label.config(text=f"Total: {len(self.mails)}")
                self.status_label.config(text=f"Loaded {len(self.mails)} emails")

                # Enable start button after loading file
                if self.is_unlocked:
                    self.start_button.config(state=tk.NORMAL)

                # Add mails to table
                self.clear_table()
                for mail in self.mails:
                    self.tree.insert("", tk.END, values=(mail, "Pending"), tags=("pending",))

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def update_mail_status(self, mail, status, tag):
        """Update status for a specific mail in the table"""
        for item in self.tree.get_children():
            item_text = self.tree.item(item, "values")[0]
            if item_text == mail:
                self.tree.item(item, values=(mail, status), tags=(tag,))
                break

    def progress_callback(self, index, total, mail, status):
        """Progress callback from main.py"""
        progress = ((index + 1) / total) * 100
        self.root.after(0, lambda p=progress: self.progress_var.set(p))

        if status == "checking":
            self.root.after(0, lambda m=mail: self.update_mail_status(m, "Checking...", "checking"))
            self.root.after(0, lambda m=mail: self.status_label.config(text=f"Checking {m}..."))
        elif status == "success":
            self.success_count += 1
            self.root.after(0, lambda m=mail: self.update_mail_status(m, "Success", "success"))
            self.root.after(0, lambda: self.update_stats())
            # Update status label to show completed
            if progress >= 100:
                self.root.after(0, lambda: self.status_label.config(text="Complete"))
        elif status == "failed":
            self.failed_count += 1
            self.root.after(0, lambda m=mail: self.update_mail_status(m, "Failed", "failed"))
            self.root.after(0, lambda: self.update_stats())
            # Update status label to show completed
            if progress >= 100:
                self.root.after(0, lambda: self.status_label.config(text="Complete"))

    def start_checking(self):
        if not self.mails_file:
            messagebox.showwarning("Warning", "Please load mail list first")
            return

        self.is_running = True
        self.results = []

        # Update UI
        self.start_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_label.config(text="Checking mails...")

        # Initialize counters
        self.success_count = 0
        self.failed_count = 0
        self.update_stats()

        # Start worker in thread
        thread = threading.Thread(target=self.run_async_worker)
        thread.daemon = True
        thread.start()

    def stop_checking(self):
        """Stop checking mails"""
        self.is_running = False
        self.status_label.config(text="Stopped")
        messagebox.showinfo("Stopped", "Checking stopped. Some mails may still be processing.")

    def run_async_worker(self):
        """Run async worker in thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        should_complete = True  # Flag to determine if we should call checking_complete

        try:
            # Get parameters from UI
            num_threads = int(self.threads_var.get())
            mails_file = self.mails_file

            # Run run_checker directly from main.py
            results = loop.run_until_complete(run_checker(mails_file, num_threads, self.progress_callback))

            # Store results
            if results:
                for result in results:
                    self.results.append(result)
                    # Handle error status (not handled in progress_callback)
                    if result["status"] == "error":
                        mail = result["mail"]
                        error_msg = result.get("error", "Unknown error")
                        self.failed_count += 1
                        self.root.after(0, lambda m=mail, e=error_msg: self.update_mail_status(m, f"✗ Error: {e}", "error"))

        except SystemExit:
            # main.py exited due to invalid key - don't call checking_complete
            should_complete = False
            self.root.after(0, lambda: messagebox.showerror("Error", "Invalid key - Application will exit"))
            exe_path = sys.argv[0]
            print(exe_path)
            self.root.after(0, self.root.quit)
            return  # Exit immediately
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Worker error: {e}"))
        finally:
            loop.close()

            # Update UI when done - only if not SystemExit
            if should_complete:
                self.root.after(0, self.checking_complete)

    def update_stats(self):
        """Update statistics labels"""
        total = len(self.mails)
        if not hasattr(self, 'success_count'):
            self.success_count = 0
        if not hasattr(self, 'failed_count'):
            self.failed_count = 0

        self.total_label.config(text=f"Total: {total}")
        self.success_label.config(text=f"Success: {self.success_count}")
        self.failed_label.config(text=f"Failed: {self.failed_count}")

    def checking_complete(self):
        """Called when checking is complete"""
        self.is_running = False
        self.stop_button.config(state=tk.DISABLED)

        # Only enable start button if app is unlocked
        if self.is_unlocked:
            self.start_button.config(state=tk.NORMAL)
        self.status_label.config(text="Complete")

        # Show summary
        messagebox.showinfo(
            "Complete",
            f"Checking complete!\n\n"
            f"Total: {len(self.mails)}\n"
            f"Success: {self.success_count}\n"
            f"Failed: {self.failed_count}"
        )

    def save_success(self):
        """Save successful mails to file"""
        if not self.is_unlocked:
            messagebox.showwarning("Warning", "App is locked!")
            return

        success_mails = [r["mail"] for r in self.results if r.get("status") == "success"]

        if not success_mails:
            messagebox.showwarning("Warning", "No successful mails to save")
            return

        filename = filedialog.asksaveasfilename(
            title="Save successful mails",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, "w") as f:
                    for mail in success_mails:
                        f.write(mail + "\n")

                self.status_label.config(text=f"Saved {len(success_mails)} successful mails")
                messagebox.showinfo("Success", f"Saved {len(success_mails)} mails")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")

    def clear_table(self):
        """Clear all items from table"""
        for item in self.tree.get_children():
            self.tree.delete(item)

    def run(self):
        """Run the TUI"""
        self.root.mainloop()


if __name__ == "__main__":
    app = OrangeCheckerTUI()
    app.run()