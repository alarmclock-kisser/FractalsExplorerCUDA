namespace FractalsExplorerCUDA
{
    partial class WindowMain
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

		#region Windows Form Designer generated code

		/// <summary>
		///  Required method for Designer support - do not modify
		///  the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.listBox_log = new ListBox();
			this.comboBox_devices = new ComboBox();
			this.comboBox_kernels = new ComboBox();
			this.button_exec = new Button();
			this.progressBar_vram = new ProgressBar();
			this.panel_kernelArgs = new Panel();
			this.checkBox_darkMode = new CheckBox();
			this.checkBox_silent = new CheckBox();
			this.pictureBox_view = new PictureBox();
			this.label_cached = new Label();
			this.label_meta = new Label();
			this.checkBox_record = new CheckBox();
			this.button_create = new Button();
			this.button_import = new Button();
			this.button_export = new Button();
			this.listBox_images = new ListBox();
			this.numericUpDown_fps = new NumericUpDown();
			this.checkBox_crosshair = new CheckBox();
			this.panel_view = new Panel();
			this.checkBox_invariables = new CheckBox();
			((System.ComponentModel.ISupportInitialize) this.pictureBox_view).BeginInit();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_fps).BeginInit();
			this.panel_view.SuspendLayout();
			this.SuspendLayout();
			// 
			// listBox_log
			// 
			this.listBox_log.FormattingEnabled = true;
			this.listBox_log.ItemHeight = 15;
			this.listBox_log.Location = new Point(12, 307);
			this.listBox_log.Name = "listBox_log";
			this.listBox_log.Size = new Size(472, 139);
			this.listBox_log.TabIndex = 0;
			// 
			// comboBox_devices
			// 
			this.comboBox_devices.FormattingEnabled = true;
			this.comboBox_devices.Location = new Point(12, 12);
			this.comboBox_devices.Name = "comboBox_devices";
			this.comboBox_devices.Size = new Size(200, 23);
			this.comboBox_devices.TabIndex = 1;
			this.comboBox_devices.Text = "Initialize CUDA-Device ...";
			// 
			// comboBox_kernels
			// 
			this.comboBox_kernels.FormattingEnabled = true;
			this.comboBox_kernels.Location = new Point(218, 12);
			this.comboBox_kernels.Name = "comboBox_kernels";
			this.comboBox_kernels.Size = new Size(200, 23);
			this.comboBox_kernels.TabIndex = 2;
			this.comboBox_kernels.Text = "Load CUDA-Kernel ...";
			// 
			// button_exec
			// 
			this.button_exec.Location = new Point(424, 11);
			this.button_exec.Name = "button_exec";
			this.button_exec.Size = new Size(60, 23);
			this.button_exec.TabIndex = 3;
			this.button_exec.Text = "EXEC";
			this.button_exec.UseVisualStyleBackColor = true;
			this.button_exec.Click += this.button_exec_Click;
			// 
			// progressBar_vram
			// 
			this.progressBar_vram.Location = new Point(12, 41);
			this.progressBar_vram.Name = "progressBar_vram";
			this.progressBar_vram.Size = new Size(200, 12);
			this.progressBar_vram.TabIndex = 4;
			// 
			// panel_kernelArgs
			// 
			this.panel_kernelArgs.BackColor = SystemColors.ControlLight;
			this.panel_kernelArgs.Location = new Point(218, 41);
			this.panel_kernelArgs.Name = "panel_kernelArgs";
			this.panel_kernelArgs.Size = new Size(266, 200);
			this.panel_kernelArgs.TabIndex = 5;
			// 
			// checkBox_darkMode
			// 
			this.checkBox_darkMode.AutoSize = true;
			this.checkBox_darkMode.Location = new Point(400, 452);
			this.checkBox_darkMode.Name = "checkBox_darkMode";
			this.checkBox_darkMode.Size = new Size(84, 19);
			this.checkBox_darkMode.TabIndex = 6;
			this.checkBox_darkMode.Text = "Dark mode";
			this.checkBox_darkMode.UseVisualStyleBackColor = true;
			this.checkBox_darkMode.CheckedChanged += this.checkBox_darkMode_CheckedChanged;
			// 
			// checkBox_silent
			// 
			this.checkBox_silent.AutoSize = true;
			this.checkBox_silent.Location = new Point(12, 452);
			this.checkBox_silent.Name = "checkBox_silent";
			this.checkBox_silent.Size = new Size(55, 19);
			this.checkBox_silent.TabIndex = 7;
			this.checkBox_silent.Text = "Silent";
			this.checkBox_silent.UseVisualStyleBackColor = true;
			// 
			// pictureBox_view
			// 
			this.pictureBox_view.Location = new Point(24, 21);
			this.pictureBox_view.Name = "pictureBox_view";
			this.pictureBox_view.Size = new Size(152, 136);
			this.pictureBox_view.TabIndex = 8;
			this.pictureBox_view.TabStop = false;
			// 
			// label_cached
			// 
			this.label_cached.AutoSize = true;
			this.label_cached.Location = new Point(12, 264);
			this.label_cached.Name = "label_cached";
			this.label_cached.Size = new Size(51, 15);
			this.label_cached.TabIndex = 9;
			this.label_cached.Text = "Cache: -";
			// 
			// label_meta
			// 
			this.label_meta.AutoSize = true;
			this.label_meta.Location = new Point(12, 244);
			this.label_meta.Name = "label_meta";
			this.label_meta.Size = new Size(101, 15);
			this.label_meta.TabIndex = 10;
			this.label_meta.Text = "No image loaded.";
			// 
			// checkBox_record
			// 
			this.checkBox_record.AutoSize = true;
			this.checkBox_record.Location = new Point(12, 282);
			this.checkBox_record.Name = "checkBox_record";
			this.checkBox_record.Size = new Size(63, 19);
			this.checkBox_record.TabIndex = 11;
			this.checkBox_record.Text = "Record";
			this.checkBox_record.UseVisualStyleBackColor = true;
			// 
			// button_create
			// 
			this.button_create.Location = new Point(358, 250);
			this.button_create.Name = "button_create";
			this.button_create.Size = new Size(60, 23);
			this.button_create.TabIndex = 12;
			this.button_create.Text = "Create";
			this.button_create.UseVisualStyleBackColor = true;
			this.button_create.Click += this.button_create_Click;
			// 
			// button_import
			// 
			this.button_import.Location = new Point(424, 250);
			this.button_import.Name = "button_import";
			this.button_import.Size = new Size(60, 23);
			this.button_import.TabIndex = 13;
			this.button_import.Text = "Import";
			this.button_import.UseVisualStyleBackColor = true;
			this.button_import.Click += this.button_import_Click;
			// 
			// button_export
			// 
			this.button_export.Location = new Point(424, 279);
			this.button_export.Name = "button_export";
			this.button_export.Size = new Size(60, 23);
			this.button_export.TabIndex = 14;
			this.button_export.Text = "Export";
			this.button_export.UseVisualStyleBackColor = true;
			this.button_export.Click += this.button_export_Click;
			// 
			// listBox_images
			// 
			this.listBox_images.FormattingEnabled = true;
			this.listBox_images.ItemHeight = 15;
			this.listBox_images.Location = new Point(218, 244);
			this.listBox_images.Name = "listBox_images";
			this.listBox_images.Size = new Size(134, 64);
			this.listBox_images.TabIndex = 15;
			// 
			// numericUpDown_fps
			// 
			this.numericUpDown_fps.Location = new Point(358, 278);
			this.numericUpDown_fps.Maximum = new decimal(new int[] { 144, 0, 0, 0 });
			this.numericUpDown_fps.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
			this.numericUpDown_fps.Name = "numericUpDown_fps";
			this.numericUpDown_fps.Size = new Size(60, 23);
			this.numericUpDown_fps.TabIndex = 16;
			this.numericUpDown_fps.Value = new decimal(new int[] { 10, 0, 0, 0 });
			// 
			// checkBox_crosshair
			// 
			this.checkBox_crosshair.AutoSize = true;
			this.checkBox_crosshair.Location = new Point(319, 452);
			this.checkBox_crosshair.Name = "checkBox_crosshair";
			this.checkBox_crosshair.Size = new Size(75, 19);
			this.checkBox_crosshair.TabIndex = 17;
			this.checkBox_crosshair.Text = "Crosshair";
			this.checkBox_crosshair.UseVisualStyleBackColor = true;
			this.checkBox_crosshair.CheckedChanged += this.checkBox_crosshair_CheckedChanged;
			// 
			// panel_view
			// 
			this.panel_view.Controls.Add(this.pictureBox_view);
			this.panel_view.Location = new Point(12, 59);
			this.panel_view.Name = "panel_view";
			this.panel_view.Size = new Size(200, 182);
			this.panel_view.TabIndex = 18;
			// 
			// checkBox_invariables
			// 
			this.checkBox_invariables.AutoSize = true;
			this.checkBox_invariables.Location = new Point(218, 452);
			this.checkBox_invariables.Name = "checkBox_invariables";
			this.checkBox_invariables.Size = new Size(114, 19);
			this.checkBox_invariables.TabIndex = 19;
			this.checkBox_invariables.Text = "Show invariables";
			this.checkBox_invariables.UseVisualStyleBackColor = true;
			// 
			// WindowMain
			// 
			this.AutoScaleDimensions = new SizeF(7F, 15F);
			this.AutoScaleMode = AutoScaleMode.Font;
			this.ClientSize = new Size(496, 473);
			this.Controls.Add(this.checkBox_invariables);
			this.Controls.Add(this.panel_view);
			this.Controls.Add(this.checkBox_crosshair);
			this.Controls.Add(this.numericUpDown_fps);
			this.Controls.Add(this.listBox_images);
			this.Controls.Add(this.button_export);
			this.Controls.Add(this.button_import);
			this.Controls.Add(this.button_create);
			this.Controls.Add(this.checkBox_record);
			this.Controls.Add(this.label_meta);
			this.Controls.Add(this.label_cached);
			this.Controls.Add(this.checkBox_silent);
			this.Controls.Add(this.checkBox_darkMode);
			this.Controls.Add(this.panel_kernelArgs);
			this.Controls.Add(this.progressBar_vram);
			this.Controls.Add(this.button_exec);
			this.Controls.Add(this.comboBox_kernels);
			this.Controls.Add(this.comboBox_devices);
			this.Controls.Add(this.listBox_log);
			this.Name = "WindowMain";
			this.Text = "Fractals Explorer using CUDA";
			((System.ComponentModel.ISupportInitialize) this.pictureBox_view).EndInit();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_fps).EndInit();
			this.panel_view.ResumeLayout(false);
			this.ResumeLayout(false);
			this.PerformLayout();
		}

		#endregion

		private ListBox listBox_log;
		private ComboBox comboBox_devices;
		private ComboBox comboBox_kernels;
		private Button button_exec;
		private ProgressBar progressBar_vram;
		private Panel panel_kernelArgs;
		private CheckBox checkBox_darkMode;
		private CheckBox checkBox_silent;
		private PictureBox pictureBox_view;
		private Label label_cached;
		private Label label_meta;
		private CheckBox checkBox_record;
		private Button button_create;
		private Button button_import;
		private Button button_export;
		private ListBox listBox_images;
		private NumericUpDown numericUpDown_fps;
		private CheckBox checkBox_crosshair;
		private Panel panel_view;
		private CheckBox checkBox_invariables;
	}
}
