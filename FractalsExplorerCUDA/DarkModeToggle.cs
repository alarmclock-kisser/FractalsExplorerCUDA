using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using System.Linq;
using System.Runtime.CompilerServices;

namespace FractalsExplorerCUDA
{
	public static class DarkModeToggle
	{
		// ----- ----- ----- PALETTES ----- ----- ----- \\
		private static readonly Dictionary<Type, Color> _darkPalette = new()
		{
			{ typeof(Form), Color.FromArgb(32, 32, 32) },
			{ typeof(Panel), Color.FromArgb(40, 40, 40) },
			{ typeof(Button), Color.FromArgb(60, 60, 60) },
			{ typeof(TextBox), Color.FromArgb(50, 50, 50) },
			{ typeof(Label), Color.Transparent },
			{ typeof(ListBox), Color.FromArgb(45, 45, 45) },
			{ typeof(ComboBox), Color.FromArgb(45, 45, 45) },
			{ typeof(CheckBox), Color.Transparent },
			{ typeof(RadioButton), Color.Transparent },
			{ typeof(GroupBox), Color.FromArgb(40, 40, 40) },
			{ typeof(TabControl), Color.FromArgb(40, 40, 40) },
			{ typeof(TabPage), Color.FromArgb(40, 40, 40) }
		};

		private static readonly Dictionary<Type, Color> _lightPalette = new()
		{
			{ typeof(Form), SystemColors.Control },
			{ typeof(Panel), SystemColors.ControlLight },
			{ typeof(Button), SystemColors.ControlLightLight },
			{ typeof(TextBox), Color.White },
			{ typeof(Label), Color.Transparent },
			{ typeof(ListBox), Color.White },
			{ typeof(ComboBox), Color.White },
			{ typeof(CheckBox), Color.Transparent },
			{ typeof(RadioButton), Color.Transparent },
			{ typeof(GroupBox), SystemColors.ControlLight },
			{ typeof(TabControl), SystemColors.ControlLight },
			{ typeof(TabPage), SystemColors.ControlLight }
		};



		// ----- ----- ----- STATE ----- ----- ----- \\
		public static bool IsDarkMode { get; private set; }
		public static Form? MainForm { get; private set; } = null;



		// ----- ----- ----- METHODS ----- ----- ----- \\
		public static void ToggleDarkMode(Form? window = null, bool? forceDarkMode = null)
		{
			// Verify window
			window ??= DarkModeToggle.MainForm ?? Application.OpenForms[0];
			if (window == null)
			{
				return;
			}

			// Determine new mode
			DarkModeToggle.IsDarkMode = forceDarkMode ?? window.BackColor.GetBrightness() < 0.5f;

			// Apply to main form
			DarkModeToggle.ApplyColors(window, DarkModeToggle.IsDarkMode);

			// Recursively apply to all child controls
			DarkModeToggle.ApplyToControlCollection(window.Controls, DarkModeToggle.IsDarkMode);
		}

		private static void ApplyToControlCollection(Control.ControlCollection controls, bool darkMode)
		{
			foreach (Control control in controls)
			{
				DarkModeToggle.ApplyColors(control, darkMode);

				// Recursive call for container controls
				if (control.HasChildren)
				{
					DarkModeToggle.ApplyToControlCollection(control.Controls, darkMode);
				}
			}
		}

		private static void ApplyColors(Control control, bool darkMode)
		{
			Type controlType = control.GetType();

			// Find the most specific type match (including base types)
			Dictionary<Type, Color> palette = darkMode ? DarkModeToggle._darkPalette : DarkModeToggle._lightPalette;
			Type colorKey = palette.Keys.FirstOrDefault(t => t.IsAssignableFrom(controlType)) ?? controlType;

			if (palette.TryGetValue(colorKey, out Color backColor))
			{
				// button_darkMode Text to light / dark mode
				if (control is Button button)
				{
					if (button.Name == "button_darkMode")
					{
						button.Text = darkMode ? "Light mode" : "Dark mode";
					}

					if (button.Name == "button_createColor")
					{
						return;
					}
				}

				// Special handling for transparent controls
				if (backColor != Color.Transparent)
				{
					control.BackColor = backColor;
				}

				// Set foreground color based on background brightness
				if (control.ForeColor != Color.Red && control.ForeColor != Color.DarkOrange)
				{
					control.ForeColor = darkMode ? Color.White : Color.Black;
				}

				// Special cases
				if (control is TextBoxBase textBox)
				{
					textBox.BorderStyle = darkMode ? BorderStyle.FixedSingle : BorderStyle.Fixed3D;
				}
			}
		}



		// ----- ----- ----- EXTRAS ----- ----- ----- \\
		public static List<T> FindControls<T>(Form? window = null, string searchName = "", bool ignoreCase = false, bool searchChildren = false) where T : Control
		{
			// Verify window
			window ??= DarkModeToggle.MainForm ?? Application.OpenForms[0];
			if (window == null)
			{
				return [];
			}

			// Get all controls of type T
			List<T> controls = window.Controls.OfType<T>().ToList();

			// If searchChildren is true, search recursively
			if (searchChildren)
			{
				foreach (Control control in window.Controls)
				{
					if (control.HasChildren)
					{
						controls.AddRange(control.Controls.OfType<T>());
					}
				}
			}

			// Find controls by name (optionally ignoring case)
			if (ignoreCase)
			{
				controls = controls.Where(c => c.Name.Contains(searchName, StringComparison.OrdinalIgnoreCase)).ToList();
			}
			else
			{
				controls = controls.Where(c => c.Name.Contains(searchName)).ToList();
			}

			// Return the list of controls
			return controls;
		}

		public static string? CopyLogToClipboard(ListBox? listBox_log = null, bool showMsgbox = false)
		{
			// Verify listBox_log
			listBox_log ??= DarkModeToggle.FindControls<ListBox>(DarkModeToggle.MainForm, "listBox_log", true, true).FirstOrDefault();
			if (listBox_log == null)
			{
				return null;
			}

			// Get selected items (all if none selected)
			string copyString = "[LOG] " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + Environment.NewLine + Environment.NewLine;
			if (listBox_log.SelectedItems.Count == 0)
			{
				copyString += string.Join(Environment.NewLine, listBox_log.Items.Cast<string>());
			}
			else
			{
				copyString += string.Join(Environment.NewLine, listBox_log.SelectedItems.Cast<string>());
			}

			// Copy to clipboard
			Clipboard.SetText(copyString);

			// Optionally show message box
			if (showMsgbox)
			{
				MessageBox.Show("Log copied to clipboard: \n\n" + copyString, "Info", MessageBoxButtons.OK, MessageBoxIcon.Information);
			}

			// Return the copied string
			return copyString;
		}

	}
}