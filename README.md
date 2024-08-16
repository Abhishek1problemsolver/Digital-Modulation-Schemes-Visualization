# Digital Modulation Schemes Visualization

## Overview

This project implements a visualization tool for digital modulation schemes using C++. The tool generates signals modulated using three common digital modulation techniques: Amplitude Shift Keying (ASK), Frequency Shift Keying (FSK), and Phase Shift Keying (PSK). The signals are then visualized using GNUplot.

## Features

- **Modulation Techniques Implemented:**
  - ASK (Amplitude Shift Keying)
  - FSK (Frequency Shift Keying)
  - PSK (Phase Shift Keying)
  
- **Visualization:**
  - Time-domain representation of the modulated signals.

## Dependencies

- **GNU Scientific Library (GSL):** Used for numerical computations.
- **GNUplot:** Used for plotting signals.

## Installation of Dependencies

### Ubuntu/Linux

```bash
sudo apt-get install libgsl-dev gnuplot
```

## Build and Run

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 2: Compile the C++ Code

Use `g++` to compile the source code with the necessary libraries.

```bash
g++ -o modulation_visualization modulation_visualization.cpp -lgsl -lgslcblas -lm
```

### Step 3: Run the Executable

```bash
./modulation_visualization
```

### Step 4: View the Results

The program will generate `.dat` files for each modulation scheme and corresponding `.png` files showing the time-domain signals:

- `ask_signal.dat` and `ASK_Modulation.png`
- `fsk_signal.dat` and `FSK_Modulation.png`
- `psk_signal.dat` and `PSK_Modulation.png`

These PNG files will contain visualizations of the modulated signals.

## Project Structure

```
modulation_visualization/
├── modulation_visualization.cpp   # Main C++ source file
├── ask_signal.dat                 # ASK modulated signal data (generated)
├── fsk_signal.dat                 # FSK modulated signal data (generated)
├── psk_signal.dat                 # PSK modulated signal data (generated)
├── ASK_Modulation.png             # ASK modulation plot (generated)
├── FSK_Modulation.png             # FSK modulation plot (generated)
└── PSK_Modulation.png             # PSK modulation plot (generated)
```

## Customization

You can adjust parameters such as `CARRIER_FREQ`, `BIT_RATE`, `SAMPLING_RATE`, and `DURATION` directly in the `modulation_visualization.cpp` file to explore different scenarios.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License.

## Contact

For any inquiries, please contact [Your Name] at [Your Email].
