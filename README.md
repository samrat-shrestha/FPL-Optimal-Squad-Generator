# FPL-Optimal-Squad-Generator

## Overview

The Fantasy Premier League Squad Optimizer is a Python application designed to help users create an optimal fantasy football squad for the Premier League. By leveraging data from the official Fantasy Premier League API, this tool analyzes player statistics and simulates performance to maximize expected points while adhering to budget and team constraints.

## Features

- **Data Retrieval**: Automatically fetches player data from the Fantasy Premier League API.
- **Data Processing**: Cleans and processes player information, including costs and expected points.
- **Player Selection**: Implements a strategy to select the best players for each position based on expected performance.
- **Optimization Algorithm**: Utilizes a custom optimization function to maximize expected points while minimizing risk, ensuring compliance with team and budget constraints.
- **Multiple Attempts**: Runs multiple optimization attempts to find the best possible squad configuration.
- **Results Display**: Outputs the optimized squad, total cost, expected points, and team distribution.
- **CSV Export**: Saves the optimized squad to a CSV file for easy access and sharing.

## Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `scipy`, `requests`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fpl-squad-optimizer.git
   cd fpl-squad-optimizer
   ```

2. Install the required libraries:
   ```bash
   pip install numpy pandas scipy requests
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## Usage

Simply run the application, and it will fetch the latest player data, perform the optimization, and display the results in the console. The optimized squad will also be saved to a CSV file named `best_squad.csv`.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
