from pathlib import Path

import generate_data
import generate_figures
import generate_report


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
REPORT_PATH = PROJECT_DIR / "reports" / "experiment_decision_report.md"
FIGURE_DIR = PROJECT_DIR / "figures"


def main():
    print("generating data")
    generate_data.main()

    print("generating figures")
    generate_figures.main()

    print("generating report")
    generate_report.main()

    print("done")
    print(f"Data directory: {DATA_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")
    print(f"Decision report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
