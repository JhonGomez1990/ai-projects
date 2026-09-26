from pathlib import Path
import pandas as pd

def save_result(model_name: str, accuracy: float, output_path: Path) -> None:
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_result = pd.DataFrame(
        [
            {
                "model": model_name,
                "accuracy": accuracy,
                "accuracy_percentage": accuracy * 100,
            }
        ]
    )

    if output_path.exists():
        previous_results = pd.read_csv(output_path)

        results = pd.concat(
            [previous_results, new_result],
            ignore_index=True,
        )
    else:
        results = new_result

    results.to_csv(
        output_path,
        index=False,
    )