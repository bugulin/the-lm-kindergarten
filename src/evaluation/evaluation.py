from datasets import load_dataset

from ..inference import SyllogismSolver


def download_and_evaluate(solver: SyllogismSolver, dataset_path: str):
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    total = len(dataset)

    base_correct = 0
    tuned_correct = 0
    base_predictions = []
    tuned_predictions = []
    for i, input_item in enumerate(dataset):
        tuned_pred = next(solver.solve([input_item]))["validity"]
        with solver.model.disable_adapter():
            base_pred = next(solver.solve([input_item]))["validity"]

        base_predictions.append(base_pred)
        tuned_predictions.append(tuned_pred)

        if base_pred == input_item["validity"]:
            base_correct += 1
        if tuned_pred == input_item["validity"]:
            tuned_correct += 1

        print(
            f"Example {i}/{total}: gold={input_item["validity"]} | base={base_pred} | tuned={tuned_pred} | base_cor={base_correct} | tune_cor={tuned_correct}"
        )

    base_accuracy = base_correct / total
    tuned_accuracy = tuned_correct / total

    print("\n--- FINAL RESULTS ---")
    print(f"Number of test examples: {total}")
    print(f"Base model accuracy:  {base_accuracy:.4f}")
    print(f"Tuned model accuracy: {tuned_accuracy:.4f}")
    print(f"Improvement:          {tuned_accuracy - base_accuracy:+.4f}")
