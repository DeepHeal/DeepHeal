import argparse
import pandas as pd
import torch
import os
import sys
from deepheal.deepheal import DeepHeal

def main():
    parser = argparse.ArgumentParser(description="DeepHeal Training CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV (log2FC matrix).")
    parser.add_argument("--id-col", type=str, required=True, help="Column name for Sample ID.")
    parser.add_argument("--meta", type=str, default=None, help="Path to meta CSV (optional).")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension.")
    parser.add_argument("--output", type=str, default="embeddings.csv", help="Output path for embeddings CSV.")
    parser.add_argument("--no-batch", action="store_true", help="Disable batch correction (treat as single domain).")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")

    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading input data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    if args.id_col not in df.columns:
        print(f"Error: ID column '{args.id_col}' not found in input CSV.")
        sys.exit(1)

    # Separate IDs and Features
    sample_ids = df[args.id_col].astype(str).values
    input_features = df.drop(columns=[args.id_col])

    # 2. Load Meta (Optional)
    meta_df = None
    if args.meta:
        print(f"Loading meta data from {args.meta}...")
        try:
            meta_df = pd.read_csv(args.meta)
            # Ensure alignment? For now, we assume user aligns or we just use what matches?
            # DeepHeal.set_data expects aligned input and meta if meta is provided.
            # But simple usage might not need meta.
            # If meta is provided, we should probably align it to input_features.
            if args.id_col not in meta_df.columns:
                print(f"Error: ID column '{args.id_col}' not found in meta CSV.")
                sys.exit(1)

            # Align meta to input
            meta_df[args.id_col] = meta_df[args.id_col].astype(str)
            meta_df = meta_df.set_index(args.id_col).reindex(sample_ids).reset_index()

        except Exception as e:
            print(f"Error reading meta file: {e}")
            sys.exit(1)

    # 3. Initialize DeepHeal
    print("Initializing DeepHeal...")
    # Map CLI args to DeepHeal config
    config = {
        'latent_dim': args.latent_dim,
        'n_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'verbose': True
    }

    if args.no_batch:
        config['domain_key'] = None

    model = DeepHeal(save_dir=os.path.dirname(args.output), **config)

    model.set_data(input_features, meta_df)

    # 4. Train
    print("Starting training...")
    model.train(save_model=True)

    # 5. Generate Embeddings
    print("Generating embeddings...")
    embeddings = model.predict(input_features)

    # 6. Save Output
    print(f"Saving embeddings to {args.output}...")
    output_df = pd.DataFrame(embeddings, columns=[f"z{i+1}" for i in range(embeddings.shape[1])])
    output_df.insert(0, args.id_col, sample_ids)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
