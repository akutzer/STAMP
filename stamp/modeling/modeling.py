import argparse
from pathlib import Path

from .marugoto.transformer.helpers import train_categorical_model_, deploy_categorical_model_, categorical_crossval_

def main():
    parser = argparse.ArgumentParser(
        description='Associative modeling with a Vision Transformer.')
    
    parser.add_argument("--clini_table", type=Path, help="Path to clini_excel file")
    parser.add_argument("--slide_table", type=Path, help="Path to slide_table file")
    parser.add_argument("--feature_dir", type=Path, help="Path to feature directory")
    parser.add_argument("--output_path", type=Path, help="Path to output file")
    parser.add_argument("--target_label", type=str, help="Target label")
    parser.add_argument("--cat_labels", type=str, nargs="+", default=[], help="Category labels")
    parser.add_argument("--cont_labels", type=str, nargs="+", default=[], help="Continuous labels")
    parser.add_argument("--categories", type=str, nargs="+", default=None, help="Categories")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--one_model", action="store_true", help="Run full training instead of cross-validation")
    #group.add_argument("--deploy_model", type=Path, help="Path to the model .pkl to deploy")
    parser.add_argument("--deploy_models", nargs="+", type=Path, help="Paths to model files")
    group.add_argument("--n_splits", type=int, default=5, help="Number of splits")
    
    args = parser.parse_args()

    if args.one_model:
        #run full training for 1 model
        train_categorical_model_(clini_table=args.clini_table, 
                                 slide_table=args.slide_table,
                                 feature_dir=args.feature_dir, 
                                 output_path=args.output_path,
                                 target_label=args.target_label, 
                                 cat_labels=args.cat_labels,
                                 cont_labels=args.cont_labels, 
                                 categories=args.categories)
    elif args.deploy_models:
        #deploy 1 model on data
        deploy_categorical_model_(clini_table=args.clini_table,
                                  slide_table=args.slide_table,
                                  feature_dir=args.feature_dir,
                                  model_paths=args.deploy_models,
                                  output_path=args.output_path,
                                  target_label=args.target_label,
                                  cat_labels=args.cat_labels,
                                  cont_labels=args.cont_labels)

    else:
        #run cross validation for n_splits models
        categorical_crossval_(clini_table=args.clini_table, 
                              slide_table=args.slide_table,
                              feature_dir=args.feature_dir,
                              output_path=args.output_path,
                              target_label=args.target_label,
                              cat_labels=args.cat_labels,
                              cont_labels=args.cont_labels,
                              categories=args.categories,
                              n_splits=args.n_splits)


if __name__ == "__main__":
    main()