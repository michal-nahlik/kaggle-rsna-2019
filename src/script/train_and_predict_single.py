import conf.model_005 as config

import net.util as net_util
import net.loader as net_loader
import data.factory as data_factory
import net.runner as runner
import post.make_submission as make_submission


def stage_1_training(cfg, model, data_df, patient_df):
    """
    Stage 1 of training - train on fixed balanced dataset.
    """
    cfg = config.get_stage_1_conf(cfg)
    data_loader_train, data_loader_val = data_factory.get_fold_data_loaders(cfg, data_df, patient_df)
    data_loader_train.dataset.balance_data(cfg.seed)
    data_loader_val.dataset.balance_data(cfg.seed)
    runner.train(cfg, model, data_loader_train, data_loader_val)


def stage_2_training(cfg, model, data_df, patient_df):
    """
    Stage 2 of training - train on changing balanced dataset. Dataset is undersampled using different seed each epoch.
    """
    cfg = config.get_stage_2_conf(cfg)
    data_loader_train, data_loader_val = data_factory.get_fold_data_loaders(cfg, data_df, patient_df)
    data_loader_val.dataset.balance_data(cfg.seed)
    runner.train(cfg, model, data_loader_train, data_loader_val)


def stage_3_training(cfg, model, data_df, patient_df):
    """
    Stage 3 of training - train on full dataset.
    """
    cfg = config.get_stage_3_conf(cfg)
    data_loader_train, data_loader_val = data_factory.get_fold_data_loaders(cfg, data_df, patient_df)
    data_loader_val.dataset.balance_data(cfg.seed)
    runner.train(cfg, model, data_loader_train, data_loader_val)


def train_model(cfg, data_df, patient_df):
    model = net_loader.get_model(cfg.n_classes, cfg.model.source, cfg.model.name)
    model.to(cfg.model.device)

    stage_1_training(cfg, model, data_df, patient_df)
    stage_2_training(cfg, model, data_df, patient_df)
    stage_3_training(cfg, model, data_df, patient_df)

    return model


def main():
    for fold in [0, 2, 4]:
        cfg = config.get_default_conf()
        cfg.fold = fold
        # Set seeds
        net_util.set_seed(cfg.seed)
        # Load csv to dataframes
        data_df, patient_df = data_factory.load_train_dataframes(cfg)
        # Train model
        model = train_model(cfg, data_df, patient_df)
        make_submission.predict_with_tta(cfg, model)

if __name__ == '__main__':
    main()
