import torch
import net.util as net_util
import net.loader as net_loader
import net.runner as runner
import data.factory as data_factory
import post.make_submission as sub


def stage_1_training(config, cfg, model, data_df, patient_df):
    """
    Stage 1 of training - train on fixed balanced dataset.
    """
    cfg = config.get_stage_1_conf(cfg)
    data_loader_train, data_loader_val = data_factory.get_fold_data_loaders(cfg, data_df, patient_df)
    data_loader_train.dataset.balance_data(cfg.seed)
    data_loader_val.dataset.balance_data(cfg.seed)
    runner.train(cfg, model, data_loader_train, data_loader_val)


def stage_2_training(config, cfg, model, data_df, patient_df):
    """
    Stage 2 of training - train on changing balanced dataset. Dataset is undersampled using different seed each epoch.
    """
    cfg = config.get_stage_2_conf(cfg)
    data_loader_train, data_loader_val = data_factory.get_fold_data_loaders(cfg, data_df, patient_df)
    data_loader_val.dataset.balance_data(cfg.seed)
    runner.train(cfg, model, data_loader_train, data_loader_val)


def stage_3_training(config, cfg, model, data_df, patient_df):
    """
    Stage 3 of training - train on full dataset.
    """
    cfg = config.get_stage_3_conf(cfg)
    data_loader_train, data_loader_val = data_factory.get_fold_data_loaders(cfg, data_df, patient_df)
    runner.train(cfg, model, data_loader_train, data_loader_val)


def train_model(config, cfg, data_df, patient_df):
    model = net_loader.get_model(cfg.n_classes, cfg.model.source, cfg.model.name)
    model.to(cfg.model.device)

    stage_1_training(config, cfg, model, data_df, patient_df)
    stage_2_training(config, cfg, model, data_df, patient_df)
    stage_3_training(config, cfg, model, data_df, patient_df)

    return model


def setup_and_train(config):
    cfg = config.get_default_conf()
    # Set seeds
    net_util.set_seed(cfg.seed)
    # Load csv to dataframes
    data_df, patient_df = data_factory.load_train_dataframes(cfg)
    train_model(config, cfg, data_df, patient_df)


def load_model(cfg, stage, epoch):
    saved_model = '{}_{}_{}_{}'.format(cfg.model.out_file_name, cfg.fold, stage, epoch)
    model_path = '{}model/{}.pth'.format(cfg.output_path, cfg.model.out_file_name, saved_model)
    print('Loading {}'.format(model_path))

    model = torch.load(model_path)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(cfg.model.device)

    return model


def main():
    config_module = __import__('conf', globals(), locals(),
                               ['model_001', 'model_002', 'model_003', 'model_004', 'model_005', 'model_006',
                                'model_007', 'model_008', 'model_009', 'model_010', 'model_011'], 0)
    submission_list = []

    print('Start model 001')
    config = config_module.model_001
    setup_and_train(config)
    cfg = config.get_default_conf()
    model = load_model(cfg, 3, 1)
    sub.predict_with_tta(cfg, model)
    submission_list.extend(['submission_{}_{}'.format(cfg.model.out_file_name, cfg.fold)])

    print('Start model 002')
    config = config_module.model_002
    setup_and_train(config)
    cfg = config.get_default_conf()
    model = load_model(cfg, 3, 1)
    sub.predict_with_tta(cfg, model)
    submission_list.extend(['submission_{}_{}'.format(cfg.model.out_file_name, cfg.fold)])

    print('Start model 005')
    config = config_module.model_005
    setup_and_train(config)
    cfg = config.get_default_conf()
    model = load_model(cfg, 3, 0)
    sub.predict_with_tta(cfg, model)
    submission_list.extend(['submission_{}_{}'.format(cfg.model.out_file_name, cfg.fold)])

    print('Start model 006')
    config = config_module.model_006
    setup_and_train(config)
    cfg = config.get_default_conf()
    model = load_model(cfg, 3, 4)
    sub.predict_with_tta(cfg, model)
    submission_list.extend(['submission_{}_{}'.format(cfg.model.out_file_name, cfg.fold)])

    print(submission_list)
    sub.ensemble(submission_list, output_name='comb')

if __name__ == '__main__':
    main()
