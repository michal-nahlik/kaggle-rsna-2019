import itertools
import numpy as np
import pandas as pd
import data.factory as data_factory
import data.util as data_util
import net.runner as runner

def predict_with_tta(cfg, model, suffix=''):
    """
    Predict on data in cfg.test_df_name using cfg.test.transform_list transformations and average the result.
    Save resulting predictions to submission file named based on cfg.model.out_file_name fold and suffix.
    """
    class_names = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

    test_df = data_util.load_csv(cfg.path + cfg.test.df_name)
    image_ids = test_df['Image'].values
    outputs = np.zeros((len(test_df), len(class_names)))

    for transform in cfg.test.transform_list:
        cfg.test.transform = transform
        loader = data_factory.get_data_loader(cfg, cfg.test, test_df)
        result = runner.run_nn(cfg.model, 'test', model, loader)
        outputs += result['outputs']

    outputs_names = ['_'.join(list(x)) for x in list(itertools.product(image_ids, class_names))]

    outputs /= len(cfg.test.transform_list)
    outputs = outputs.reshape((len(test_df) * len(class_names), 1))

    submission_path = '{}submission/submission_{}_{}{}.csv'.format(cfg.output_path, cfg.model.out_file_name, cfg.fold, suffix)
    submission = pd.DataFrame({'ID': outputs_names, 'Label': np.squeeze(outputs)})
    submission.to_csv(submission_path, index=False)
    print('Submission saved: {}'.format(submission_path))


def ensemble(submission_list, output_name=None, output_path='../output/'):
    """
    load and ensemble submissions saved in output_path/submissions using averaging and save the result to file if output name
    is given.
    :return dataframe with ensembled predictions and stack of predictions loaded from submissions
    """
    ensemble_df = None
    ensemble_stack = None
    for i, submission in enumerate(submission_list):
        sub_df = pd.read_csv('{}submission/{}.csv'.format(output_path, submission))

        if ensemble_df is None:
            ensemble_df = sub_df
            ensemble_stack = np.zeros((len(sub_df), len(submission_list)))
        else:
            ensemble_df['Label'] += sub_df['Label']

        ensemble_stack[:, i] = sub_df['Label']

    ensemble_df['Label'] = ensemble_df['Label'] / len(submission_list)

    if output_name is not None:
        ensemble_df.to_csv('{}submission/{}.csv'.format(output_path, output_name))

    return ensemble_df, ensemble_stack
