import copy as cp
import io
import json
import math
import multiprocessing as mp
import os
import os.path as osp
import pickle
import random as rd
from collections import defaultdict
from datetime import datetime
from typing import List, Optional

try:
    import cv2
except ImportError:
    import traceback

    traceback.print_exc()
    raise ImportError(
        'Import cv2 failed. Please install it with '
        '"pip install opencv-python-headless" and try again.\n\n'
        'If the prompt `ImportError: libGL.so.1` appears,'
        ' you may consider one of the following two methods:\n'
        'Method 1 - Uninstall opencv and then install opencv-headless\n'
        'pip uninstall opencv-python; pip install opencv-python-headless\n\n'
        'Method 2: Install the missing dynamic link libraries\n'
        'sudo apt-get update; sudo apt-get install -y libgl1 libglib2.0-0')
import mmengine
import numpy as np
import pandas as pd
from mmengine import ConfigDict
from tabulate import tabulate
from tqdm import tqdm

from opencompass.utils import build_dataset_from_cfg, dataset_abbr_from_cfg



def draw_heatmap(hmap, title):
    """Draw a heatmap using the given data.

    Args:
        hmap (pd.DataFrame): The data for the heatmap.
        title (str): The title for the heatmap.

    Returns:
        np.ndarray: An image of the heatmap.
    """
    from matplotlib import font_manager
    if FONT_FILE is None:
        fontP = font_manager.FontProperties()
    else:
        fontP = font_manager.FontProperties(fname=FONT_FILE)
    fontP.set_size(18)
    import matplotlib.pyplot as plt
    import seaborn as sns
    ax = sns.heatmap(hmap,
                     annot=True,
                     cmap='Blues',
                     annot_kws={'size': 35 / np.sqrt(len(hmap))})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
    plt.yticks(rotation=0)
    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top')
    plt.title(title, color='Blue', fontproperties=fontP)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    plt.close()
    buffer.seek(0)
    image_data = buffer.getvalue()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    return image


class SubjectiveSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
        vispair (List[str], optional): List of
            two models to visualize.
        refm (str, optional): Reference model
            for win rate comparison.
        col_name (str): Name of the column
            containing evaluation results.
        fout (str): Output file name.
        ignore (str, optional): Ignore certain
            comparisons based on a file.
    """

    def __init__(
        self,
        config: ConfigDict,
        vispair: Optional[List[str]] = None,
        refm: Optional[str] = None,
        col_name: str = 'gpt4',
        fout: str = 'report.md',
        ignore: Optional[str] = None,
    ) -> None:
        self.tasks = []
        self.cfg = config
        self.vispair = vispair
        self.refm = refm
        self.col_name = col_name
        self.fout = fout
        self.ignore = ignore

    def summarize(self,
                  time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """

        dataset_cfgs = self.cfg['datasets']
        eval_cfg = self.cfg['eval']
        work_dir = self.cfg['work_dir']
        self.work_dir = work_dir

        self.time_str = time_str
        output_path = osp.join(self.work_dir, 'summary',
                               f'summary_{self.time_str}.txt')
        output_dir = osp.join(osp.split(output_path)[0], f'{self.time_str}')
        mmengine.mkdir_or_exist(output_dir)
        fout = open(osp.join(output_dir, self.fout), 'w')
        results_folder = osp.join(work_dir, 'results')
        data_list = []
        for subdir in os.listdir(results_folder):
            subdir_path = os.path.join(results_folder, subdir)
            if os.path.isdir(subdir_path):
                model1, model2 = subdir.split('_')
                for dataset in dataset_cfgs:
                    origin_dataset = build_dataset_from_cfg(dataset)
                    dataset_abbr = dataset_abbr_from_cfg(dataset)
                    filepath = os.path.join(subdir_path,
                                            dataset_abbr + '.json')
                    result = mmengine.load(filepath)
                    if eval_cfg['partitioner']['mode'] == 'all':
                        for key, value in result.items():
                            prediction = value.get('prediction', None)
                            q_index = origin_dataset.test[int(key) % len(
                                origin_dataset.test)]['index']
                            cmp_index = f'{q_index};{model1};{model2}'
                            data_list.append(
                                [cmp_index, model1, model2, prediction])

        data = pd.DataFrame(data_list, columns=['cmp_index', 'A', 'B', 'gpt4'])
        meta = pd.read_excel(
            osp.join(dataset_cfgs[0]['path'],
                     dataset_cfgs[0]['name'] + '.xlsx'))

        if self.ignore is not None:
            q_index = [x.split(';')[0] for x in data['cmp_index']]
            to_ignore = set(mrlines(self.ignore))
            flag = [x not in to_ignore for x in q_index]
            data = data[flag]

        double_log('# Subjective Analysis', fout)
        capas = proc_capa(meta['capability'])
        capa_map = {i: c for i, c in zip(meta['index'], meta['capability'])}

        nonem = [x != 'EM' for x in data[self.col_name]]
        double_log(
            f'A total of {len(data)} comparisons, of which {sum(nonem)} '
            f'comparisons are meaningful (A / B answers inconsistent)', fout)
        data = data[nonem]

        data['capability'] = [
            capa_map[str(i).split(';')[0]] for i in data['cmp_index']
        ]
        data['extracted'] = [match_answer(ans) for ans in data[self.col_name]]

        succeed = [not pd.isna(x) for x in data['extracted']]
        succeed_rate = np.mean(succeed)
        double_log(
            f'A total of {len(succeed)} answer comparisons, successfully '
            f'extracted {sum(succeed)} answers from GPT-4 replies, with '
            f'an extraction success rate of {succeed_rate * 100:.2f}%', fout)
        data = data[succeed]

        cons, incons = find_inconsistent(data, 'ABCD')
        if len(cons) != len(data):
            double_log(
                f'A total of {len(data)} answer comparisons, {len(cons)} '
                f'pairs (A vs. B <-> B vs. A) are consistent，consistent '
                f'rate is {len(cons) / len(data) * 100:.2f}%', fout)

        dump(cons, osp.join(output_dir, 'consistent_cmp.xlsx'))
        dump(incons, osp.join(output_dir, 'inconsistent_cmp.xlsx'))

        data = cons
        if self.vispair is not None and len(self.vispair) == 2:
            extract_vispair(data, vispair=self.vispair)

        data['lang'] = [x.split('-')[0] for x in data['cmp_index']]
        langs = [None, 'cn', 'en']
        return self.analyze(data, self.refm, langs, capas, fout)