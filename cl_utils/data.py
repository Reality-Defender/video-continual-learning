from avalanche.benchmarks.generators import paths_benchmark
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import os


def stable_diffusion_scenario(transform):
    sd_root_dir = '/home/ubuntu/data/self_created_stable_diffusion_images/1_fake'
    gan_root_dir = '/home/ubuntu/data/Dataset-GAN-eval-v1'
    sg2_root_dir = '/home/ubuntu/data/stylegan2/099000'

    sd_filenames_list = os.listdir(sd_root_dir)

    sd_experiences = []
    for name in sd_filenames_list:
        instance_tuple = (os.path.join(sd_root_dir, name), 1)  # all fakes here
        sd_experiences.append(instance_tuple)

    sd_train_experiences, sd_test_experiences = train_test_split(sd_experiences, test_size=0.3, random_state=21)

    gan_experiences = []
    for name in glob.glob(os.path.join(gan_root_dir, 'ffhq_images1024x1024*.png')):
        instance_tuple = (name, 0)
        gan_experiences.append(instance_tuple)
    for name in glob.glob(os.path.join(sg2_root_dir, '*.png')):
        instance_tuple = (name, 1)
        gan_experiences.append(instance_tuple)

    train_experiences = [sd_train_experiences]
    test_experiences = [sd_test_experiences, gan_experiences]

    scenario = paths_benchmark(train_lists_of_files=train_experiences,
                               test_lists_of_files=test_experiences,
                               task_labels=[0, 1],
                               complete_test_set_only=False,
                               train_transform=transform,
                               eval_transform=transform)

    # print('-' * 80)
    # print('TRAINING:')
    # for experience in scenario.train_stream:
    #     t = experience.task_label
    #     exp_id = experience.current_experience
    #     training_dataset = experience.dataloader
    #     print(f'Task {t} batch {exp_id}')
    #     print(f'This batch contains {len(training_dataset)} patterns')
    #
    # print('\nTESTING:')
    # for experience in scenario.test_stream:
    #     t = experience.task_label
    #     exp_id = experience.current_experience
    #     training_dataset = experience.dataloader
    #     print(f'Task {t} batch {exp_id}')
    #     print(f'This batch contains {len(training_dataset)} patterns')
    # print('-' * 80)

    return scenario


class GenericDataset(Dataset):
    def __init__(self, transform):
        super(GenericDataset, self).__init__()

        self.transform = transform
        self.file_list = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        path, label = self.file_list[i]
        img = np.asarray(Image.open(path))

        img = self.transform(image=img)['image']

        label = torch.tensor(label)

        return img, label


class GanDataset(GenericDataset):
    def __init__(self, transform, phase: str):
        super(GanDataset, self).__init__(transform=transform)

        gan_root_dir = '/home/ubuntu/data/Dataset-GAN-eval-v1'
        sg2_root_dir = '/home/ubuntu/data/stylegan2/099000'

        file_list = []
        for name in glob.glob(os.path.join(gan_root_dir, 'ffhq_images1024x1024*.png')):
            instance_tuple = (name, 0)
            file_list.append(instance_tuple)
        for name in glob.glob(os.path.join(sg2_root_dir, '*.png')):
            instance_tuple = (name, 1)
            file_list.append(instance_tuple)

        _, val_file_list = train_test_split(file_list, test_size=0.6, random_state=21)

        if phase == 'test':
            self.file_list = file_list

        if phase == 'val':
            self.file_list = val_file_list

class SDDataset(GenericDataset):
    def __init__(self, transform, phase: str):
        super(SDDataset, self).__init__(transform=transform)

        sd_root_dir = '/home/ubuntu/data/self_created_stable_diffusion_images/1_fake'

        file_list = []
        for name in os.listdir(sd_root_dir):
            instance_tuple = (os.path.join(sd_root_dir, name), 1)  # all fakes here
            file_list.append(instance_tuple)

        train_file_list, test_file_list = train_test_split(file_list, test_size=0.3, random_state=21)
        train_file_list, val_file_list = train_test_split(train_file_list, test_size=0.2, random_state=21)

        if phase == 'train':
            self.file_list = train_file_list
        elif phase == 'val':
            self.file_list = val_file_list
        elif phase == 'test':
            self.file_list = test_file_list
        else:
            raise ValueError('Phase must be in [`train`, `val`, `test`].')
