import os
import datetime
from shutil import copyfile
from collections import namedtuple, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import random
import consts
import torch_geometric.transforms as T
from torch_geometric.data import Data
from pytorch3d.io import load_obj, save_obj

def save_image_normalized(*args, **kwargs):
    save_image(*args, **kwargs, normalize=True, range=(-1, 1), padding=4)

def two_sided(x):
    return 2 * (x - 0.5)

def one_sided(x):
    return (x + 1) / 2

pil_to_model_tensor_transform = transforms.Compose(
    [
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.mul(2).sub(1))  # Tensor elements domain: [0:1] -> [-1:1]
    ]
)

def get_utkface_dataset(root):
    print(root)
    ret = lambda: ImageFolder(os.path.join(root, 'labeled'), transform=pil_to_model_tensor_transform)
    try:
        return ret()
    except (RuntimeError, FileNotFoundError):
        sort_to_classes(os.path.join(root, 'unlabeled'), print_cycle=1000)
        return ret()

def get_timeinterval(source_mesh_name, target_mesh_name):
    source_yearmon = source_mesh_name.split('_')[1]
    target_yearmon = target_mesh_name.split('_')[1]
    year_interval = (int(target_yearmon[0:4]) - int(source_yearmon[0:4]))*12
    month_interval = int(target_yearmon[4:6]) - int(source_yearmon[4:6])
    time_interval = year_interval + month_interval
    age = source_mesh_name.split('_')[2]
    if source_mesh_name.split('_')[3]=='Female':
        gender = 0
    else:
        gender = 1
    return time_interval, age, gender

# Datasets for GCNGNNEND
class AortaMesh_Dataset(data.Dataset):
    def __init__(self, data_lists, label_lists, transform=None, target_transform=None):
        super(AortaMesh_Dataset, self).__init__()
        self.data_paths = data_lists    # Could be a list: ['./train/input/image_1.obj', './train/input/image_2.obj', ...]
        self.label_paths = label_lists  # Could be a nested list: ['./train/GT/image_1_1.obj', './train/GT/image_1_2.obj', ...]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        source_mesh_name, source_vert, target_vert, label, source_faces = load_mesh_data(self.data_paths, index)
        return source_mesh_name, source_vert, target_vert, label, source_faces

    def __len__(self):
        return len(self.data_paths)

class AortaMesh_TestDataset(data.Dataset):
    def __init__(self, data_lists, time_interval):
        super(AortaMesh_TestDataset, self).__init__()
        self.data_paths = data_lists
        self.time_interval = time_interval

    def __getitem__(self, index):
        source_mesh_name, target_mesh_name, source_verts, target_verts, label, source_faces = load_test_mesh_data(self.data_paths, self.time_interval, index)
        return source_mesh_name, target_mesh_name, source_verts, target_verts, label, source_faces

    def __len__(self):
        return len(self.data_paths)

def load_mesh_data(data_path, index):
    # trimesh: Load 3D Meshes VTK
    source_mesh_path=data_path[index]
    source_mesh_name = source_mesh_path.split('/')[-1]
    patient_ID = source_mesh_name.split('_')[0]
    target_meshes_path=[target_mesh_path for target_mesh_path in data_path if patient_ID in target_mesh_path and target_mesh_path!=source_mesh_path]
    target_mesh_path = target_meshes_path[random.randint(0, len(target_meshes_path)-1)]
    target_mesh_name = target_mesh_path.split('/')[-1]
    time_intervel, age, gender = get_timeinterval(source_mesh_name, target_mesh_name)
    label = str_to_tensor_interval_gender_age(time_intervel, age, gender)
    source_verts, source_faces, source_aux = load_obj(source_mesh_path)
    target_verts, target_faces, target_aux = load_obj(target_mesh_path)

    # Faces to Edges in GNN
    source_face_transpose = source_faces.verts_idx.transpose(0, 1)
    source_transform_face2edge = T.FaceToEdge()
    source_data = Data(pos=source_verts, face=source_face_transpose)
    source_data = source_transform_face2edge(source_data)
    source_faces = source_data.edge_index
    # target_face_transpose = target_faces.verts_idx.transpose(0, 1)
    # target_transform_face2edge = T.FaceToEdge()
    # target_data = Data(pos=target_verts, face=target_face_transpose)
    # target_data = target_transform_face2edge(target_data)
    # target_faces = target_data.edge_index
    return source_mesh_name, source_verts, target_verts, label, source_faces

def load_test_mesh_data(data_path, time_interval, index):
    # trimesh: Load 3D Meshes VTK
    source_mesh_path=data_path[index]
    source_mesh_name = source_mesh_path.split('/')[-1]
    ##  Without Ground Truth
    # PID, date, agestr, genderstr = source_mesh_name.split('_')
    # age = int(agestr)
    # if 'Female' in genderstr:
    #     gender = 0
    # else:
    #     gender = 1
    # label = str_to_tensor_interval_gender_age(time_interval, age, gender)

    ##  With Ground Truth
    patient_ID = source_mesh_name.split('_')[0]
    target_meshes_path=[target_mesh_path for target_mesh_path in data_path if patient_ID in target_mesh_path and target_mesh_path!=source_mesh_path]
    target_mesh_path = target_meshes_path[random.randint(0, len(target_meshes_path)-1)]
    target_mesh_name = target_mesh_path.split('/')[-1]
    time_intervel, age, gender = get_timeinterval(source_mesh_name, target_mesh_name)
    label = str_to_tensor_interval_gender_age(time_intervel, age, gender)
    source_verts, source_faces, source_aux = load_obj(source_mesh_path)
    target_verts, target_faces, target_aux = load_obj(target_mesh_path)

    # Faces to Edges in GNN
    source_face_transpose = source_faces.verts_idx.transpose(0, 1)
    source_transform_face2edge = T.FaceToEdge()
    source_data = Data(pos=source_verts, face=source_face_transpose)
    source_data = source_transform_face2edge(source_data)
    source_faces = source_data.edge_index
    return source_mesh_name, target_mesh_name, source_verts, target_verts, label, source_faces

def save_mesh_data(save_path, meshname, generated_vert, source_faces):
    # pytorch3d: Save 3D Meshes: verts and faces
    save_obj(os.path.join(save_path, meshname), generated_vert, source_faces)
    return

def sort_to_classes(root, print_cycle=np.inf):

    def log(text):
        print('[UTKFace dset labeler] ' + text)

    log('Starting labeling process...')
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    if not files:
        raise FileNotFoundError('No image files in '+root)
    copied_count = 0
    sorted_folder = os.path.join(root, '..', 'labeled')
    if not os.path.isdir(sorted_folder):
        os.mkdir(sorted_folder)

    for f in files:
        matcher = consts.UTKFACE_ORIGINAL_IMAGE_FORMAT.match(f)
        if matcher is None:
            continue
        age, gender, dtime = matcher.groups()
        srcfile = os.path.join(root, f)
        label = Label(int(age), int(gender))
        dstfolder = os.path.join(sorted_folder, label.to_str())
        dstfile = os.path.join(dstfolder, dtime+'.jpg')
        if os.path.isfile(dstfile):
            continue
        if not os.path.isdir(dstfolder):
            os.mkdir(dstfolder)
        copyfile(srcfile, dstfile)
        copied_count += 1
        if copied_count % print_cycle == 0:
            log('Copied %d files.' % copied_count)
    log('Finished labeling process.')


def get_fgnet_person_loader(root):
    return DataLoader(dataset=ImageFolder(root, transform=pil_to_model_tensor_transform), batch_size=1)


def str_to_tensor(text, normalize=False):
    age_group, gender = text.split('.')
    age_tensor = -torch.ones(consts.NUM_AGES)
    age_tensor[int(age_group)] *= -1
    gender_tensor = -torch.ones(consts.NUM_GENDERS)
    gender_tensor[int(gender)] *= -1
    if normalize:
        gender_tensor = gender_tensor.repeat(consts.NUM_AGES // consts.NUM_GENDERS)
    result = torch.cat((age_tensor, gender_tensor), 0)
    return result

def str_to_tensor_interval_gender_age(time_interval, age, gender, normalize=True):
    # interval vector
    interval_tensor = torch.ones(consts.NUM_INTERVELS)
    if time_interval>50 or time_interval<-49:
        # print('Invalid time interval, -49<time_interval<50')
        if time_interval>50:
            time_interval=50
        else:
            time_interval=-49

    if int(time_interval)==-49:
        interval_tensor[int(49 + time_interval)] *= 0
    else:
        interval_tensor[0:int(49+time_interval)] *= 0  # time interval lies between -49 month and 50 month
    # age vector
    if int(age)>100 or int(age)<1:
        print('Invalid age, 1=<age<=100')
    age_tensor = torch.ones(consts.NUM_AGES)
    age_tensor[0:int(age)] *= 0

    gender_tensor = torch.ones(consts.NUM_GENDERS)
    gender_tensor[int(gender)] *= 0
    if normalize:
        gender_tensor = gender_tensor.repeat(consts.NUM_AGES // consts.NUM_GENDERS)
    result = torch.cat((interval_tensor, age_tensor), 0)
    result = torch.cat((result, gender_tensor), 0)
    return result


class Label(namedtuple('Label', ('age', 'gender'))):
    def __init__(self, age, gender):
        super(Label, self).__init__()
        self.age_group = self.age_transform(self.age)

    def to_str(self):
        return '%d.%d' % (self.age_group, self.gender)

    @staticmethod
    def age_transform(age):
        age -= 1
        if age < 20:
            # first 4 age groups are for kids <= 20, 5 years intervals
            return max(age // 5, 0)
        else:
            # last (6?) age groups are for adults > 20, 10 years intervals
            return min(4 + (age - 20) // 10, consts.NUM_AGES - 1)

    def to_tensor(self, normalize=False):
        return str_to_tensor(self.to_str(), normalize=normalize)

fmt_t = "%H_%M"
fmt = "%Y_%m_%d"

def default_train_results_dir():
    return os.path.join('.', 'trained_models', datetime.datetime.now().strftime(fmt), datetime.datetime.now().strftime(fmt_t))

def default_where_to_save(eval=True):
    path_str = os.path.join('.', 'results', datetime.datetime.now().strftime(fmt), datetime.datetime.now().strftime(fmt_t))
    if not os.path.exists(path_str):
        os.makedirs(path_str)

def default_test_results_dir(eval=True):
    return os.path.join('.', 'test_results', datetime.datetime.now().strftime(fmt) if eval else fmt)

def print_timestamp(s):
    print("[{}] {}".format(datetime.datetime.now().strftime(fmt_t.replace('_', ':')), s))

class LossTracker(object):
    def __init__(self, use_heuristics=False, plot=False, eps=1e-3):
        # assert 'train' in names and 'valid' in names, str(names)
        self.losses = defaultdict(lambda: [])
        self.paths = []
        self.epochs = 0
        self.use_heuristics = use_heuristics
        if plot:
           # print("names[-1] - "+names[-1])
            plt.ion()
            plt.show()
        else:
            plt.switch_backend("agg")

    # deprecated
    def append(self, train_loss, valid_loss, tv_loss, uni_loss, path):
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        self.tv_losses.append(tv_loss)
        self.uni_losses.append(uni_loss)
        self.paths.append(path)
        self.epochs += 1
        if self.use_heuristics and self.epochs >= 2:
            delta_train = self.train_losses[-1] - self.train_losses[-2]
            delta_valid = self.valid_losses[-1] - self.valid_losses[-2]
            if delta_train < -self.eps and delta_valid < -self.eps:
                pass  # good fit, continue training
            elif delta_train < -self.eps and delta_valid > +self.eps:
                pass  # overfit, consider stop the training now
            elif delta_train > +self.eps and delta_valid > +self.eps:
                pass  # underfit, if this is in an advanced epoch, break
            elif delta_train > +self.eps and delta_valid < -self.eps:
                pass  # unknown fit, check your model, optimizers and loss functions
            elif 0 < delta_train < +self.eps and self.epochs >= 3:
                prev_delta_train = self.train_losses[-2] - self.train_losses[-3]
                if 0 < prev_delta_train < +self.eps:
                    pass  # our training loss is increasing but in less than eps,
                    # this is a drift that needs to be caught, consider lower eps next time
            else:
                pass  # saturation \ small fluctuations

    def append_single(self, name, value):
        self.losses[name].append(value)

    def append_many(self, **names):
        for name, value in names.items():
            self.append_single(name, value)

    def append_many_and_plot(self, **names):
        self.append_many(**names)

    def plot(self):
        print("in plot")
        plt.clf()
        graphs = [plt.plot(loss, label=name)[0] for name, loss in self.losses.items()]
        plt.legend(handles=graphs)
        plt.xlabel('Epochs')
        plt.ylabel('Averaged loss')
        plt.title('Losses by epoch')
        plt.grid(True)
        plt.draw()
        plt.pause(0.001)

    @staticmethod
    def show():
        print("in show")
        plt.show()

    @staticmethod
    def save(path):
        plt.savefig(path, transparent=True)

    def __repr__(self):
        ret = {}
        for name, value in self.losses.items():
            ret[name] = value[-1]
        return str(ret)

def mean(l):
    return np.array(l).mean()

def uni_loss(input):
    assert len(input.shape) == 2
    batch_size, input_size = input.size()
    hist = torch.histc(input=input, bins=input_size, min=-1, max=1)
    return mse(hist, batch_size * torch.ones_like(hist)) / input_size

def easy_deconv(in_dims, out_dims, kernel, stride=1, groups=1, bias=True, dilation=1):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)

    c_in, h_in, w_in = in_dims
    c_out, h_out, w_out = out_dims

    padding = [0, 0]
    output_padding = [0, 0]

    lhs_0 = -h_out + (h_in - 1) * stride[0] + kernel[0]  # = 2p[0] - o[0]
    if lhs_0 % 2 == 0:
        padding[0] = lhs_0 // 2
    else:
        padding[0] = lhs_0 // 2 + 1
        output_padding[0] = 1

    lhs_1 = -w_out + (w_in - 1) * stride[1] + kernel[1]  # = 2p[1] - o[1]
    if lhs_1 % 2 == 0:
        padding[1] = lhs_1 // 2
    else:
        padding[1] = lhs_1 // 2 + 1
        output_padding[1] = 1

    return torch.nn.ConvTranspose2d(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=kernel,
        stride=stride,
        padding=tuple(padding),
        output_padding=tuple(output_padding),
        groups=groups,
        bias=bias,
        dilation=dilation
    )


def remove_trained(folder):
    if os.path.isdir(folder):
        removed_ctr = 0
        for tm in os.listdir(folder):
            tm = os.path.join(folder, tm)
            if os.path.splitext(tm)[1] == consts.TRAINED_MODEL_EXT:
                try:
                    os.remove(tm)
                    removed_ctr += 1
                except OSError as e:
                    print("Failed removing {}: {}".format(tm, e))
        if removed_ctr > 0:
            print("Removed {} trained models from {}".format(removed_ctr, folder))


def merge_images(batch1, batch2):
    assert batch1.shape == batch2.shape
    merged = torch.zeros(batch1.size(0) * 2, batch1.size(1), batch1.size(2), batch1.size(3), dtype=batch1.dtype)
    for i, (image1, image2) in enumerate(zip(batch1, batch2)):
        merged[2 * i] = image1
        merged[2 * i + 1] = image2
    return merged
