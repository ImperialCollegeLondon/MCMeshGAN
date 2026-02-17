import re
from utils import *
import consts
import logging
import random
from torch.nn.functional import l1_loss
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits_loss
from torch.optim import Adam
import torch.nn as nn
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
import numpy as np
from torch_geometric.nn import GCNConv, PointNetConv, GCN2Conv, SSGConv, PMLP
import pytorch3d.ops as ops
from pytorch3d.io import load_obj
import matplotlib.pyplot as plt
import torch.nn.functional as F

def data_distribution(data, name=''):
    # data = np.random.rand(1000)
    plt.hist(data, bins=100)
    plt.show()
    plt.savefig(name+'_plot_hist.png')
    plt.close()


def knn_point(k, xyz1, xyz2):
    xyz1 = xyz1.transpose(1, 2)
    xyz2 = xyz2.transpose(1, 2)
    dist, idx, _ = ops.knn_points(xyz1, xyz2, K=k)
    return dist.transpose(1,2), idx.transpose(1,2)

def group_point(x, idx):
    x = x.transpose(1,2)
    idx = idx.transpose(1,2)
    # print(x.shape, idx.shape)
    feature = ops.knn_gather(x, idx)
    return feature.transpose(1,3)

def group(xyz, points, k):
    _, idx = knn_point(k+1, xyz, xyz)
    idx = idx[:,1:,:] # exclude self matching
    grouped_xyz = group_point(xyz, idx) # (batch_size, num_dim, k, num_points)
    b,d,n = xyz.shape
    grouped_xyz -= xyz.unsqueeze(2).expand(-1,-1,k,-1) # translation normalization, (batch_size, num_points, k, num_dim(3))

    if points is not None:
        grouped_points = group_point(points, idx)
    else:
        grouped_points = grouped_xyz

    return grouped_xyz, grouped_points, idx


def farthest_point_sample(xyz, npoint):
    xyz = xyz.transpose(1, 2)
    points, inds = ops.sample_farthest_points(xyz,K=npoint)
    return points

def pool(xyz, points, k, npoint):

    new_xyz = farthest_point_sample(xyz, npoint)
    new_xyz = new_xyz.transpose(1,2)

    _, idx = knn_point(k, new_xyz, xyz)
    new_points, _ = torch.max(group_point(points, idx),dim=2)

    return new_xyz, new_points

def init_weight_(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

class FeatureNet(nn.Module):
    def __init__(self, k=8, dim=128, num_block=3):
        super(FeatureNet, self).__init__()
        self.k = k
        self.num_block = num_block

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(3, dim, 1, 1 ,bias=False))
        for i in range(self.num_block - 1):
            self.conv_layers.append(nn.Conv2d(dim, dim, 1, 1,bias=False))

        self.conv_layers.apply(init_weight_)

    def forward(self, x):
        _, out, _ = group(x, None, self.k) # (batch_size, num_dim(3), k, num_points)

        for conv_layer in self.conv_layers:
            out = F.relu(conv_layer(out))

        out, _ = torch.max(out, dim=2) # (batch_size, num_dim, num_points)

        return out


class AgeGenderNet(nn.Module):
    def __init__(self, num_block=3):
        super(AgeGenderNet, self).__init__()
        self.num_block = num_block

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(consts.NUM_AGES_GENDERS_INTERVELS, 10*consts.NUM_AGES_GENDERS_INTERVELS, 1, 1 ,bias=False))
        for i in range(self.num_block - 1):
            self.conv_layers.append(nn.Conv1d(10*consts.NUM_AGES_GENDERS_INTERVELS, 10*consts.NUM_AGES_GENDERS_INTERVELS, 1, 1,bias=False))
        self.conv_layers.append(nn.Conv1d(10*consts.NUM_AGES_GENDERS_INTERVELS, 3*consts.NUM_VERTICES, 1, 1, bias=False))
        self.conv_layers.apply(init_weight_)

    def forward(self, x):
        out = x
        for conv_layer in self.conv_layers:
            out = F.relu(conv_layer(out))
        return out

class ResGraphConvUnpool(nn.Module):
    def __init__(self, k=8, in_dim=128, dim=128):
        super(ResGraphConvUnpool, self).__init__()
        self.k = k
        self.num_blocks = 12
        self.bn_relu_layers = nn.ModuleList()
        for i in range(self.num_blocks):
            self.bn_relu_layers.append(nn.ReLU())
        self.conv_layers = nn.ModuleList()
        for i in range(self.num_blocks):
            self.conv_layers.append(nn.Conv2d(in_dim, dim, 1, 1,bias=False))
            self.conv_layers.append(nn.Conv2d(dim, dim, 1, 1,bias=False))
        # self.unpool_center_conv = nn.Conv2d(dim, 3, 1, 1,bias=False)
        # self.unpool_neighbor_conv = nn.Conv2d(dim, 3, 1, 1,bias=False)
        self.unpool_center_conv = nn.Sequential(
            nn.Conv2d(dim, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.unpool_neighbor_conv = nn.Sequential(
            nn.Conv2d(dim, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.conv_layers.apply(init_weight_)
        self.unpool_center_conv.apply(init_weight_)
        self.unpool_neighbor_conv.apply(init_weight_)

    def forward(self, xyz, points):
        # xyz: (batch_size, num_dim(3), num_points)
        # points: (batch_size, num_dim(128), num_points)
        indices = None
        for idx in range(self.num_blocks):
            shortcut = points # (batch_size, num_dim(128), num_points)
            points = self.bn_relu_layers[idx](points) # ReLU
            if idx == 0 and indices is None:
                _, grouped_points, indices = group(xyz, points, self.k) # (batch_size, num_dim, k, num_points)
            else:
                grouped_points = group_point(points, indices)
            # Center Conv
            b,d,n = points.shape
            center_points = points.view(b, d, 1, n)
            points = self.conv_layers[2 * idx](center_points)  # (batch_size, num_dim(128), 1, num_points)
            # Neighbor Conv
            grouped_points_nn = self.conv_layers[2 * idx + 1](grouped_points)
            # Fusion of Center and Neighbour: Concat and Mean
            points = torch.mean(torch.cat((points, grouped_points_nn), dim=2), dim=2) + shortcut

            if idx == self.num_blocks-1:
                num_points = xyz.shape[-1]
                # Center Conv
                points_xyz = self.unpool_center_conv(center_points) # (batch_size, 3*up_ratio, 1, num_points)
                # Neighbor Conv
                grouped_points_xyz = self.unpool_neighbor_conv(grouped_points) # (batch_size, 3*up_ratio, k, num_points)
                # Fusion of Center and Neighbour: Concat and Mean
                new_xyz = torch.mean(torch.cat((points_xyz, grouped_points_xyz), dim=2), dim=2) # (batch_size, 3*up_ratio, num_points)
                new_xyz = new_xyz
                return new_xyz, points

class MultiScaleVertexAttention(nn.Module):
    """
    Inputs:
        x5000: (B, 3, 5000)
        x2500: (B, 3, 2500)
        x1250: (B, 3, 1250)

    Output:
        fused: (B, 3, 10000)
        weights: (B, 3)
    """
    def __init__(self, channels=3, reduction=4, out_vertices=10000):
        super(MultiScaleVertexAttention, self).__init__()
        self.out_vertices = out_vertices

        # global pooling to extract scale-level descriptor
        self.pool = nn.AdaptiveAvgPool1d(1)

        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1, bias=False)
        )

    def _logit(self, x):
        # x: (B, C, N)
        g = self.pool(x).squeeze(-1)     # (B, C)
        logit = self.mlp(g)              # (B, 1)
        return logit

    def forward(self, x5000, x2500, x1250):
        B, C, _ = x5000.shape

        # --------- 1) Upsample all to 10000 ----------
        x5000_up = F.interpolate(x5000, size=self.out_vertices, mode='linear', align_corners=False)
        x2500_up = F.interpolate(x2500, size=self.out_vertices, mode='linear', align_corners=False)
        x1250_up = F.interpolate(x1250, size=self.out_vertices, mode='linear', align_corners=False)

        # --------- 2) Scale attention logits ---------
        l5000 = self._logit(x5000_up)   # (B,1)
        l2500 = self._logit(x2500_up)   # (B,1)
        l1250 = self._logit(x1250_up)   # (B,1)

        logits = torch.cat([l5000, l2500, l1250], dim=1)  # (B,3)
        weights = F.softmax(logits, dim=1)                # (B,3)

        # --------- 3) Weighted fusion ---------
        # stack: (B,3,C,10000)
        x_stack = torch.stack([x5000_up, x2500_up, x1250_up], dim=1)

        # reshape weights → (B,3,1,1)
        w = weights.view(B, 3, 1, 1)

        fused = (w * x_stack).sum(dim=1)                  # (B, 3, 10000)

        return fused, weights


# ======================================================================
# =========================MCMeshGAN-KCNGCN=============================
# ======================================================================
class MeshKCNGCNDiscriminatorImg(nn.Module):
    def __init__(self):
        super(MeshKCNGCNDiscriminatorImg, self).__init__()
        in_dims = (3, 32 + consts.NUM_AGES_GENDERS_INTERVELS, 32, 64)
        out_dims = (32, 32, 64, 64)
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims), 1):
            self.conv_layers.add_module(
                'dimg_conv_%d' % i,
                nn.Sequential(
                    nn.Conv1d(in_dim, out_dim, kernel_size=2, stride=2),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU()
                )
            )

        self.fc_layers.add_module(
            'dimg_fc_1',
            nn.Sequential(
                nn.Linear(40000, 1024),  # the number should be assigned manualy according to the size of your input
                nn.LeakyReLU()
            )
        )

        self.fc_layers.add_module(
            'dimg_fc_2',
            nn.Sequential(
                nn.Linear(1024, 1),
                # nn.Sigmoid()  # commented out because logits are needed
            )
        )

    def forward(self, imgs, labels, device):
        out = imgs
        out=out.permute(0, 2, 1)
        # run convs
        for i, conv_layer in enumerate(self.conv_layers, 1):
            # print(out.shape)
            # print(conv_layer)
            out = conv_layer(out)
            if i == 1:
                # concat labels after first conv
                labels_tensor = torch.zeros(torch.Size((out.size(0), labels.size(1), out.size(2))), device=device)
                for img_idx in range(out.size(0)):
                    for label in range(labels.size(1)):
                        labels_tensor[img_idx, label, :] = labels[img_idx, label]  # fill a square
                out = torch.cat((out, labels_tensor), 1)

        # run fcs
        out = out.flatten(1, -1)
        for fc_layer in self.fc_layers:
            out = fc_layer(out)
            # print(out.shape)
            # print(fc_layer)
        return out

class MeshKCNGCNEncoder(nn.Module):
    def __init__(self):
        super(MeshKCNGCNEncoder, self).__init__()

        # # ===========The Conditional branch==============
        self.agegendernet = AgeGenderNet(3)

        # # ===========The KNN-based KCN branch==============
        self.k = 8
        self.feat_dim = 128
        self.res_conv_dim = 128
        self.inital_CNNFeatureNet = FeatureNet(self.k, self.feat_dim,3)
        self.res_unpool_1 = ResGraphConvUnpool(self.k, self.feat_dim, self.res_conv_dim)

        # # ===========The Multi-scale attention over xyz_5000/2500/1250==============
        self.ms_attn = MultiScaleVertexAttention(channels=3, reduction=4)

        # # ================The GNN brach============================
        # 1. Original Graph Convolutional Network
        self.num_node_features = 3
        self.output_size = 3
        self.conv1 = GCNConv(self.num_node_features, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 32)
        self.conv4 = GCNConv(32, self.output_size)

        # # 2. PointNet++
        # self.conv1 = PointNetConv()
        # self.conv2 = PointNetConv()
        # self.conv3 = PointNetConv()
        # self.conv4 = PointNetConv()

        # # 3. GCN2Conv
        # self.conv1 = GCN2Conv(channels=3, alpha=0.1)
        # self.conv2 = GCN2Conv(channels=3, alpha=0.1)
        # self.conv3 = GCN2Conv(channels=3, alpha=0.1)
        # self.conv4 = GCN2Conv(channels=3, alpha=0.1)

        # # 4. SSGConv
        # self.conv1 = SSGConv(self.num_node_features, 32, alpha=0.1)
        # self.conv2 = SSGConv(32, 64, alpha=0.1)
        # self.conv3 = SSGConv(64, 32, alpha=0.1)
        # self.conv4 = SSGConv(32, self.output_size, alpha=0.1)

        # # 5. PMLP
        # self.conv1 = PMLP(in_channels=self.num_node_features, hidden_channels=3, out_channels=3, num_layers=3, dropout= 0.5)
        # self.conv2 = PMLP(in_channels=3, hidden_channels=3, out_channels=3, num_layers=3, dropout= 0.5)
        # self.conv3 = PMLP(in_channels=3, hidden_channels=3, out_channels=3, num_layers=3, dropout= 0.5)
        # self.conv4 = PMLP(in_channels=3, hidden_channels=3, out_channels=3, num_layers=3, dropout= 0.5)



    def forward(self, xyz, xyz_5000, xyz_2500, xyz_1250, edge_index, age_gender):
    # def forward(self, xyz_10000, xyz_5000, xyz_2500, xyz_1250, edge_index, age_gender):
        # # ================The Conditional branch============================
        batch_size = len(xyz)
        age_gender = age_gender.view(batch_size, consts.NUM_AGES_GENDERS_INTERVELS, 1)
        age_gender = self.agegendernet(age_gender)
        age_gender = age_gender.view(batch_size, 3, consts.NUM_VERTICES)
        condition_xyz = age_gender

        # # ================The Multi-Scale KNN-based KCN branch============================
        # # =====The branch for 5000 vertices=====
        xyz_5000 = xyz_5000.transpose(1, 2)  # (batch_size, 3, num_points)
        points = self.inital_CNNFeatureNet(xyz_5000)  # (batch_size, feat_dim, num_points)
        new_xyz_5000, points = self.res_unpool_1(xyz_5000, points)

        # # =====The branch for 2500 vertices=====
        xyz_2500 = xyz_2500.transpose(1, 2)  # (batch_size, 3, num_points)
        points = self.inital_CNNFeatureNet(xyz_2500)  # (batch_size, feat_dim, num_points)
        new_xyz_2500, points = self.res_unpool_1(xyz_2500, points)

        # # =====The branch for 5000 vertices=====
        xyz_1250 = xyz_1250.transpose(1, 2)  # (batch_size, 3, num_points)
        points = self.inital_CNNFeatureNet(xyz_1250)  # (batch_size, feat_dim, num_points)
        new_xyz_1250, points = self.res_unpool_1(xyz_1250, points)

        # # ===== Multi-scale attention fusion (all: B,3,Ni) =====
        fused_xyz, scale_weights = self.ms_attn(new_xyz_5000, new_xyz_2500, new_xyz_1250)
        # fused_xyz: (B,3,1000), scale_weights: (B,3)
        KCN_new_xyz = fused_xyz


        # # ================The GNN branch============================
        # ===========Comparison of different Graph Structures==============
        xyz_GNN = torch.squeeze(xyz)
        edge_index_GNN = torch.squeeze(edge_index)
        # # 1. Original Graph Convolutional Network
        x_GNN = self.conv1(xyz_GNN, edge_index_GNN)
        x_GNN = F.relu(x_GNN)
        x_GNN = F.dropout(x_GNN, training=self.training)
        x_GNN = self.conv2(x_GNN, edge_index_GNN)
        x_GNN = F.relu(x_GNN)
        x_GNN = F.dropout(x_GNN, training=self.training)
        x_GNN = self.conv3(x_GNN, edge_index_GNN)
        x_GNN = F.relu(x_GNN)
        x_GNN = F.dropout(x_GNN, training=self.training)
        x_GNN = self.conv4(x_GNN, edge_index_GNN)
        x_GNN = F.relu(x_GNN)

        # # ================Fusion of Conditional branch, KCN brach, and GNN branch================
        GNN_new_xyz = x_GNN.unsqueeze(0).transpose(1, 2)
        new_xyz = condition_xyz+GNN_new_xyz+KCN_new_xyz
        return new_xyz

class MeshKCNGCNGenerator(nn.Module):
    def __init__(self):
        super(MeshKCNGCNGenerator, self).__init__()
        self.k = 8
        self.feat_dim = 128
        self.res_conv_dim = 128
        self.featurenet = FeatureNet(self.k, self.feat_dim,3)
        self.res_unpool_1 = ResGraphConvUnpool(self.k, self.feat_dim, self.res_conv_dim)
        self.res_unpool_2 = ResGraphConvUnpool(self.k, self.res_conv_dim, self.res_conv_dim)
        self.res_unpool_3 = ResGraphConvUnpool(self.k, self.res_conv_dim, self.res_conv_dim)
        self.res_unpool_4 = ResGraphConvUnpool(self.k, self.res_conv_dim, self.res_conv_dim)

    def forward(self, xyz, input_xyz):
        """
        xyz:       (B, 3, V)
        input_xyz: (B, V, 3)
        """
        points = self.featurenet(xyz) # (batch_size, feat_dim, num_points)
        new_xyz, points = self.res_unpool_1(xyz, points)
        new_xyz, points = self.res_unpool_2(new_xyz, points)
        new_xyz, points = self.res_unpool_3(new_xyz, points)
        new_xyz, points = self.res_unpool_4(new_xyz, points)
        new_xyz = new_xyz.transpose(1, 2)+input_xyz
        return new_xyz

class MeshKCNGCNNet(object):
    def __init__(self):
        self.E = MeshKCNGCNEncoder()
        self.Dimg = MeshKCNGCNDiscriminatorImg()
        self.G = MeshKCNGCNGenerator()

        self.eg_optimizer = Adam(list(self.E.parameters()) + list(self.G.parameters()))
        self.di_optimizer = Adam(self.Dimg.parameters())

        self.device = None
        self.cpu()  # initial, can later move to cuda

    def __call__(self, *args, **kwargs):
        self.test_single(*args, **kwargs)

    def __repr__(self):
        return os.linesep.join([repr(subnet) for subnet in (self.E, self.Dimg, self.G)])

    def morph(self, image_tensors, ages, genders, length, target):

        self.eval()

        original_vectors = [None, None]
        for i in range(2):
            z = self.E(image_tensors[i].unsqueeze(0))
            l = Label(ages[i], genders[i]).to_tensor(normalize=True).unsqueeze(0).to(device=z.device)
            z_l = torch.cat((z, l), 1)
            original_vectors[i] = z_l

        z_vectors = torch.zeros((length + 1, z_l.size(1)), dtype=z_l.dtype)
        for i in range(length + 1):
            z_vectors[i, :] = original_vectors[0].mul(length - i).div(length) + original_vectors[1].mul(i).div(length)

        generated = self.G(z_vectors)
        dest = os.path.join(target, 'morph.png')
        save_image_normalized(tensor=generated, filename=dest, nrow=generated.size(0))
        print_timestamp("Saved test result to " + dest)
        return dest

    def kids(self, image_tensors, length, target):

        self.eval()

        original_vectors = [None, None]
        for i in range(2):
            z = self.E(image_tensors[i].unsqueeze(0)).squeeze(0)
            original_vectors[i] = z

        z_vectors = torch.zeros((length, consts.NUM_Z_CHANNELS), dtype=z.dtype)
        z_l_vectors = torch.zeros((length, consts.NUM_Z_CHANNELS + consts.LABEL_LEN_EXPANDED), dtype=z.dtype)
        for i in range(length):
            for j in range(consts.NUM_Z_CHANNELS):
                r = random.random()
                z_vectors[i][j] = original_vectors[0][j].mul(r) + original_vectors[1][j].mul(1 - r)

            fake_age = 0
            fake_gender = random.choice([consts.MALE, consts.FEMALE])
            l = Label(fake_age, fake_gender).to_tensor(normalize=True).to(device=z.device)
            z_l = torch.cat((z_vectors[i], l), 0)
            z_l_vectors[i, :] = z_l

        generated = self.G(z_l_vectors)
        dest = os.path.join(target, 'kids.png')
        save_image_normalized(tensor=generated, filename=dest, nrow=generated.size(0))
        print_timestamp("Saved test result to " + dest)
        return dest

    def test_single(self, data_list, target, epoch_id, watermark):
        with torch.no_grad():  # test
            txt_results = open('MAECD.txt', 'a+')
            dataset = AortaMesh_TestDataset(data_list)
            test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=True)
            save_path = '/mnt/storage/home/lchen6/lchen6/Remote/AgeAortaGCN/demo_results/'

            L1_loss_list = []
            CD_loss_list = []
            L1_loss_list_str = ''
            CD_loss_list_str = ''
            with torch.no_grad():  # test
                self.eval()  # move to test mode

                for ii, (source_mesh_names, target_mesh_names, source_verts, source_verts_5000, source_verts_2500, source_verts_1250, target_verts, labels, source_faces) in enumerate(test_loader, 1):
                    # Encoding Stage
                    source_verts = source_verts.to(device=self.device)
                    source_verts_5000 = source_verts_5000.to(device=self.device)
                    source_verts_2500 = source_verts_2500.to(device=self.device)
                    source_verts_1250 = source_verts_1250.to(device=self.device)
                    target_verts = target_verts.to(device=self.device)
                    source_faces = source_faces.to(device=self.device)
                    labels = torch.stack([element for element in labels])
                    labels = labels.to(device=self.device)
                    age_gender = labels
                    z = self.E(source_verts, source_verts_5000, source_verts_2500, source_verts_1250, source_faces,
                               age_gender)
                    # Decoding Stage
                    generated_verts = self.G(z, source_verts)
                    generated_verts_np = torch.squeeze(generated_verts)
                    generated_verts_np = generated_verts_np.cpu().numpy()
                    target_verts_np = torch.squeeze(target_verts)
                    target_verts_np = target_verts_np.cpu().numpy()
                    L1_distance = np.mean(np.abs(generated_verts_np - target_verts_np))
                    CD_distance = evaluate_EMD_CD(generated_verts_np, target_verts_np)

                    L1_loss_list.append(L1_distance)
                    CD_loss_list.append(CD_distance)
                    L1_loss_list_str = L1_loss_list_str + str(L1_distance) + ', '
                    CD_loss_list_str = CD_loss_list_str + str("{:.7f}".format(CD_distance)) + ', '

                    print('Predicted-Target: L1 Distance: ' + str(L1_distance) + ', Chamfer Distance: ' + str(
                        CD_distance))

                    source_verts_np = torch.squeeze(source_verts)
                    source_verts_np = source_verts_np.cpu().numpy()
                    source_L1_distance = np.mean(np.abs(source_verts_np - target_verts_np))
                    source_CD_distance = evaluate_EMD_CD(source_verts_np, target_verts_np)
                    print(
                        'Source-Target: L1 Distance: ' + str(source_L1_distance) + ', Chamfer Distance: ' + str(
                            source_CD_distance))

                    # Save predicted meshes
                    root_path = os.path.dirname(data_list[0])
                    for ii in range(0, len(target_mesh_names)):
                        source_mesh_name = source_mesh_names[ii]
                        target_mesh_name = target_mesh_names[ii]
                        source_verts, source_faces, source_aux = load_obj(os.path.join(root_path, source_mesh_name))
                        cur_epoch = epoch_id.split('/')[-2]
                        target_mesh_name = target_mesh_name.replace('.obj', str(cur_epoch) + '.obj')
                        # # Without GT
                        # if time_interval > 0:
                        #     mesh_name = target_mesh_name.replace('.obj', str(time_interval) + 'MonthLater.obj')
                        # else:
                        #     mesh_name = target_mesh_name.replace('.obj', str(time_interval) + 'MonthAgo.obj')
                        save_meshdata(save_path, target_mesh_name, generated_verts[ii], source_faces.verts_idx)

                MAE_L1_Loss = np.mean(L1_loss_list)
                STD_L1_Loss = np.std(L1_loss_list)
                MAE_CD_Loss = np.mean(CD_loss_list)
                STD_CD_Loss = np.std(CD_loss_list)


                txt_results.writelines(epoch_id + '\n')
                txt_results.writelines(L1_loss_list_str)
                txt_results.writelines('\n')
                txt_results.writelines(CD_loss_list_str)
                txt_results.writelines('\n')
                txt_results.writelines(
                    'MAE_L1_Loss: ' + str(MAE_L1_Loss) + ', STD: ' + str(STD_L1_Loss) + ', MSE_CD_Loss: ' + str(
                        MAE_CD_Loss) + ', STD: ' + str(STD_CD_Loss))
                txt_results.writelines('\n')
                txt_results.writelines('\n')
                txt_results.close()


    # ===================================================================================
    # Split the training and validation data at patient level
    # ===================================================================================
    @staticmethod
    def extract_ptaap_id(path: str) -> str:
        """Return the patient token like 'PTAAP203' from a file path."""
        m = re.search(r'(PTAAP\d+)', os.path.basename(path))
        if not m:
            raise ValueError(f"Could not find PTAAP id in: {path}")
        return m.group(1)

    # ===================================================================================
    # The complete training pipeline
    # ===================================================================================
    def teachMesh(
            self,
            data_list,
            label_list,
            batch_size=2,
            epochs=1,
            weight_decay=1e-5,
            lr=2e-4,
            should_plot=False,
            betas=(0.9, 0.999),
            valid_size=None,
            where_to_save=None,
            models_saving='always',
    ):
        where_to_save = where_to_save or default_where_to_save()
        # ===============================================================================
        # 1. Splitting the data into training and validation sets at patient level
        # ===============================================================================
        # (1) Group files by patient id
        patients = sorted({self.extract_ptaap_id(p) for p in data_list})
        random.seed(42)  # reproducible split
        random.shuffle(patients)

        # (2) Compute train:val=7:1 split by patient count, the test set is manually selected: train:val:test=7:1:2
        total_patients = len(patients)
        val_count = max(1, round(total_patients * (1 / 8)))  # 1/8 for validation
        val_patients = set(patients[:val_count])
        train_patients = set(patients[val_count:])

        # (3) Build file lists for each split
        train_data_list = [f for f in data_list if self.extract_ptaap_id(f) in train_patients]
        val_data_list = [f for f in data_list if self.extract_ptaap_id(f) in val_patients]
        train_label_list = [f for f in label_list if self.extract_ptaap_id(f) in train_patients]
        val_label_list = [f for f in label_list if self.extract_ptaap_id(f) in val_patients]

        # (4) Sanity checks
        train_ids = {self.extract_ptaap_id(f) for f in train_data_list}
        val_ids = {self.extract_ptaap_id(f) for f in val_data_list}
        overlap = train_ids & val_ids
        print(f"[Split] Total patients: {total_patients} | Train: {len(train_patients)} | Val: {len(val_patients)}")
        print(f"[Split] Samples -> Train: {len(train_data_list)} | Val: {len(val_data_list)}")
        print(f"[Split] Patient ID overlap? {bool(overlap)} (should be False)")

        # (5) Build datasets/loaders
        train_dataset = AortaMesh_Dataset(train_data_list, train_label_list)
        val_dataset = AortaMesh_Dataset(val_data_list, val_label_list)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # ===============================================================================
        # 2. Set up loss functions, optimizers, and trackers
        # ===============================================================================
        input_output_loss = l1_loss
        for optimizer in (self.eg_optimizer, self.di_optimizer):
            for param in ('weight_decay', 'betas', 'lr'):
                val = locals()[param]
                if val is not None:
                    optimizer.param_groups[0][param] = val

        loss_tracker = LossTracker(plot=should_plot)
        where_to_save_epoch = ""
        save_count = 0
        paths_for_gif = []

        # ===============================================================================
        # 3. Training loop
        # ===============================================================================
        for epoch in range(1, epochs + 1):
            # ===============================================================================
            # 3.1. Model Training
            # ===============================================================================
            where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch))
            try:
                losses = defaultdict(lambda: [])
                self.train()  # move to train mode
                for i, (source_mesh_names, target_mesh_names, source_verts, source_verts_5000, source_verts_2500, source_verts_1250, target_verts, labels, source_faces) in enumerate(train_loader, 1):
                    source_verts = source_verts.to(device=self.device)
                    source_verts_5000 = source_verts_5000.to(device=self.device)
                    source_verts_2500 = source_verts_2500.to(device=self.device)
                    source_verts_1250 = source_verts_1250.to(device=self.device)
                    target_verts = target_verts.to(device=self.device)
                    source_faces = source_faces.to(device=self.device)
                    labels = torch.stack([element for element in labels])
                    labels = labels.to(device=self.device)
                    age_gender = labels
                    z = self.E(source_verts, source_verts_5000, source_verts_2500, source_verts_1250, source_faces, age_gender)  # The encoder consists of Conditional, KCN and GNN branches

                    # Input\Output Loss
                    generated = self.G(z, source_verts)
                    eg_loss = input_output_loss(generated, target_verts)
                    losses['eg'].append(eg_loss.item())
                    # print('eg_loss:', eg_loss.item())

                    # Total Variance Regularization Loss
                    reg_loss = 0
                    # reg_loss.to(self.device)
                    losses['reg'].append(reg_loss)

                    # DiscriminatorImg Loss
                    d_i_input = self.Dimg(target_verts.detach(), labels.detach(), self.device)
                    d_i_output = self.Dimg(generated.detach(), labels.detach(), self.device)
                    di_input_loss = bce_with_logits_loss(d_i_input, torch.ones_like(d_i_input))
                    di_output_loss = bce_with_logits_loss(d_i_output, torch.zeros_like(d_i_output))
                    di_loss_tot = (di_input_loss + di_output_loss)
                    losses['di'].append(di_loss_tot.item())

                    # Generator\DiscriminatorImg Loss
                    dg_loss = 0.0001 * bce_with_logits_loss(d_i_output, torch.ones_like(d_i_output))
                    losses['dg'].append(dg_loss.item())

                    # Start back propagation
                    # Back prop on Encoder\Generator
                    self.eg_optimizer.zero_grad()
                    loss = eg_loss + dg_loss

                    loss.backward(retain_graph=True)
                    self.eg_optimizer.step()

                    # Back prop on DiscriminatorImg
                    self.di_optimizer.zero_grad()
                    di_loss_tot.backward()
                    self.di_optimizer.step()
                    now = datetime.datetime.now()

                logging.info('[{h}:{m}[Epoch {e}] Loss: {t}'.format(h=now.hour, m=now.minute, e=epoch, t=loss.item()))
                print_timestamp(f"[Epoch {epoch:d}] Loss: {loss.item():f}")
                print("Epoch Finished")
                # Save the model each 1000 epoch
                if epoch % 5 == 0:
                    if not os.path.exists(where_to_save_epoch):
                        os.makedirs(where_to_save_epoch)
                    paths_for_gif.append(where_to_save_epoch)

                    to_save_models = models_saving in ('always', 'tail')
                    cp_path = self.save(where_to_save_epoch, to_save_models=to_save_models)
                    if models_saving == 'tail':
                        prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                        remove_trained(prev_folder)
                    loss_tracker.save(os.path.join(cp_path, 'losses.png'))


                # ===============================================================================
                # 3.2. Model Validation
                # ===============================================================================
                with torch.no_grad():
                    self.eval()  # move to eval mode
                    for i, (source_mesh_names, target_mesh_names, source_verts, source_verts_5000, source_verts_2500,
                            source_verts_1250, target_verts, labels, source_faces) in enumerate(train_loader, 1):
                        # Encoding Stage
                        source_verts = source_verts.to(device=self.device)
                        source_verts_5000 = source_verts_5000.to(device=self.device)
                        source_verts_2500 = source_verts_2500.to(device=self.device)
                        source_verts_1250 = source_verts_1250.to(device=self.device)
                        target_verts = target_verts.to(device=self.device)
                        source_faces = source_faces.to(device=self.device)
                        labels = torch.stack([element for element in labels])
                        labels = labels.to(device=self.device)
                        age_gender = labels
                        z = self.E(source_verts, source_verts_5000, source_verts_2500, source_verts_1250, source_faces, age_gender)
                        # Decoding Stage
                        generated_verts = self.G(z, source_verts)
                        # Output Stage
                        # loss = input_output_loss(generated_verts, target_verts)
                        file_name = os.path.join(where_to_save_epoch, 'validation.png')
                        save_path='./val_results/'
                        root_path = os.path.dirname(data_list[0])
                        for ii in range(0, len(source_mesh_names)):
                            mesh_name = source_mesh_names[ii]
                            source_verts, source_faces, source_aux = load_obj(os.path.join(root_path, mesh_name))
                            mesh_name = 'Epoch ' + str(epoch) + ' ' + mesh_name
                            save_meshdata(save_path, mesh_name, generated_verts[ii], source_faces.verts_idx)
                        losses['valid'].append(loss.item())
                        break

                loss_tracker.append_many(**{k: mean(v) for k, v in losses.items()})
                loss_tracker.plot()

                logging.info(
                    '[{h}:{m}[Epoch {e}] Loss: {l}'.format(h=now.hour, m=now.minute, e=epoch, l=repr(loss_tracker)))

            except KeyboardInterrupt:
                print_timestamp("{br}CTRL+C detected, saving model{br}".format(br=os.linesep))
                if models_saving != 'never':
                    cp_path = self.save(where_to_save_epoch, to_save_models=True)
                if models_saving == 'tail':
                    prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                loss_tracker.save(os.path.join(cp_path, 'losses.png'))
                raise

        if models_saving == 'last':
            cp_path = self.save(where_to_save_epoch, to_save_models=True)
        loss_tracker.plot()


    def _mass_fn(self, fn_name, *args, **kwargs):
        """Apply a function to all possible Net's components.
        :return:
        """
        for class_attr in dir(self):
            if not class_attr.startswith('_'):  # ignore private members, for example self.__class__
                class_attr = getattr(self, class_attr)
                if hasattr(class_attr, fn_name):
                    fn = getattr(class_attr, fn_name)
                    fn(*args, **kwargs)

    def to(self, device):
        self._mass_fn('to', device=device)

    def cpu(self):
        self._mass_fn('cpu')
        self.device = torch.device('cpu')

    def cuda(self):
        self._mass_fn('cuda')
        self.device = torch.device('cuda')

    def eval(self):
        """Move Net to evaluation mode.

        :return:
        """
        self._mass_fn('eval')

    def train(self):
        """Move Net to training mode.

        :return:
        """
        self._mass_fn('train')

    def save(self, path, to_save_models=True):
        """Save all state dicts of Net's components.

        :return:
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        # path = os.path.join(path, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        if not os.path.isdir(path):
            os.mkdir(path)

        saved = []
        if to_save_models:
            for class_attr_name in dir(self):
                if not class_attr_name.startswith('_'):
                    class_attr = getattr(self, class_attr_name)
                    if hasattr(class_attr, 'state_dict'):
                        state_dict = class_attr.state_dict
                        fname = os.path.join(path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name))
                        torch.save(state_dict, fname)
                        saved.append(class_attr_name)

        if saved:
            print_timestamp("Saved {} to {}".format(', '.join(saved), path))
        elif to_save_models:
            raise FileNotFoundError("Nothing was saved to {}".format(path))
        return path

    def load(self, path, slim=True):
        """Load all state dicts of Net's components.

        :return:
        """
        loaded = []
        for class_attr_name in dir(self):
            if (not class_attr_name.startswith('_')) and ((not slim) or (class_attr_name in ('E', 'G'))):
                class_attr = getattr(self, class_attr_name)
                fname = os.path.join(path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name))
                if hasattr(class_attr, 'load_state_dict') and os.path.exists(fname):
                    class_attr.load_state_dict(torch.load(fname)())
                    loaded.append(class_attr_name)
        if loaded:
            print_timestamp("Loaded {} from {}".format(', '.join(loaded), path))
        else:
            raise FileNotFoundError("Nothing was loaded from {}".format(path))

def evaluate_EMD_CD(source_verts, target_verts):
    gt_points = target_verts
    pre_points = source_verts
    gt2pre, _ = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(pre_points).kneighbors(gt_points)
    pre2gt, _ = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(gt_points).kneighbors(pre_points)
    gt2pre, pre2gt = np.hstack(gt2pre), np.hstack(pre2gt)
    CD = 0.5 * (np.mean(gt2pre) + np.mean(pre2gt))
    return CD

