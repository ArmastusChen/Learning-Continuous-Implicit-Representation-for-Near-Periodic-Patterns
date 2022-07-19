import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from utils.extract_glimpse import *




class GridPatchSampler():
    def __init__(self, img,  mask, N_samples, patch_size, height, width, pool_train, pool_val, selected_shifts, no_reg_sampling):
        '''
        Args:
            img: input image (with unknown regions)
            mask: unknown image
            N_samples: number of fake patch to sample
            patch_sizes: the resolution of the sampled patch
            height: image height
            width: image width
            pool_train:  pixel coordinates in training set (known regions).
            pool_val:  pixel coordinates in val set (unknown regions).
            selected_shifts: periodicity, represented as $d$ in the paper.

            NOTE: We also use real and fake to denote train (known) and val (unknown) regions, respectively.
        '''
        super(GridPatchSampler, self).__init__()

        self.N_samples = int(N_samples)
        self.coord_patches = None
        self.height, self.width = height, width
        self.img, self.mask = img.permute(0, 3, 1,2 ) , mask.permute(0, 3, 1,2 )

        # only use top-1 periodicity for sampling
        selected_shifts = selected_shifts[0]

        # the first coordinate should be along the vertical (height) direction
        self.selected_shifts = [torch.tensor([selected_shift[1], selected_shift[0]]) for selected_shift in selected_shifts]

        # if not using periodicity aware sampling (random sample strategy)
        self.no_reg_sampling = no_reg_sampling

        h, w = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        self.coords_img = torch.stack([h, w], dim=-1)[None]

        # initialize real (known) patches that can be sampled
        self.reset_patchsize(img, mask, patch_size, N_samples)

        # initialize fake patches that can be sampled
        self.reset_pool(pool_train, pool_val)

    def reset_patchsize(self, img, mask, patch_size, N_samples, ratio=0.0):
        '''
        initialize real (known) patches that can be sampled

        Args:
            img: input image (with unknown regions)
            mask: unknown image
            N_samples: number of fake patch to sample
            patch_size: the resolution of the sampled patch
            ratio:  thresh for invalid patch filtering
        '''

        if self.coord_patches is not None:
            del self.mask_patches, self.img_patches, self.coord_patches
        self.N_samples = N_samples
        self.patch_size_h_half, self.patch_size_w_half = patch_size // 2, patch_size // 2

        size = patch_size  # patch size
        stride = patch_size // 10  # patch stride

        # create all potential patches for known patches sampling
        mask_patches = mask.unfold(1, size, stride).unfold(2, size, stride)
        img_patches = img.unfold(1, size, stride).unfold(2, size, stride)
        coord_patches = self.coords_img.unfold(1, size, stride).unfold(2, size, stride)


        mask_patches = mask_patches.reshape(img.shape[0], -1, 1, patch_size, patch_size)[0]
        img_patches = img_patches.reshape(img.shape[0], -1, 3, patch_size, patch_size)[0]
        coord_patches = coord_patches.reshape(img.shape[0], -1, 2, patch_size, patch_size)[0]

        # filter patches that contains unknown regions large than   patch_size ** 2 * ratio
        invalid_mask_ind = (torch.sum((mask_patches < 0.5), dim=[1,2,3]) > (patch_size ** 2 * ratio))
        valid_mask_ind = ~invalid_mask_ind
        self.mask_patches = mask_patches[valid_mask_ind]
        self.img_patches = img_patches[valid_mask_ind]
        self.coord_patches = coord_patches[valid_mask_ind]
        # get the coordinates of centroid for patches
        self.real_coord_centroids = self.coord_patches[:, :, self.patch_size_h_half, self.patch_size_w_half][None].tile((self.N_samples,1,1))

        # limit the range of real patches that can be sampled for a specific fake patch.
        self.max_shifting_ind = 10
        self.permutation1, self.permutation2 = torch.meshgrid(torch.arange(-self.max_shifting_ind, self.max_shifting_ind),
                                                              torch.arange(-self.max_shifting_ind, self.max_shifting_ind))
        self.permute_distance = (abs(self.permutation1) + abs(self.permutation2)).reshape(-1).tile(self.N_samples)
        self.permutation1 = self.permutation1[..., None, None, None]
        self.permutation2 = self.permutation2[..., None, None, None]

        # batch index indication after reshape operations
        self.coord_batch_indicator = torch.ones((self.N_samples,
                                                 self.max_shifting_ind * 2,
                                                 self.max_shifting_ind * 2)) * torch.arange(self.N_samples)[:, None, None]
        self.coord_batch_indicator = self.coord_batch_indicator.reshape(-1)

    def reset_pool(self, pool_train, pool_val):
        '''
        initialize fake patches that can be sampled

        Args:
          pool_train:  pixel coordinates in training set (known regions).
            pool_val:  pixel coordinates in test set (unknown regions).
        '''

        def _get_valid_centroid(pool):
            '''
            get the pixel coordinates of patch centroid, whose corresponding patch does not exceed image boundary
            '''
            valid1 = pool[:, 0] > self.patch_size_h_half
            valid2 = pool[:, 0] < self.height - (self.patch_size_h_half + 1)
            valid3 = pool[:, 1] > self.patch_size_w_half
            valid4 = pool[:, 1] < self.width - (self.patch_size_w_half + 1)

            valid = valid1 & valid2 & valid3 & valid4
            return pool[valid]

        self.pool_train = _get_valid_centroid(pool_train)
        self.pool_val = _get_valid_centroid(pool_val)


    def sample_patch_real(self, fake_coords=None, topk = 5, invalid_ratio=0.3):
        '''
        Sample real (known) patches

        Args:
            fake_coords:  pixel coordinates value of sampled fake patches.
            topk:  number of real (known or gt) patches that are sampled per fake (predicted) patch
            invalid_ratio: threshold to filter invalid real patches.

        Return:
            select_img_patches: rgb value of sampled real patches
            select_mask_patches: mask value of sampled real patches
            weight_topks: the weight of sampled real patches
            topk_min: number of real patches that are sampled
        '''

        # periodicity-guided patch sampling
        if fake_coords is not None and not self.no_reg_sampling:
            # get centroid of fake patches
            fake_coords_centroids =  fake_coords[:, self.patch_size_h_half, self.patch_size_w_half, :][:, None, None, :, None, None ]

            # get all the available centroids of real patches for sampling
            selected_shift1, selected_shift2 = self.selected_shifts
            selected_shift1 = selected_shift1[None, None, :, None, None]
            selected_shift2 = selected_shift2[None, None, :, None, None]
            total_shift = selected_shift1 * self.permutation1 + selected_shift2 * self.permutation2
            real_coords_centroids_pool = (fake_coords_centroids + total_shift)[..., 0,0]

            # filter centroids with their patches exceed the image boundary
            valid1 = real_coords_centroids_pool[..., 0] > 0
            valid2 = real_coords_centroids_pool[..., 0] < self.height - 1
            valid3 = real_coords_centroids_pool[..., 1] > 0
            valid4 = real_coords_centroids_pool[..., 1] < self.width - 1
            valid_in_bound = valid1 & valid2 & valid3 & valid4
            real_coords_centroids_pool = real_coords_centroids_pool.reshape(fake_coords.shape[0], -1, 2).reshape(-1, 2)
            valid_in_bound = valid_in_bound.reshape(fake_coords.shape[0], -1, 1).reshape(-1)
            real_coords_centroids_pool = real_coords_centroids_pool[valid_in_bound]
            coord_batch_indicator = self.coord_batch_indicator[valid_in_bound]

            # get the distance of all the candidate real patches
            permute_distance = self.permute_distance[valid_in_bound]

            # sample candidate real rgb and mask patches
            chunk = coord_batch_indicator.shape[0]
            real_mask = extract_glimpse(input=self.mask.tile([chunk, 1,1,1]), size=(self.patch_size_h_half * 2, self.patch_size_w_half * 2),
                                        offsets=real_coords_centroids_pool.flip((1)), padding_mode='zeros', mode='nearest',
                                        normalized=False, centered=False)

            real_img = extract_glimpse(input=self.img.tile([chunk, 1, 1, 1]),
                                   size=(self.patch_size_h_half * 2, self.patch_size_w_half * 2),
                                   offsets=real_coords_centroids_pool.flip((1)), padding_mode='zeros',  mode='nearest',
                                   normalized=False, centered=False)

            # filter real patches with the ratio of unknown regions (to full patches) larger than patch_size ** 2 * invalid_ratio
            invalid_mask_ind = (torch.sum((real_mask < 0.5), dim=[1, 2, 3]) > (self.patch_size_h_half * self.patch_size_w_half * 4 * invalid_ratio))
            valid_mask_ind = ~invalid_mask_ind
            real_mask = real_mask[valid_mask_ind]
            real_img = real_img[valid_mask_ind]
            permute_distance = permute_distance[valid_mask_ind]
            coord_batch_indicator = coord_batch_indicator[valid_mask_ind]

            select_img_patches = []
            select_mask_patches = []
            weight_topks = []

            # sample real patches based on distance (smaller distance is preferred to preserve local variations).
            topk_min = topk
            for i in range(self.N_samples):
                inds = (coord_batch_indicator > (i - 0.5)) & (coord_batch_indicator < (i + 0.5))
                distance = permute_distance[inds]
                distance[distance == 0] = 10000
                # exclude itself
                if min(len(distance)-1, topk) < topk_min:
                    topk_min = min(len(distance)-1, topk)
                    if topk_min <= 0:
                        return  None, None, None, 0
                distance_topk, inds_topk = torch.topk(distance, k = topk_min, largest=False)
                distance_topk = 1 / distance_topk
                # self patch included when you see nan in the runtime
                weight_topk =  distance_topk / torch.sum(distance_topk)

                select_img_patches.append(real_img[inds][inds_topk])
                select_mask_patches.append(real_mask[inds][inds_topk])
                weight_topks.append(weight_topk)

            # if the number of sampled patch we sampled is less than the number of patch we want to sample
            if topk_min < topk:
                for i in range(len(weight_topks)):
                    weight_topks[i] = weight_topks[i][:topk_min]
                    select_img_patches[i] = select_img_patches[i][:topk_min]
                    select_mask_patches[i] = select_mask_patches[i][:topk_min]

            weight_topks = torch.cat(weight_topks)
            select_img_patches = torch.stack(select_img_patches)
            select_mask_patches = torch.stack(select_mask_patches)
        # random patch sampling strategy
        else:
            select_inds = np.random.choice(self.img_patches.shape[0], size=[self.N_samples * topk], replace=False)  # (N_rand,)

            select_img_patches = self.img_patches[select_inds]
            select_mask_patches = self.mask_patches[select_inds]
            select_img_patches = select_img_patches.reshape(self.N_samples, topk, 3, select_img_patches.shape[2], select_img_patches.shape[3])
            select_mask_patches = select_mask_patches.reshape(self.N_samples, topk, 1, select_mask_patches.shape[2], select_mask_patches.shape[3])
            weight_topks, topk_min = None, topk



        select_img_patches = select_img_patches.permute(0, 1,3,4,2)  # (N_rand, 3)
        select_mask_patches = select_mask_patches.permute(0, 1,3,4,2)   # (N_rand, 1)

        return select_img_patches, select_mask_patches, weight_topks, topk_min




    def sample_patch_fake(self, mode):
        '''
        Randomly Sample fake patches

        Args:
            mode: sampled from training set or validation set

        Return:
            select_patch: rgb value of sampled fake patches
            select_patch_mask: mask value of sampled fake patches
            select_patch_grids: pixel coordinates value of sampled fake patches
        '''
        if mode == 'train':
            pool = self.pool_train
        elif mode == 'val':
            pool = self.pool_val

        # sample centroid
        select_inds = np.random.choice(pool.shape[0], size=[self.N_samples], replace=False)  # (N_rand,)
        select_centroid = pool[select_inds]
        left_hs = select_centroid[:,0] - self.patch_size_h_half
        right_hs =  select_centroid[:,0] + self.patch_size_h_half
        left_ws = select_centroid[:, 1] - self.patch_size_w_half
        right_ws = select_centroid[:, 1] + self.patch_size_w_half

        # sample pixel coordinates of fake patches
        select_patch_grids = []
        for i in range(len(left_hs)):
            left_h = left_hs[i]
            right_h = right_hs[i]
            left_w = left_ws[i]
            right_w = right_ws[i]

            h, w = torch.meshgrid(torch.arange(int(left_h), int(right_h)), torch.arange(int(left_w), int(right_w)))

            select_patch_grid = torch.stack([h, w], dim=-1)

            select_patch_grids.append(select_patch_grid)
        select_patch_grids = torch.stack(select_patch_grids)

        # sample rgb values and mask values of fake patches
        chunk = self.N_samples
        select_patch = extract_glimpse(input=self.img.tile([chunk, 1, 1, 1]),
                                   size=(self.patch_size_h_half * 2, self.patch_size_w_half * 2),
                                   offsets=select_centroid.flip((1)), padding_mode='zeros', mode='nearest',
                                   normalized=False, centered=False)
        select_patch_mask = extract_glimpse(input=self.mask.tile([chunk, 1, 1, 1]),
                                   size=(self.patch_size_h_half * 2, self.patch_size_w_half * 2),
                                   offsets=select_centroid.flip((1)), padding_mode='zeros', mode='nearest',
                                   normalized=False, centered=False)

        return select_patch, select_patch_mask, select_patch_grids



    def sample_patches(self, topk, invalid_ratio):
        '''
        Sample GT real and fake patches.

        Please note, all the sampled rgb patches (including fake and real) are from input image (with mask).
        This will serve as GT patch for patch loss. That being said, the sampled fake rgb patches are NOT from
        network output image!!!


        Args:
            topk: number of real (known) patches that are sampled for a specific fake patch
            invalid_ratio: threshold to filter invalid real patches.

        Returns:
            select_real_patch: sampled real RGB patches from input image.
            select_real_patch_mask: sampled real mask patches from input image.
            select_fake_patch: sampled fake RGB patches from input image.
            select_fake_patch_mask: sampled fake mask patches from input image.
            select_fake_patch_coords: sampled fake coordinate patches.
            patch_source:
                    (1) 'val':  sample pred patches in the unknown regions, and their gt patches in known regions (based on periodicity).
                    (2) 'train':  sample pred patches in the known regions, and their gt patches in known regions (based on periodicity).
                    (3) 'same':  sample pred patch and their gt patches in same (known) regions.
            topk: number of real patches that are sampled per fake patch in the runtime.
            weight_topk: weight of sampled real patches.
        '''
        # select patch from validation / train set
        prob = np.random.uniform(0, 1)

        # sample pred patches in the unknown regions, and their gt patches in known regions (based on periodicity).
        if prob < 0.5:
            patch_source = 'val'
            select_fake_patch, select_fake_patch_mask, select_fake_patch_coords = self.sample_patch_fake('val')
            select_real_patch, select_real_patch_mask, weight_topk, topk = self.sample_patch_real(select_fake_patch_coords, topk=topk, invalid_ratio=invalid_ratio)

        # sample pred patches in the known regions, and their gt patches in known regions (based on periodicity).
        elif prob > 0.5 and prob <0.8:
            patch_source = 'train'
            select_fake_patch, select_fake_patch_mask, select_fake_patch_coords =  self.sample_patch_fake('train')
            select_real_patch, select_real_patch_mask, weight_topk, topk = self.sample_patch_real(select_fake_patch_coords, topk=topk, invalid_ratio=invalid_ratio)
        # sample pred patch and their gt patches in same (known) regions.
        else:
            select_fake_patch, select_fake_patch_mask, select_fake_patch_coords = self.sample_patch_fake('train')
            # real patches are at the same location as the sampled fake patches
            select_real_patch, select_real_patch_mask = select_fake_patch.clone().permute(0, 2, 3, 1), select_fake_patch_mask.clone().permute(0, 2, 3, 1)
            select_real_patch, select_real_patch_mask = select_real_patch[:, None], select_real_patch_mask[:, None]

            patch_source = 'same'
            topk = 1
            weight_topk = torch.FloatTensor([1.]).tile(self.N_samples).cuda()

        if topk == 0:
            return None, None, None, None, None, None, topk, None

        if select_fake_patch_mask is not None:
            select_fake_patch, select_fake_patch_mask = select_fake_patch[:, None].tile([1, topk, 1,1,1]), select_fake_patch_mask[:, None].tile([1, topk, 1,1,1])

        return select_real_patch, select_real_patch_mask, select_fake_patch, select_fake_patch_mask, select_fake_patch_coords, patch_source, topk, weight_topk






class FlexGridPatchSampler():
    def __init__(self, N_samples,img , mask, patch_size, height, width):
        self.N_samples = int(N_samples)
        super(FlexGridPatchSampler, self).__init__()

        self.patch_size_h_half, self.patch_size_w_half = patch_size // 2, patch_size // 2

        self.height, self.width = height, width
        # nn.functional.grid_sample grid value range in [-1,1]
        # w, h = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        # w = (w / width - 0.5) * 2
        # h = (h / height - 0.5) * 2
        # self.coords_img = torch.stack([h, w], dim=-1)

    def reset_patchsize(self, img, mask, patch_size, patch_num):
        self.patch_size_h_half, self.patch_size_w_half = patch_size // 2, patch_size // 2
        self.N_samples = patch_num


    def sample_patch(self, pool, img, mask):
        img = img.permute(0, 3, 1,2)
        mask = mask.permute(0, 3, 1,2)

        select_inds = np.random.choice(pool.shape[0], size=[self.N_samples], replace=False)  # (N_rand,)
        select_centroid = pool[select_inds]

        left_hs = select_centroid[:,0] - self.patch_size_h_half
        right_hs =  select_centroid[:,0] + self.patch_size_h_half
        left_ws = select_centroid[:, 1] - self.patch_size_w_half
        right_ws = select_centroid[:, 1] + self.patch_size_w_half

        select_patch_grids = []
        select_patch_grid_norms = []
        for i in range(len(left_hs)):
            left_h = left_hs[i]
            right_h = right_hs[i]
            left_w = left_ws[i]
            right_w = right_ws[i]

            h, w = torch.meshgrid(torch.arange(int(left_h), int(right_h)), torch.arange(int(left_w), int(right_w)))

            select_patch_grid = torch.stack([h, w], dim=-1)

            select_patch_grids.append(select_patch_grid)

            w = (w / self.width - 0.5) * 2
            h = (h / self.height - 0.5) * 2
            select_patch_grid_norm = torch.stack([w, h], dim=-1)
            select_patch_grid_norms.append(select_patch_grid_norm)


        select_patch_grid_norm = torch.stack(select_patch_grid_norms)
        select_patch_grids = torch.stack(select_patch_grids)

        select_patch = F.grid_sample(img.repeat(self.N_samples, 1,1,1), select_patch_grid_norm, mode='nearest' )
        select_patch_mask = F.grid_sample(mask.repeat(self.N_samples, 1,1,1), select_patch_grid_norm, mode='nearest' )

        # img[:, :, left_h:right_h, left_w:right_w] = 1
        # plt.imshow(img.cpu()[0].permute(1,2,0))
        # plt.show()
        #
        # # plt.imshow(mask[0,0].cpu())
        # # plt.show()
        # plt.imshow(select_patch.cpu()[0].permute(1,2,0))
        # plt.show()
        #
        # plt.imshow(select_patch_mask[0,0].cpu())
        # plt.show()

        # w, h = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        # w = (w / width - 0.5) * 2
        # h = (h / height - 0.5) * 2
        # self.coords_img = torch.stack([h, w], dim=-1)
            #
            # self.coords_img[select_centroid[:,0]-self.patch_size_h_half: select_centroid[:,0]+self.patch_size_h_half,
            #                             : ]
        return select_patch, select_patch_mask, select_patch_grids


