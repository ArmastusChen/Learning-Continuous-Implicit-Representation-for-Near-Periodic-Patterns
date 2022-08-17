import glob

import time
import scipy.ndimage as ndimage

import kornia
from tqdm import tqdm, trange
from externel_lib.robust_loss_pytorch import AdaptiveLossFunction
import matplotlib.pyplot as plt
import cv2
import externel_lib.lpips as lpips
import externel_lib.contextual_loss as cl
from skimage import morphology
from loaders.loaders import *
from models.sampler import *
from models.helpers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.random.manual_seed(0)
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


def train():
    '''
    load the parser
    '''
    import options.arg_config as option
    parser = option.config_parser().segmentation_config()
    args = parser.parse_args()

    N_rand = args.N_rand
    contextual_weight = args.contextual_weight
    perceptual_weight = args.perceptual_weight
    use_perceptual_loss = args.use_perceptual_loss
    use_contextual_loss = args.use_contextual_loss
    use_comp = args.use_comp

    '''
    Create log dir and copy the config file
    '''
    basedir = args.basedir
    expname = f'{args.expname}_top{args.p_topk}'
    name = args.datadir.split('/')[-1].replace('.png', '')
    save_path = os.path.join(basedir, expname, name)
    os.makedirs(save_path, exist_ok=True)


    '''
    initialize patch loss
    '''
    percepLoss = lpips.LPIPS(net='vgg')
    contextualLoss = cl.ContextualLoss(use_vgg=True)

    '''
    Load data
    '''
    img, period_mask, non_period_mask, blur_img, valid_mask, selected_shifts, selected_angles, selected_periods \
        = load_NPP_segmentation(args)
    print('Loaded NPP', blur_img.shape, args.datadir)
    print('selected_angles: ' + str(selected_angles))
    print('selected_periods: ' + str(selected_periods))
    print('selected_shifts: ' + str(selected_shifts))

    # image resolution
    res = (blur_img.shape[1], blur_img.shape[2])


    valid_mask_np = valid_mask.copy()
    period_mask_np = period_mask.copy().astype(np.float32)

    '''
    get pixel coordinate of training (known from initial periodic region) 
    and val (unknown from initial non-periodic region)
    '''
    train_splits = np.stack(np.nonzero(period_mask_np * valid_mask_np )[1:3], axis=1)
    val_splits = np.stack(np.nonzero((1 - period_mask_np ) * valid_mask_np )[1:3], axis=1)
    i_train, i_val = torch.Tensor(train_splits), torch.Tensor(val_splits)

    selected_angles = torch.Tensor(selected_angles)
    selected_periods = torch.Tensor(selected_periods)


    '''
    get all pixel coordinate. This is used for patch sampling 
    '''
    h, w = torch.meshgrid(torch.arange(0, res[0]), torch.arange(0, res[1]))
    i_all = torch.stack([h, w], dim=-1).reshape(-1, 2)

    '''
    Create NPP-Net model
    '''
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, embedder, embedder_periodic \
        = create_npp_net(args, selected_angles, selected_periods, res, percepLoss)

    '''
     create positional embedding
    '''
    i_train_emb_periodic = []
    i_val_emb_periodic = []
    i_all_emb_periodic = []
    global_step = start
    for i in range(args.p_topk):
        # periodic positional encoding
        i_train_emb_periodic_ = embedder_periodic[i].embed(i_train.clone())
        i_val_emb_periodic_ = embedder_periodic[i].embed(i_val.clone())
        i_all_emb_periodic_ = embedder_periodic[i].embed(i_all.clone())
        # nerf positional encoding
        i_train_emb_periodic.append(embedder.embed(i_train_emb_periodic_))
        i_val_emb_periodic.append(embedder.embed(i_val_emb_periodic_))
        i_all_emb_periodic.append(embedder.embed(i_all_emb_periodic_))

    i_train_emb_periodic = torch.cat(i_train_emb_periodic, 1)
    i_val_emb_periodic = torch.cat(i_val_emb_periodic, 1)
    i_all_emb_periodic = torch.cat(i_all_emb_periodic, 1).reshape(res[0], res[1], -1)

    print('positional embedding has been created')



    '''
    Move training data to GPU
    '''
    img = torch.Tensor(img).to(device)
    period_mask = torch.Tensor(period_mask).to(device)
    non_period_mask = torch.Tensor(non_period_mask).to(device)
    blur_img = torch.Tensor(blur_img).to(device)
    valid_mask = torch.Tensor(valid_mask).to(device)
    full_mask = (valid_mask * period_mask)
    period_mask = period_mask.permute(0, 3, 1,2 )
    non_period_mask = non_period_mask.permute(0, 3, 1,2 )



    print('Begin')
    print('TRAIN pixels are', i_train.shape)
    print('VAL pixels are', i_val.shape)

    # create the patch sampler
    N_rand = args.N_rand
    patch_size = args.patch_size
    patch_num = args.patch_num
    patch_sampler = GridPatchSampler(N_samples=patch_num, img=blur_img, mask=full_mask, patch_size=patch_size,
                                     height=img.shape[1], width=img.shape[2], pool_train=i_train.clone(),
                                     pool_val=i_val.clone(), selected_shifts=selected_shifts, no_reg_sampling=args.no_reg_sampling)

    start = start + 1
    for i in trange(start, args.N_iters):
        time0 = time.time()

        '''
        decrease the patch size every args.patch_size_decay epoch
        '''
        if i % args.patch_size_decay == 0 and (not i == 1) and patch_size > 31:
            patch_size = patch_size // 2
            patch_num = patch_num * 2
            patch_sampler.reset_patchsize(img=blur_img, mask=full_mask, N_samples=patch_num, patch_size=patch_size)
            patch_sampler.reset_pool(i_train.clone(), i_val.clone())

        '''
        sample real and fake patches from input images for patch losses. We denote fake patches as predicted patches 
        and real patch as its (fake patches') corresponding patches sampled based on periodicity.  

        Three types of patches may be sampled according to probability:
        (1) patch_source = 'val':  sample pred patches in the unknown regions, and their gt patches in known regions (based on periodicity).
        (2) patch_source = 'train':  sample pred patches in the known regions, and their gt patches in known regions (based on periodicity).
        (3) patch_source = 'same':  sample pred patch and their gt patches in same (known) regions.

        Note that all rgb patches here are sampled from input image.
        '''
        select_real_patch, select_real_patch_mask, select_fake_patch, \
        select_fake_patch_mask, select_fake_patch_coords, \
        patch_source, topk_patch_num, topk_weight = patch_sampler.sample_patches(topk=args.num_real_patch_per_sample,
                                                                                 invalid_ratio=args.invalid_ratio)
        # if none of patches are sampled
        if topk_patch_num == 0:
            continue

        # coordinates of predicted patches
        select_fake_patch_coords = select_fake_patch_coords.reshape(-1, 2)
        # predicted patches positional encoding
        select_coords_emb_patch_periodic = i_all_emb_periodic[select_fake_patch_coords[:, 0],
                                           select_fake_patch_coords[:, 1], :]


        '''
        sample pixels for pixel loss
        '''
        select_inds = np.random.choice(i_train.shape[0], size=[N_rand], replace=False)  # (N_rand,)
        select_coords = i_train[select_inds].long()  # (N_rand, 2)
        # we use blur image here as GT
        gt_rgb = blur_img[0, select_coords[:, 0], select_coords[:, 1], :]  # (N_rand, 3)
        # for pixel loss, all the gt values are available
        gt_mask = torch.ones_like(gt_rgb[:, :1])
        # sampled pixel positional encoding
        select_coords_emb_periodic = i_train_emb_periodic[select_inds]  # (N_rand, embedding)

        # concat pixel and fake patch positional encoding
        select_coords_emb_periodic = torch.cat([select_coords_emb_periodic, select_coords_emb_patch_periodic])

        '''
         run the network
         '''
        # first argument is only used for periodicity searching
        pred_rgb = render(None, select_coords_emb_periodic, args, **render_kwargs_train)

        '''
        optimization
        '''
        optimizer.zero_grad()


        # pixel loss
        loss = img2mse(pred_rgb[:len(select_inds)], gt_rgb, args.loss_type, adaptive_pix, gt_mask[:len(select_inds)])

        if args.no_pix_loss:
            loss = 0



        # pred rgb value for patch loss
        pred_patch_rgb = pred_rgb[len(select_inds):]
        pred_patch_rgb = pred_patch_rgb.reshape(patch_num, 1, patch_size, patch_size, 3).permute(0, 1, 4, 2, 3)
        pred_patch_rgb = pred_patch_rgb.tile((1, topk_patch_num, 1, 1, 1))

        # rgb value of real patches from input image
        gt_patch_rgb = select_real_patch.reshape(-1, topk_patch_num, 3)
        real_patch_rgb = gt_patch_rgb.reshape(patch_num, topk_patch_num, patch_size, patch_size, 3).permute(0, 1, 4, 2,
                                                                                                            3)

        # preparation for patch loss
        if use_perceptual_loss or use_contextual_loss:
            # mask of real patch from input image
            select_real_patch_mask = select_real_patch_mask.permute(0, 1, 4, 2, 3).reshape(-1, 1, patch_size,
                                                                                           patch_size)
            # rgb of fake patch from network output
            pred_patch_rgb = pred_patch_rgb.reshape(-1, 3, patch_size, patch_size)
            # rgb of real patch from input image
            real_patch_rgb = real_patch_rgb.reshape(-1, 3, patch_size, patch_size)
            # rgb of fake patch from input image
            select_fake_patch = select_fake_patch.reshape(-1, 3, patch_size, patch_size)
            # mask of fake patch from input image
            select_fake_patch_mask = select_fake_patch_mask.reshape(-1, 1, patch_size, patch_size)

            if not args.use_patch_weight:
                topk_weight = None

        if use_contextual_loss:
            if use_comp and patch_source == 'val':
                # copy and paste known region to the predicted patch.
                comp_patch_rgb = select_fake_patch * select_fake_patch_mask + pred_patch_rgb * (
                            1 - select_fake_patch_mask)
                contextual_loss = contextualLoss(comp_patch_rgb * select_real_patch_mask,
                                                 real_patch_rgb * select_real_patch_mask, topk_weight)
            else:
                contextual_loss = contextualLoss(pred_patch_rgb * select_real_patch_mask,
                                                 real_patch_rgb * select_real_patch_mask, topk_weight)
            loss += contextual_loss * contextual_weight

        if use_perceptual_loss:
            # only apply perceptual loss when the patch_source is same
            if patch_source == 'same':
                perc_loss = percepLoss(pred_patch_rgb * select_real_patch_mask,
                                       select_fake_patch * select_real_patch_mask,
                                       use_robust=args.use_adaptive_perceptual_loss, normalize=True)

                if topk_weight is not None:
                    perc_loss = torch.sum(perc_loss.squeeze() * topk_weight)
                else:
                    perc_loss = torch.mean(perc_loss)

                loss += perc_loss * perceptual_weight

        loss.backward()
        optimizer.step()


        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 100
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        # print(new_lrate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################
        #

        '''
        Visualization
        '''
        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, name, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('visualize the result')
            train_coords = i_train.long()  # (N_train, 2)
            val_coords = i_val.long()  # (N_val, 2)

            chunk = 20000

            with torch.no_grad():
                '''
                Visualize network output and compute mse in the known region 
                '''
                gt_rgb_train_pixs = img[0, train_coords[:, 0], train_coords[:, 1], :]
                pred_rgb_train_pixs = torch.zeros_like(gt_rgb_train_pixs)
                pred_rgb_train_img = torch.zeros_like(blur_img).cuda()


                for j in range(0, len(train_coords), chunk):
                    train_coord = train_coords[j: j + chunk]
                    train_coord_emb_periodic = i_train_emb_periodic[j: j + chunk]
                    current_chunk = len(train_coord_emb_periodic)
                    pred_rgb_train_pix = render(None, train_coord_emb_periodic, args, **render_kwargs_train)
                    pred_rgb_train_img[:, train_coord[:, 0], train_coord[:, 1], :] = pred_rgb_train_pix
                    pred_rgb_train_pixs[j: j + chunk, :] = pred_rgb_train_pix[:current_chunk, :]


                '''
                Visualize network output and compute mse in the unknown region 
                '''
                gt_rgb_val_pixs = img[0, val_coords[:, 0], val_coords[:, 1], :]
                pred_rgb_val_pixs = torch.zeros_like(gt_rgb_val_pixs)
                pred_rgb_val_img = torch.zeros_like(blur_img).cuda()
                for j in range(0, len(val_coords), chunk):
                    val_coord = val_coords[j: j + chunk]
                    val_coord_emb_periodic = i_val_emb_periodic[j: j + chunk]
                    current_chunk = len(val_coord_emb_periodic)
                    pred_rgb_val_pix = render(None, val_coord_emb_periodic, args, **render_kwargs_train)
                    pred_rgb_val_img[:, val_coord[:, 0], val_coord[:, 1], :] = pred_rgb_val_pix
                    pred_rgb_val_pixs[j: j + chunk, :] = pred_rgb_val_pix[:current_chunk, :]

                '''
                Compute the error in non-periodic region
                The goal is to convert non-periodic region into periodic region
                '''
                # prepare predicted image for error computation
                pred_rgb_img = (pred_rgb_val_img + pred_rgb_train_img) * valid_mask
                pred_rgb_img_ = (pred_rgb_img * valid_mask).permute(0, 3, 1, 2)
                pred_rgb_img_ = kornia.rgb_to_grayscale(pred_rgb_img_)

                # prepare blur image (GT) for error computation
                blur_img_ = (blur_img * valid_mask).permute(0, 3, 1, 2)
                blur_img_ = kornia.rgb_to_grayscale(blur_img_)

                '''
                Criterion 1: L1 loss
                '''
                l1_img = torch.sum(abs(pred_rgb_img_ - blur_img_), 1, keepdim=True)
                l1_img = torch.clamp(l1_img, min=0, max=0.99)
                # apply threshold (region with smaller error should be periodic region)
                l1_img_mask = l1_img < args.l1_thresh
                l1_img = l1_img * valid_mask.permute(0, 3, 1, 2)

                # visualization
                plt.imsave(f'{testsavedir}/l1_diff_img.png', l1_img.cpu()[0,0])
                plt.imsave(f'{testsavedir}/l1_img_mask.png', ~l1_img_mask.cpu()[0,0])

                '''
                Criterion 2: perceptual loss
                '''
                metric_func = lpips.LPIPS(net='alex', spatial=True, )
                lpips_img_final, lpips_img_list = metric_func(pred_rgb_img_, blur_img_, False, retPerLayer=True, normalize=True)

                # mask for final non-periodic region
                non_period_mask_final = None
                for i in range(args.lpips_layers):
                    # get the lpips image in the layer
                    lpips_img = lpips_img_list[i]

                    # only focus on error in non-periodic region
                    lpips_img_non_period = non_period_mask * lpips_img

                    # apply threshold (region with smaller error should be periodic region)
                    lpips_img_mask_i = (lpips_img_non_period < args.lpips_thresh)


                    # only region satisfies two criterion are periodic
                    period_mask_final_i = lpips_img_mask_i & l1_img_mask
                    non_period_mask_final_i = (~period_mask_final_i.cpu()[0, 0]).float().numpy()

                    if non_period_mask_final is None:
                        non_period_mask_final = non_period_mask_final_i
                    else:
                        non_period_mask_final = non_period_mask_final + non_period_mask_final_i

                    # visualization
                    plt.imsave(f'{testsavedir}/lpips_diff_img_{i}.png', lpips_img_non_period.cpu()[0, 0])
                    plt.imsave(f'{testsavedir}/lpips_img_mask_{i}.png', ~lpips_img_mask_i.cpu()[0, 0])

                # post-processing for non-periodic region
                non_period_mask_final = non_period_mask_final > 0
                non_period_mask_final = ndimage.binary_fill_holes(non_period_mask_final).astype(np.float)
                non_period_mask_final = non_period_mask_final[..., None]
                non_period_mask_final = morphology.remove_small_objects(non_period_mask_final.astype(bool), min_size=500, connectivity=1).astype(int)

                # visualization
                np_color = (0, 255, 0)
                alpha = 0.7
                rgb_img = img[0].cpu().numpy()
                valid_mask_vis = valid_mask[0].cpu().numpy()

                vis_seg_img_final = rgb_img * alpha + (1 - alpha) * (np_color * non_period_mask_final + rgb_img * (1 - non_period_mask_final))
                vis_seg_img_final = vis_seg_img_final * valid_mask_vis

                cv2.imwrite(f'{testsavedir}/segment.png', np.uint8(vis_seg_img_final[..., ::-1] * 255))

    global_step += 1



if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
