import glob
import os, sys
import time

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.chdir("..")
from arg_config import *
from tqdm import tqdm, trange
from externel_lib.robust_loss_pytorch import AdaptiveLossFunction
import externel_lib.lpips as lpips
import externel_lib.contextual_loss as cl
from models.mse_calculator import *
from models.sampler import *
from models.helpers import *
from load_NPP import load_NPP_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(2)
torch.random.manual_seed(0)


def train():
    '''
    load the parser
    '''
    parser = config_parser()
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
    os.makedirs(os.path.join(basedir, expname, name), exist_ok=True)


    '''
    initialize patch loss
    '''
    percepLoss = lpips.LPIPS(net='vgg')
    contextualLoss = cl.ContextualLoss(use_vgg=True)

    '''
    Load data
    '''
    img, mask, masked_img, valid_mask, i_split, selected_shifts, selected_angles, selected_periods = load_NPP_data(args)
    print('Loaded NPP', masked_img.shape, args.datadir)
    print('selected_angles: ' + str(selected_angles))
    print('selected_periods: ' + str(selected_periods))
    print('selected_shifts: ' + str(selected_shifts))

    # image resolution
    res = (masked_img.shape[1], masked_img.shape[2])

    '''
    get pixel coordinate of training (known) and val (unknown)
    '''
    i_train, i_val = i_split
    i_train, i_val = torch.Tensor(i_train), torch.Tensor(i_val)

    selected_angles = torch.Tensor(selected_angles)
    selected_periods = torch.Tensor(selected_periods)

    '''
    get all pixel coordinate. This is used for patch sampling 
    '''
    h, w = torch.meshgrid(torch.arange(0, res[0]), torch.arange(0, res[1]))
    i_all = torch.stack([h, w], dim=-1).reshape(-1,2)


    '''
    Create NPP-Net model
    '''
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, embedder, embedder_periodic\
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
    mask = torch.Tensor(mask).to(device)
    masked_img = torch.Tensor(masked_img).to(device)
    valid_mask = torch.Tensor(valid_mask).to(device)
    full_mask = (valid_mask * mask)


    contextual_loss = torch.Tensor([0])
    perc_loss = torch.Tensor([0])

    print('Begin')
    print('TRAIN pixels are', i_train.shape)
    print('VAL pixels are', i_val.shape)

    # create the patch sampler
    patch_size = args.patch_size
    patch_num = args.patch_num
    patch_sampler = GridPatchSampler(N_samples=patch_num, img = masked_img, mask = full_mask, patch_size=patch_size,
                                     height=res[0], width=res[1], pool_train=i_train.clone(), pool_val=i_val.clone(),
                                     selected_shifts=selected_shifts, no_reg_sampling=args.no_reg_sampling)

    start = start + 1
    for i in trange(start, args.N_iters):
        '''
        decrease the patch size every args.patch_size_decay epoch
        '''
        if i % args.patch_size_decay == 0 and (not i == 1) and patch_size > 31:
            patch_size = patch_size // 2
            patch_num = patch_num * 2
            patch_sampler.reset_patchsize(img = masked_img, mask = full_mask, N_samples=patch_num, patch_size=patch_size)
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
        select_fake_patch_coords = select_fake_patch_coords.reshape(-1,2)
        # predicted patches positional encoding
        select_coords_emb_patch_periodic = i_all_emb_periodic[select_fake_patch_coords[:, 0], select_fake_patch_coords[:, 1], :]



        '''
        sample pixels for pixel loss
        '''
        select_inds = np.random.choice(i_train.shape[0], size=[N_rand], replace=False)  # (N_rand,)
        select_coords = i_train[select_inds].long() # (N_rand, 2)
        gt_rgb = masked_img[0, select_coords[:, 0], select_coords[:, 1], :]  # (N_rand, 3)
        # for pixel loss, all the gt values are available
        gt_mask = torch.ones_like(gt_rgb[:, :1])
        # sampled pixel positional encoding
        select_coords_emb_periodic = i_train_emb_periodic[select_inds] # (N_rand, embedding)

        # concat pixel and fake patch positional encoding
        select_coords_emb_periodic = torch.cat([select_coords_emb_periodic, select_coords_emb_patch_periodic])


        '''
        run the network
        '''
        pred_rgb = render(select_coords_emb_periodic, args, **render_kwargs_train)


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
        pred_patch_rgb = pred_patch_rgb.tile((1, topk_patch_num, 1,1, 1))

        # rgb value of real patches from input image
        gt_patch_rgb = select_real_patch.reshape(-1, topk_patch_num, 3)
        real_patch_rgb = gt_patch_rgb.reshape(patch_num, topk_patch_num, patch_size, patch_size, 3).permute(0, 1, 4, 2, 3)


        # preparation for patch loss
        if use_perceptual_loss or use_contextual_loss:
            # mask of real patch from input image
            select_real_patch_mask = select_real_patch_mask.permute(0, 1, 4, 2, 3).reshape(-1, 1, patch_size, patch_size)
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
                comp_patch_rgb = select_fake_patch * select_fake_patch_mask + pred_patch_rgb * (1-select_fake_patch_mask)
                contextual_loss = contextualLoss(comp_patch_rgb * select_real_patch_mask, real_patch_rgb * select_real_patch_mask, topk_weight)
            else:
                contextual_loss = contextualLoss(pred_patch_rgb * select_real_patch_mask, real_patch_rgb * select_real_patch_mask, topk_weight)
            loss += contextual_loss * contextual_weight

        if use_perceptual_loss:
            # only apply perceptual loss when the patch_source is same
            if  patch_source == 'same':
                perc_loss = percepLoss(pred_patch_rgb * select_real_patch_mask, select_fake_patch * select_real_patch_mask, use_robust=args.use_adaptive_perceptual_loss, normalize=True)


                if topk_weight is not None:
                    perc_loss = torch.sum(perc_loss.squeeze() * topk_weight)
                else:
                    perc_loss = torch.mean(perc_loss)

                loss += perc_loss * perceptual_weight


        # print(loss)
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
        if i % args.i_testset==0 and i > 0:
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
                gt_rgb_train_pixs = masked_img[0, train_coords[:, 0], train_coords[:, 1], :]
                pred_rgb_train_pixs = torch.zeros_like(gt_rgb_train_pixs)
                pred_rgb_train_img = torch.zeros_like(masked_img).cuda()

                for j in range(0, len(train_coords), chunk):
                    train_coord = train_coords[j: j+chunk]
                    train_coord_emb_periodic = i_train_emb_periodic[j: j+chunk]
                    current_chunk = len(train_coord_emb_periodic)
                    pred_rgb_train_pix = render( train_coord_emb_periodic, args, **render_kwargs_train)
                    pred_rgb_train_img[:, train_coord[:, 0], train_coord[:, 1], :] = pred_rgb_train_pix
                    pred_rgb_train_pixs[j: j+chunk, :] = pred_rgb_train_pix[:current_chunk, :]
                img_train_loss = img2mse(pred_rgb_train_pixs, gt_rgb_train_pixs,args.loss_type, adaptive_pix, )

                '''
                Visualize network output and compute mse in the unknown region 
                '''
                gt_rgb_val_pixs = img[0, val_coords[:, 0], val_coords[:, 1], :]
                pred_rgb_val_pixs = torch.zeros_like(gt_rgb_val_pixs)
                pred_rgb_val_img = torch.zeros_like(masked_img).cuda()
                for j in range(0, len(val_coords), chunk):
                    val_coord = val_coords[j: j+chunk]
                    val_coord_emb_periodic = i_val_emb_periodic[j: j+chunk]
                    current_chunk = len(val_coord_emb_periodic)
                    pred_rgb_val_pix = render(val_coord_emb_periodic, args, **render_kwargs_train)
                    pred_rgb_val_img[:, val_coord[:, 0], val_coord[:, 1], :] = pred_rgb_val_pix
                    pred_rgb_val_pixs[j: j+chunk, :] = pred_rgb_val_pix[:current_chunk, :]
                img_val_loss = img2mse(pred_rgb_val_pixs, gt_rgb_val_pixs,args.loss_type, adaptive_pix,)

                '''
                compose the final image for visualization
                '''
                pred_rgb_train_img = pred_rgb_train_img * valid_mask
                pred_rgb_val_img = pred_rgb_val_img * valid_mask
                img = img * valid_mask
                masked_img = masked_img * valid_mask


                plt.imsave(f'{testsavedir}/pred_rgb_train_img.png', pred_rgb_train_img[0].cpu().numpy())
                plt.imsave(f'{testsavedir}/pred_rgb_val_img.png', pred_rgb_val_img[0].cpu().numpy())
                plt.imsave(f'{testsavedir}/gt_rgb_img.png', img[0, ..., :3].cpu().numpy())
                plt.imsave(f'{testsavedir}/input_rgb_img.png', masked_img[0, ..., :3].cpu().numpy())

                plt.imsave(f'{testsavedir}/pred_rgb_img.png',
                           pred_rgb_val_img[0].cpu().numpy() + pred_rgb_train_img[0].cpu().numpy())

                plt.imsave(f'{testsavedir}/pred_rgb_img_comp.png',
                           pred_rgb_val_img[0].cpu().numpy() + masked_img[0].cpu().numpy() * mask[0].cpu().numpy())

                print(f'img_train_loss: {img_train_loss}')
                print(f'img_val_loss: {img_val_loss}')

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()} Contextual Loss: {contextual_loss.item() * contextual_weight} Percep Loss {perc_loss.item() * perceptual_weight} ")

        global_step += 1








if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
