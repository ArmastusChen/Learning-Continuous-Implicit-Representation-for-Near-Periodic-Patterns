import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.chdir("..")
#
# print("Current working directory: {0}".format(os.getcwd()))
import time
from tqdm import trange
from utils.miscs import mask2ltrb
from externel_lib.robust_loss_pytorch import AdaptiveLossFunction
import externel_lib.lpips as lpips
import externel_lib.contextual_loss as cl
from models.mse_calculator import *
from loaders.loaders import *
from utils.periodicity_visualizer import GridProgram
from models.networks import *
from models.helpers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0)
torch.random.manual_seed(0)



def search():
    '''
    load the parser
    '''
    import options.arg_config as option
    parser = option.config_parser().searching_config()
    args = parser.parse_args()

    '''
    Create log dir
    '''
    name = args.datadir.split('/')[-1]
    file_dir = f'{args.outdir}/{name}'

    if os.path.exists(file_dir):
        print('Searching: file exists, exit!!')
        exit()

    '''
    initialize patch loss (ONLY for evaluation, not for training)
    '''
    percepLoss = lpips.LPIPS(net='vgg').cuda()
    contextualLoss = cl.ContextualLoss(use_vgg=True).cuda()

    os.makedirs(file_dir, exist_ok=True)


    '''
    Load data
    '''
    img, mask, unknown_mask, masked_img, valid_mask, i_split, all_selected_shifts, all_selected_angles, all_selected_periods = load_NPP_proposal(args)
    print('Loaded texture', masked_img.shape, args.datadir)
    print('selected_angles: ' + str(all_selected_angles))
    print('selected_periods: ' + str(all_selected_periods))
    print('selected_shifts: ' + str(all_selected_shifts))

    '''
    get pixel coordinate of training (known) and val (unknown)
    '''
    i_train, i_val = i_split
    i_train, i_val = torch.Tensor(i_train), torch.Tensor(i_val)

    '''
    Move training data to GPU
    '''
    img = torch.Tensor(img).to(device)
    masked_img = torch.Tensor(masked_img).to(device)
    unknown_mask = torch.Tensor(unknown_mask).to(device)
    valid_mask = torch.Tensor(valid_mask).to(device)

    distances = []
    best_selected_angles = []
    best_selected_periods = []
    best_selected_shifts = []
    '''
    Iteration for selected shifts for ranking 
    '''
    for search_id in range(len(all_selected_angles)):
        selected_angles = torch.Tensor(all_selected_angles[search_id])
        selected_periods = torch.Tensor(all_selected_periods[search_id])
        selected_shifts = all_selected_shifts[search_id]

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)

        '''
        Create NPP-Net model
        '''
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, embedder, embedder_periodic = \
            create_npp_net(args, selected_angles, selected_periods, masked_img.shape[1:3], percep_net=None, is_search=True)

        global_step = start

        '''
         create positional embedding
        '''
        i_train_emb = embedder.embed(i_train.clone())
        i_val_emb = embedder.embed(i_val.clone())

        i_train_emb_periodic = embedder_periodic.embed(i_train)
        i_val_emb_periodic = embedder_periodic.embed(i_val)

        N_iters = args.N_iters + 1
        start = start + 1
        for i in trange(start, N_iters):
            '''
               sample pixels for pixel loss
            '''
            select_inds = np.random.choice(i_train.shape[0], size=[args.N_rand], replace=False)  # (N_rand,)
            select_coords = i_train[select_inds].long() # (N_rand, 2)
            gt_rgb = masked_img[0, select_coords[:, 0], select_coords[:, 1], :]  # (N_rand, 3)

            # sampled pixel positional encoding
            select_coords_emb = i_train_emb[select_inds] # (N_rand, embedding)
            select_coords_emb_periodic = i_train_emb_periodic[select_inds] # (N_rand, embedding)

            '''
                run the network
            '''
            # first augment is only used for periodicity searching、
            pred_rgb = render(select_coords_emb, select_coords_emb_periodic, args, **render_kwargs_train)

            '''
                optimization
            '''
            optimizer.zero_grad()

            # pixel loss
            loss = img2mse(pred_rgb, gt_rgb, args.loss_type, adaptive_pix, None)
            loss.backward()
            optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 100
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

            '''
            Evaluate this periodicity
            '''
            if i == args.N_iters:
                val_coords = i_val.long()  # (N_val, 2)

                chunk = 20000

                with torch.no_grad():
                    '''
                    Visualize network output pseudo mask region 
                    '''
                    gt_rgb_val_pixs = masked_img[0, val_coords[:, 0], val_coords[:, 1], :]

                    pred_rgb_val_img = torch.zeros_like(masked_img).cuda()
                    gt_rgb_val_img = torch.zeros_like(masked_img).cuda()
                    for i in range(0, len(val_coords), chunk):
                        val_coord = val_coords[i: i+chunk]
                        val_coord_emb = i_val_emb[i: i+chunk]
                        val_coord_emb_periodic = i_val_emb_periodic[i: i+chunk]
                        gt_rgb_val_pix = gt_rgb_val_pixs[i: i+chunk]

                        pred_rgb_val_pix = render(val_coord_emb, val_coord_emb_periodic, args, **render_kwargs_train)

                        pred_rgb_val_img[:, val_coord[:, 0], val_coord[:, 1], :] = pred_rgb_val_pix
                        gt_rgb_val_img[:, val_coord[:, 0], val_coord[:, 1], :] = gt_rgb_val_pix


                    pred_rgb_val_img_ = pred_rgb_val_img.permute(0, 3, 1,2)
                    gt_rgb_val_img_ = gt_rgb_val_img.permute(0, 3, 1,2)

                    '''
                        Crop ROI for patch loss evaluation
                    '''
                    h_min = torch.min(val_coords[:, 0])
                    h_max = torch.max(val_coords[:, 0])
                    w_min = torch.min(val_coords[:, 1])
                    w_max = torch.max(val_coords[:, 1])

                    gt_rgb_val_img_ = gt_rgb_val_img_[:, :, h_min:h_max, w_min:w_max]
                    pred_rgb_val_img_ = pred_rgb_val_img_[:, :, h_min:h_max, w_min:w_max]

                    # we use perceptual loss and contextual loss
                    val_percep = percepLoss(pred_rgb_val_img_, gt_rgb_val_img_, False)
                    val_context = contextualLoss(pred_rgb_val_img_, gt_rgb_val_img_)

                # compute the loss as evaluation of this periodicity
                distance = val_percep * args.perceptual_weight + val_context * args.contextual_weight

                best_selected_angles.append(selected_angles)
                best_selected_periods.append(selected_periods)
                best_selected_shifts.append(selected_shifts)

                distances.append(distance)
                print(f'Completed  {search_id} / {len(all_selected_periods)}, loss: {distance}')
                break

            global_step += 1

    distances = torch.cat(distances, 0)[:, 0, 0, 0]

    # in case K is larger than the number of the candidate periodicity
    args.topk_detection = min(args.topk_detection, len(distances))

    # obtain the top-K periodicity with smallest distance (loss).
    distances_sorted, distance_inds = torch.topk(distances, k = args.topk_detection, largest=False)

    best_selected_shifts = [[best_selected_shifts[idx][i].cpu().tolist() for i in range(2)] for idx in distance_inds]
    best_selected_periods = [best_selected_periods[idx].cpu().tolist() for idx in distance_inds]
    best_selected_angles = [best_selected_angles[idx].cpu().tolist() for idx in distance_inds]

    # initialize paths
    fpath_masked_img =  f'{file_dir}/masked_img.png'
    fpath_valid_mask =  f'{file_dir}/valid_mask.png'
    fpath_unknown_mask =  f'{file_dir}/unknown_mask.png'
    fpath_gt_img =  f'{file_dir}/gt_img.png'

    # prepare json file
    odgt = {
        'fpath_masked_img': fpath_masked_img,
        'fpath_valid_mask': fpath_valid_mask,
        'fpath_mask': fpath_unknown_mask,
        'fpath_gt_img': fpath_gt_img,
        'selected_angles':best_selected_angles,
        'selected_periods': best_selected_periods,
        'selected_shifts': best_selected_shifts,
        'search_range': args.search_range,
        'epoch': args.N_iters,
        'distances': distances_sorted.cpu().tolist()
    }

    valid_mask_ori = valid_mask.clone()

    valid_mask = np.uint8(valid_mask[0, ..., 0].cpu().numpy() * 255)
    unknown_mask = np.uint8(unknown_mask[..., 0].cpu().numpy() * 255)
    masked_img = np.uint8(masked_img[0].clone().cpu().numpy() * 255)
    gt_img = np.uint8(img[0].clone().cpu().numpy() * 255)


    '''
    Visualize the ranked periodicity
    '''
    ltrb = mask2ltrb(valid_mask_ori[0, ..., 0])
    for i in range(args.topk_detection):
        fpath_reg_img = f'{file_dir}/reg_img_{i}.png'

        odgt[f'fpath_reg_img_{i}'] = fpath_reg_img,

        # create visualization object based on the periodicity (represented as displacement vectors)
        reg_vis = GridProgram(resolution=masked_img.shape[:2],
                           base_point=ltrb[:2],
                           first_shift=torch.Tensor(best_selected_shifts[i][0] ),
                           second_shift=torch.Tensor(best_selected_shifts[i][1] ),
                           loss=torch.tensor(0))

        # draw 2D lattice on the image as visualization
        reg_img, _ = reg_vis.draw(masked_img, color=(255, 255, 0))

        # save drawn periodicity image
        cv2.imwrite(fpath_reg_img, reg_img[..., ::-1] )

    # save images
    cv2.imwrite(fpath_valid_mask, valid_mask)
    cv2.imwrite(fpath_unknown_mask, unknown_mask)
    cv2.imwrite(fpath_masked_img, masked_img[..., ::-1] )
    cv2.imwrite(fpath_gt_img, gt_img[..., ::-1] )


    with open(f'{file_dir}/config.odgt', 'w') as outfile:
            json.dump(odgt, outfile)
            outfile.write('\n')


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    search()
