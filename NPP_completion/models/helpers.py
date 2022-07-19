
from externel_lib.robust_loss_pytorch import AdaptiveLossFunction
from .mse_calculator import *
from .embedder import *
from .networks import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adaptive_pix = AdaptiveLossFunction(
    num_dims=3, float_dtype=np.float32, device=0)




def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs_periodic):
        return torch.cat([fn(inputs_periodic[i:i+chunk]) for i in range(0, inputs_periodic.shape[0], chunk)], 0)
    return ret


def run_network(inputs_periodic, fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """

    outputs_flat = batchify(fn, netchunk)(inputs_periodic)


    outputs = torch.reshape(outputs_flat, list(inputs_periodic.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs




def render(select_coords_emb_periodic, args, network_query_fn, network_fn):
    """Render rays
    Args:

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    # Render and reshape
    # select_coords = (select_coords / 256-0.5) * 2
    raw = network_query_fn(select_coords_emb_periodic, network_fn)

    if args.normalize_type == 1:
        pred_rgb = torch.sigmoid(raw)
    elif args.normalize_type == 2:
        pred_rgb = torch.tanh(raw)
    else:
        assert False, 'Wrong normalize type'

    return pred_rgb


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



def create_npp_net(args, selected_angles, selected_periods, res, percep_net):
    """Instantiate NPP-Net's MLP model.
    Args:
        args: arguments
        selected_angles: two orientations of periodicity
        selected_periods: two periods of periodicity along the orientations
        res: image resolution (h, w)
        percep_net: pretrained network for LPIPS.
    """

    # create positional encoder in the original nerf paper
    embedder, freq_nerf = get_embedder(args.multires, args.i_embed, res)

    # create position encoder based on the top-K periodicity
    embedder_periodics, input_ch_periodics = [], []
    for i in range(args.p_topk):
        embedder_periodic, input_ch_periodic = get_embedder(args.multires, args.i_embed, res,
                                                            selected_angles=selected_angles[i],
                                                            selected_periods=selected_periods[i],
                                                            freq_scales=args.freq_scales,
                                                            freq_offsets=args.freq_offsets,
                                                            angle_offsets= args.angle_offsets)
        embedder_periodics.append(embedder_periodic)
        input_ch_periodics.append(input_ch_periodic)

    input_ch_periodics = np.array(input_ch_periodics)

    output_ch = 3
    skips = [4]
    # apply coarse level periodicity augmentation (K > 1)
    if args.p_topk > 1:
        model = NPP_Net(D=args.netdepth, W=args.netwidth,
                           freq_nerf=freq_nerf, input_ch_periodic=input_ch_periodics[:1].sum(),
                           input_ch_periodic_aux = input_ch_periodics[1:].sum(), freq_scales=args.freq_scales,
                           freq_offsets=args.freq_offsets, angle_offsets=args.angle_offsets, output_ch=output_ch,
                           skips=skips, activation=args.activation).to(device)
    # DO NOT apply coarse level periodicity augmentation (K = 1)
    else:
        model = NPP_Net_top1(D=args.netdepth, W=args.netwidth,
                           freq_nerf=freq_nerf, input_ch_periodic=input_ch_periodics[:1].sum(), freq_scales=args.freq_scales,
                           freq_offsets=args.freq_offsets, angle_offsets=args.angle_offsets, output_ch=output_ch,
                           skips=skips, activation=args.activation).to(device)


    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # initialization if use relu
    if args.activation == 'relu':
        model.apply(weights_init_normal)

    # add trainable params for adaptive robust pixel loss
    robust_percep_param = []
    grad_vars = list(model.parameters())  + list(adaptive_pix.parameters())


    # add trainable params for adaptive robust perceptual loss
    if args.use_adaptive_perceptual_loss:
        for adaptive in percep_net.adaptive_perceps:
            robust_percep_param += list(adaptive.parameters())
        grad_vars += robust_percep_param

    network_query_fn = lambda inputs_periodic, network_fn : run_network(inputs_periodic, network_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    ##########################
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'network_fn' : model,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, embedder, embedder_periodics


