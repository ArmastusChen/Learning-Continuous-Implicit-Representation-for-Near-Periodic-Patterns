
def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=False,
                        help='config file path')
    parser.add_argument("--expname", type=str, default='completion',
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./results',
                        help='where to store ckpts and logs')


    parser.add_argument("--datadir", type=str, default='data/completion/detected/20150911134723-104840a8', help='input data directory')
    parser.add_argument('--ckpt', type=str, default=None)
    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=512,
                        help='channels per layer')

    parser.add_argument("--N_rand", type=int, default=32*32*8,
                        help='batch size for pixel loss (number of pixel used for training per iteration)')

    parser.add_argument("--warm_up_epoch", type=int, default=0 ,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--patch_num", type=int, default=2,
                        help='batch size for patch loss (number of patch used for training per iteration)')
    parser.add_argument("--num_real_patch_per_sample", type=int, default=3,
                        help='number of real (known or gt) patches that are sampled '
                             'for a specific fake (predicted) patch')
    parser.add_argument("--patch_size_decay", type=int, default=2000,
                        help='decrease patch size every patch_size_decay iterations')
    parser.add_argument("--extrapolation", action='store_true')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=500,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*4096,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--freq_scales", type=list, default=[1],
                        help='a set of fine level periodicity augmentation: augmented_p = freq_scale * p')
    parser.add_argument("--freq_offsets", type=int, default=[0, -1, 1, 0.5, -0.5],
                        help='a set of fine level periodicity augmentation: augmented_p = freq_offset + p')
    parser.add_argument("--angle_offsets", type=int, default=[0],
                        help='a set of fine level periodicity augmentation: augmented_orientation = orientation + angle_offset')

    parser.add_argument("--p_topk", type=int, default=3,
                        help='top K periodicity')
    parser.add_argument("--invalid_ratio", type=float, default=0.3,
                        help='layers in network')


    # rendering options
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--activation", type=str, default='snake',
                        help='activation function for the MLP.  Choose snake or relu')
    parser.add_argument("--normalize_type", type=int, default=1,
                        help='1. [0,1],  2.  [-1,1]')

    parser.add_argument("--loss_type", type=str, default='robust_loss_adaptive',
                        help='robust_loss_adaptive, l2, robust_loss')
    parser.add_argument("--use_adaptive_style_loss", action='store_false')
    parser.add_argument("--use_adaptive_perceptual_loss", action='store_false')
    parser.add_argument("--no_pix_loss", action='store_true', help='do not use pixel loss')
    parser.add_argument("--no_reg_sampling", action='store_true', help='do not use periodicity-based patch '
                                                                       'sampling (random patch sampling instead)')
    parser.add_argument("--use_contextual_loss", action='store_false', help='use contextual loss')
    parser.add_argument("--use_perceptual_loss", action='store_false', help='use perceptial loss')
    parser.add_argument("--use_comp", action='store_false', help='compose known regions in the input region to '
                                                                 'predicted patches if applicable')
    parser.add_argument("--use_patch_weight", action='store_true', help='assign weight for sampled patches '
                                                                        'based on the distance')


    parser.add_argument("--contextual_weight", type=float, default=0.001,
                        help='weight of contextual loss')
    parser.add_argument("--perceptual_weight", type=float, default=0.001,
                        help='weight of perceptual loss')

    # logging/saving options
    parser.add_argument("--N_iters",   type=int, default=2401,
                        help='Number of iteration for trianing')
    parser.add_argument("--i_print",   type=int, default=600,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img",     type=int, default=50000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=1000000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=600,
                        help='frequency of testset saving')


    return parser
