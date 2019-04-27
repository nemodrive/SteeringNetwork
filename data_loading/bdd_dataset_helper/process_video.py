def augmentation(args, is_train, net_inputs, net_outputs):
    # augment the network input tensors and output tensors by whether is_train
    # return augment_net_input, augment_net_output
    if args.use_data_augmentation and is_train:
        # TODO: has not debug this block yet
        images = net_inputs[0]
        with tf.variable_scope("distort_video"):
            print(
                "using random crop and brightness and constrast jittering")
            # shape = B F H W C
            shape = [x.value for x in images.get_shape()]
            images = tf.reshape(
                images,
                [shape[0] * shape[1], shape[2], shape[3], shape[4]])

            images = tf.image.random_brightness(
                images, max_delta=64. / 255.)
            images = tf.image.random_contrast(images, lower=0.6, upper=1.4)
            #images = tf.image.random_hue(images, max_delta=0.2)
            #images = tf.image.random_saturation(images, lower=0.7, upper=1.3)

            # The random_* ops do not necessarily clamp. But return uint8 thus not needed
            #images = tf.clip_by_value(images, 0, 255)

            images = tf.reshape(images, shape)
            images = tf.cast(images, tf.uint8)
        net_inputs[0] = images

    if args.use_perspective_augmentation:
        images = net_inputs[0]  # shape:: N * F * HWC
        images_shape = [x.value for x in images.get_shape()]
        future_labels = net_outputs[2]  # shape: N * F * 2
        future_labels_shape = [x.value for x in future_labels.get_shape()]

        images, future_labels = tf.py_func(MyDataset.perspective_changes,
                                           [images, future_labels],
                                           [tf.uint8, tf.float32])

        images = tf.reshape(images, [
            images_shape[0] * images_shape[1], images_shape[2],
            images_shape[3], images_shape[4]
        ])
        images = tf.image.resize_bilinear(images, (228, 228))
        images = tf.cast(images, tf.uint8)
        images = tf.reshape(
            images,
            [images_shape[0], images_shape[1], 228, 228, images_shape[4]])

        future_labels.set_shape(future_labels_shape)
        net_inputs[0] = images
        net_outputs[2] = future_labels

    return net_inputs, net_outputs

    # the input should be bottom cropped image, i.e. no car hood
def rotate_ground(original,
                  theta,
                  horizon=60,
                  half_height=360 / 2,
                  focal=1.0):
    height, width, channel = original.shape
    # the target grids
    yp = range(height - horizon, height)
    xp = range(0, width)

    # from pixel to coordinates
    y0 = (np.array(yp) - half_height) * 1.0 / half_height
    x0 = (np.array(xp) - width / 2) / (width / 2.0)

    # form the mesh
    mesh = MyDataset.generate_meshlist(x0, y0)
    # compute the source coordinates
    st = math.sin(theta)
    ct = math.cos(theta)
    deno = ct * focal + st * mesh[:, 0]
    out = np.array([(-st * focal + ct * mesh[:, 0]) / deno,
                    mesh[:, 1] / deno])

    # interpolate
    vout = []
    for i in range(3):
        f = interpolate.RectBivariateSpline(y0, x0,
                                            original[-horizon:, :, i])
        values = f(out[1, :], out[0, :], grid=False)
        vout.append(values)

    lower = np.reshape(vout, (3, width, horizon)).transpose(
        (2, 1, 0)).astype("uint8")

    # compute the upper part
    out = np.reshape(out[0, :], (width, horizon))
    out = out[:, 0]
    f = interpolate.interp1d(
        x0,
        original[:-horizon, :, :],
        axis=1,
        fill_value=(original[:-horizon, 0, :], original[:-horizon, -1, :]),
        bounds_error=False)
    ans = f(out)
    ans = ans.astype("uint8")

    return np.concatenate((ans, lower), axis=0)

def perspective_changes(images, future_labels):
    N, F, H, W, C = images.shape
    N2, F2, C2 = future_labels.shape
    assert (N == N2)
    assert (F == F2)
    assert (C2 == 2)

    perspective_aug_prob = 0.03
    perspective_recover_time = 2.0  # second
    perspective_theta_std = 0.15
    # image related
    horizon = 60
    half_height = 360 / 2
    focal = 1.0

    # precomputed constants
    downsampled_framerate = FLAGS.frame_rate / FLAGS.temporal_downsample_factor
    num_frames = int(perspective_recover_time * downsampled_framerate)

    for ni in range(N):
        i = 0
        while i + num_frames < F:
            if random.random() < perspective_aug_prob:
                # then we need to augment the images starting from this one
                # random sample a rotate angle
                theta = random.gauss(0, perspective_theta_std)
                yaw_rate_delta = -theta / perspective_recover_time

                for j in range(num_frames):
                    # distort each of the frames and yaw rate
                    images[ni, i, :, :, :] = MyDataset.rotate_ground(
                        images[ni, i, :, :, :],
                        theta * (1 - 1.0 * j / num_frames), horizon,
                        half_height, focal)
                    future_labels[ni, i, 0] += yaw_rate_delta
                    i += 1
            else:
                i += 1
    return [images, future_labels]


