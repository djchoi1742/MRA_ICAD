

class Model23:  # Spider U-Net (3 blocks) + only detection branch
    def __init__(self, input_size, f_num, is_training, det_size=16, t=True, use_ic=True, use_se=False, mtl_mode=False):

        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        concats, decodes = {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:
            concats[j] = concatenate([encodes[j], _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)], axis=-1)
            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        cls_det_conv = _conv2d(inner_conv, 1, 3, 1, t, is_training=is_training)  # conv-relu-bn

        cls_logits = GlobalAveragePooling3D()(cls_det_conv)
        cls_probs = tf.nn.sigmoid(cls_logits)

        det_logits = tf.reduce_mean(AveragePooling3D(pool_size=(8, 2, 2), padding='SAME')(cls_det_conv), axis=1)
        det_scores = tf.nn.sigmoid(det_logits)

        cls_scores = tf.reshape(cls_probs, (-1, 1, 1, 1))
        det_probs = det_scores
        # det_probs = det_scores * cls_scores

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8

        self.model = Model(inputs=self.images, outputs=[cls_probs, det_probs])
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1]])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)


class Model24:  # Spider U-Net (3 blocks) + only detection branch (each_ste=False)
    def __init__(self, input_size, f_num, is_training, det_size=16, use_ic=True, use_se=False, t=True,
                 mtl_mode=False, **kwargs):

        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        deconvs, concats, decodes = {}, {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            deconvs[j] = _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)

            if use_se:
                deconv_se = se_block_2d(_conv2d(deconvs[j], f_num[j - 1], 1, 1, t, is_training=is_training))
                encode_conv = _conv2d(encodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)
                concats[j] = concatenate([encode_conv * deconv_se, deconvs[j]], axis=-1)
            else:
                concats[j] = concatenate([encodes[j], deconvs[j]], axis=-1)

            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

            # concats[j] = concatenate([encodes[j], _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)], axis=-1)
            # decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            # decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, 1, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        cls_det_conv = _conv2d(inner_conv, 1, 3, 1, t, af=None, bn=False, is_training=is_training)

        det_logits = tf.reduce_mean(AveragePooling3D(pool_size=(8, 2, 2), padding='SAME')(cls_det_conv), axis=1)
        det_probs = tf.nn.sigmoid(det_logits)

        if False:
            inner_pool = _avgpool2d(inner_conv, t)
            det_lstm = Bidirectional(ConvLSTM2D(filters=8, kernel_size=3, strides=1, padding='same',
                                     return_sequences=False), merge_mode='ave')(inner_pool)
            det_conv = _conv2d(det_lstm, 1, 3, 1, add_time_axis=False, af=None, bn=False, is_training=is_training)
            det_probs = tf.nn.sigmoid(det_conv)

        # det_conv = _conv2d(inner_conv, 1, 3, 1, t, bn=False, is_training=is_training)
        # det_logits = tf.reduce_mean(AveragePooling3D(pool_size=(8, 2, 2), padding='SAME')(det_conv), axis=1)
        # det_probs = tf.nn.sigmoid(det_logits)

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8

        self.model = Model(inputs=self.images, outputs=[seg_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1]])

        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)


class Model25:  # Spider U-Net + Branch + Inner-connected module + Attention skip-layer fusion
    def __init__(self, input_size, f_num, is_training, use_ic=True, use_se=False, t=True,
                 mtl_mode=False, **kwargs):

        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        deconvs, concats, decodes = {}, {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            deconvs[j] = _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)

            if use_se:
                deconv_se = se_block_2d(_conv2d(deconvs[j], f_num[j - 1], 1, 1, t, is_training=is_training))
                encode_conv = _conv2d(encodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)
                concats[j] = concatenate([encode_conv * deconv_se, deconvs[j]], axis=-1)
            else:
                concats[j] = concatenate([encodes[j], deconvs[j]], axis=-1)

            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, 1, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        # inner_conv = _conv2d(inner_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_pool = _avgpool2d(inner_conv, t)
        det_lstm = Bidirectional(ConvLSTM2D(filters=8, kernel_size=3, strides=1, padding='same',
                                 return_sequences=False), merge_mode='ave')(inner_pool)
        det_conv = _conv2d(det_lstm, 1, 3, 1, add_time_axis=False, af=None, bn=False, is_training=is_training)
        det_probs = tf.nn.sigmoid(det_conv)

        # inner_pool = _avgpool2d(inner_conv, t)
        # det_conv = _conv2d(inner_pool, 1, 3, 1, t, af=None, bn=False, is_training=is_training)
        # det_logits = tf.reduce_mean(AveragePooling3D(pool_size=(8, 1, 1), padding='SAME')(det_conv), axis=1)
        # det_probs = tf.nn.sigmoid(det_logits)

        # det_conv = _conv2d(inner_conv, 1, 3, 1, t, af=None, bn=False, is_training=is_training)
        # det_logits = tf.reduce_mean(AveragePooling3D(pool_size=(8, 2, 2), padding='SAME')(det_conv), axis=1)
        # det_probs = tf.nn.sigmoid(det_logits)

        inner_gap = GlobalAveragePooling3D()(inner_conv)
        inner_cls_logits = layers.Dense(1, activation=None)(inner_gap)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], self.model.output[2]])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model271:  # 2D U-Net + Branch + Inner-connected module + Attention skip-layer fusion
    def __init__(self, input_size, f_num, is_training, use_ic=True, use_se=False, t=False, mtl_mode=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 3

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        deconvs, concats, decodes = {}, {}, {}
        decodes[b_num] = _conv2d(downs[reps], f_num[-1], 3, 1, t, is_training=is_training)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            deconvs[j] = _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)
            if use_se:
                deconv_se = se_block_2d(_conv2d(deconvs[j], f_num[j - 1], 1, 1, t, is_training=is_training))
                encode_conv = _conv2d(encodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)
                concats[j] = concatenate([encode_conv * deconv_se, deconvs[j]], axis=-1)
            else:
                concats[j] = concatenate([encodes[j], deconvs[j]], axis=-1)

            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, 1, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        # serial 11, 14, 16, 17, 18, 19, 20
        cls_det_conv = _conv2d(inner_conv, 1, 3, 1, t, is_training=is_training)  # serial 11, 18, 20, 22, 24

        cls_logits = GlobalAveragePooling3D()(cls_det_conv)
        cls_probs = tf.nn.sigmoid(cls_logits)

        det_logits = tf.reduce_mean(AveragePooling3D(pool_size=(8, 2, 2), padding='SAME')(cls_det_conv), axis=1)
        det_scores = tf.nn.sigmoid(det_logits)

        cls_scores = tf.reshape(cls_probs, (-1, 1, 1, 1))
        det_probs = det_scores * cls_scores

        # serial 13, 15
        # cls_det_conv = _conv2d(inner_conv, 1, 3, 1, t, is_training=is_training)  # serial 13
        # cls_det_conv = _conv2d(inner_conv, 1, 3, 1, t, af=None, bn=False, is_training=is_training)  # serial 15

        # cls_logits = GlobalAveragePooling3D()(cls_det_conv)
        # cls_probs = tf.nn.sigmoid(cls_logits)
        # det_logits = tf.reduce_mean(AveragePooling3D(pool_size=(8, 2, 2), padding='SAME')(cls_det_conv), axis=1)
        # det_probs = tf.nn.sigmoid(det_logits)

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], self.model.output[2]])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model272:  # Spider U-Net + Branch + Inner-connected module + Attention skip-layer fusion
    def __init__(self, input_size, f_num, is_training, use_ic=True, use_se=False, t=True, mtl_mode=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        deconvs, concats, decodes = {}, {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            deconvs[j] = _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)
            if use_se:
                deconv_se = se_block_2d(_conv2d(deconvs[j], f_num[j - 1], 1, 1, t, is_training=is_training))
                encode_conv = _conv2d(encodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)
                concats[j] = concatenate([encode_conv * deconv_se, deconvs[j]], axis=-1)
            else:
                concats[j] = concatenate([encodes[j], deconvs[j]], axis=-1)

            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, 1, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        cls_det_conv = _conv2d(inner_conv, 1, 3, 1, t, is_training=is_training)  # serial 11, 18

        cls_logits = tf.reduce_mean(cls_det_conv, axis=[2, 3, 4])
        cls_probs = tf.nn.sigmoid(cls_logits)

        det_logits = AveragePooling3D(pool_size=(1, 2, 2), padding='SAME')(cls_det_conv)
        det_scores = tf.nn.sigmoid(det_logits)

        cls_scores = tf.reshape(cls_probs, (-1, 8, 1, 1, 1))
        det_probs = det_scores * cls_scores

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], self.model.output[2]])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model281:  # each_ste=False (for 3D)
    def __init__(self, input_size, f_num, is_training, use_ic=False, use_se=False, t=True, mtl_mode=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        deconvs, concats, decodes = {}, {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            deconvs[j] = _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)
            if use_se:
                deconv_se = se_block_2d(_conv2d(deconvs[j], f_num[j - 1], 1, 1, t, is_training=is_training))
                encode_conv = _conv2d(encodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)
                concats[j] = concatenate([encode_conv * deconv_se, deconvs[j]], axis=-1)
            else:
                concats[j] = concatenate([encodes[j], deconvs[j]], axis=-1)

            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, 1, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_gap = GlobalAveragePooling3D()(inner_conv)
        inner_cls_logits = layers.Dense(1, activation=None)(inner_gap)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        det_conv = _conv2d(top_conv, 1, 3, 1, t, is_training=is_training)
        det_gap = tf.reduce_mean(AveragePooling3D(pool_size=(8, 16, 16), padding='SAME')(det_conv), axis=1)
        det_scores = tf.nn.sigmoid(det_gap)

        cls_scores = tf.reshape(cls_probs, (-1, 1, 1, 1))
        det_probs = det_scores * cls_scores

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables.copy()

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], self.model.output[2]])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model283:  # each_ste=True
    def __init__(self, input_size, f_num, is_training, use_ic=False, use_se=False, t=True, mtl_mode=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        deconvs, concats, decodes = {}, {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            deconvs[j] = _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)
            if use_se:
                deconv_se = se_block_2d(_conv2d(deconvs[j], f_num[j - 1], 1, 1, t, is_training=is_training))
                encode_conv = _conv2d(encodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)
                concats[j] = concatenate([encode_conv * deconv_se, deconvs[j]], axis=-1)
            else:
                concats[j] = concatenate([encodes[j], deconvs[j]], axis=-1)

            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, 1, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_gap = tf.reduce_mean(AveragePooling3D(pool_size=(1, 32, 32), padding='SAME')(inner_conv), axis=[2, 3])
        inner_cls_logits = tf.reduce_mean(layers.Dense(1, activation=None)(inner_gap), axis=-1)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        # det_conv = _conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 0~3,5
        det_conv = _last_conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 4
        det_scores = AveragePooling3D(pool_size=(1, 16, 16), padding='SAME')(det_conv)  # exp009 serial 0

        cls_scores = tf.reshape(cls_probs, (-1, 8, 1, 1, 1))
        det_probs = det_scores * cls_scores

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables.copy()

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], self.model.output[2],
                                              det_scores])
        self.check_model = keras.Model(inputs=self.images, outputs=det_scores)
        print(self.model.get_layer(self.cam_layer_name).output)
