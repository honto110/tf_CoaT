import tensorflow as tf
from tensorflow.keras import layers

class TFIdentity(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
    
    def call(self, x):
        return x

class TFPatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, embedding_dim, name=None):
        super().__init__(name=name)
        
        self.proj = layers.Conv2D(filters=embedding_dim, 
                                  kernel_size=(patch_size, patch_size), 
                                  strides=(patch_size, patch_size),
                                  name=f'{name}.proj')
        self.norm = layers.LayerNormalization(epsilon=1e-05, name=f'{name}.norm')
        
        self.reshape = layers.Reshape((-1, embedding_dim))
    
    def call(self, x):
        x = self.proj(x)
        x = self.reshape(x)
        x = self.norm(x)
        
        return x
    
class TFConvPositionEncoding(tf.keras.layers.Layer):
    def __init__(self, dim, k, size, name=None):
        super().__init__(name=name)
        
        self.size = size
        self.proj = layers.Conv2D(filters=dim,
                                  kernel_size=k,
                                  strides=1,
                                  padding='same',
                                  groups=dim,
                                  name=f'{name}.proj'
                                 )
        self.reshape = [layers.Reshape((size[0], size[1], dim)), layers.Reshape((size[0] * size[1], dim))]
        
    def call(self, x):
        B, N, C = x.shape
        assert N == 1 + self.size[0] * self.size[1]
        cls_tokens, img_tokens = x[:, :1], x[:, 1:]
        a = self.reshape[0](img_tokens)
        y = self.proj(a) + a
        y = self.reshape[1](y)
        y = tf.concat([cls_tokens, y], axis=1)
        
        return y
    
class TFConvRelativePositionEncoding(tf.keras.layers.Layer):
    def __init__(self, ch, h, window, size, name=None):
        super().__init__(name=name)
        
        self.size = size
        self.window = window 
        self.conv_list = []
        self.head_splits = []
        for i, (curr_window, curr_head_split) in enumerate(window.items()): 
            curr_conv= layers.Conv2D(filters=curr_head_split*ch,
                                     kernel_size=curr_window,
                                     padding='same',
                                     dilation_rate=(1, 1),
                                     groups=curr_head_split*ch,
                                     name=f'{name}.conv_list.{i}'
                                    )
            self.conv_list.append(curr_conv)
            self.head_splits.append(curr_head_split)
            
        self.channel_splits = [x*ch for x in self.head_splits]
        
        self.reshape = [layers.Reshape((size[0], size[1], h * ch)), layers.Reshape((size[0] * size[1], h, ch))] 
        
    def call(self, q, v): #[B, N, H, C], [B, N, H, C]
        B, N, H, C = q.shape
        assert N == 1  + self.size[0] * self.size[1]
        
        q_img = q[:, 1:, :, :] #[B, H*W, H, C]
        v_img = v[:, 1:, :, :] #[B, H*W, H, C]
        
        v_img = self.reshape[0](v_img) #[B, H, W, H*C]
        v_img_list = tf.split(v_img, self.channel_splits, axis=3) #[ [B, H, W, A1], [B, H, W, A2], [B, H, W, A3] ], where A1+A2+A3 = H*C
        conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)] #[ [B, H, W, A1], [B, H, W, A2], [B, H, W, A3] ], where A1+A2+A3 = H*C
        conv_v_img = tf.concat(conv_v_img_list, axis=3) #[B, H, W, H*C]
        conv_v_img = self.reshape[1](conv_v_img) #[B, H*W, H, C]
        
        EV_hat_img = q_img * conv_v_img #[B, H*W, H, C]
        zero = tf.zeros((B, 1, H, C), dtype=q.dtype) #[B, 1, H, C]
        EV_hat = tf.concat([zero, EV_hat_img], axis=1) #[B, H*W+1, H, C]
        
        return EV_hat

class TFMlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.keras.activations.gelu, drop=0.0, name=None):
        super().__init__(name=name)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features, name=f'{name}.fc1')
        self.act = act_layer
        self.fc2 = layers.Dense(out_features, name=f'{name}.fc2')
        self.drop = layers.Dropout(drop)
    
    def call(self, x):
        y = self.fc1(x)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
    
        return y

class TFFactorAtt(tf.keras.layers.Layer):
    def __init__(self, dim, size, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., shared_crpe=None, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, name=f'{name}.qkv')
        self.qkv_reshape = layers.Reshape((-1, 3, self.num_heads, head_dim))
        self.reshape = layers.Reshape((-1, dim))
        self.softmax = layers.Softmax(axis=1)
        self.attn_drop = attn_drop # not used
        self.proj = layers.Dense(dim, name=f'{name}.proj')
        self.proj_drop = layers.Dropout(proj_drop)
        
        self.crpe = shared_crpe
        self.crpe._name = f'{name}.crpe'                         
    
    def call(self, x):
        qkv = self.qkv(x)
        qkv = self.qkv_reshape(qkv)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        k_softmax = self.softmax(k)
        k_softmax_T_dot_v = tf.einsum('bnhk, bnhv->bhkv', k_softmax, v)
        factor_att = tf.einsum('bnhk, bhkv->bnhv', q, k_softmax_T_dot_v)

        crpe = self.crpe(q, v)
        
        x = self.scale * factor_att + crpe
        x = self.reshape(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class TFSerialBlock(tf.keras.layers.Layer):
    def __init__(self, dim, size, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                act_layer=tf.keras.activations.gelu, norm_layer=layers.LayerNormalization, norm_layer_eps=1e-6, shared_cpe=None, shared_crpe=None, name=None):
        super().__init__(name=name)
        
        self.cpe = shared_cpe
        self.cpe._name = f'{name}.cpe'
        self.norm1 = norm_layer(epsilon=norm_layer_eps, name=f'{name}.norm1')
        self.factoratt_crpe = TFFactorAtt(dim=dim, size=size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                          attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe, name=f'{name}.factoratt_crpe')

        self.norm2 = norm_layer(epsilon=norm_layer_eps, name=f'{name}.norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TFMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, name=f'{name}.mlp')
    
    def call(self, x):
        y = self.cpe(x)
        curr = self.norm1(y)
        curr = self.factoratt_crpe(curr)
        y = y + curr
        
        curr = self.norm2(y)
        curr = self.mlp(curr)
        y = y + curr
        
        return y
    
class TFParallelBlock(tf.keras.layers.Layer):
    def __init__(self, dims, sizes, mlp_ratios, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=tf.keras.activations.gelu, norm_layer=layers.LayerNormalization, norm_layer_eps=1e-6, 
                 shared_cpes=None, shared_crpes=None, name=None):
        
        assert len(dims) == 4
        assert len(sizes) == 4
        assert len(mlp_ratios) == 4
        assert dims[1] == dims[2] == dims[3]
        assert mlp_ratios[1] == mlp_ratios[2] == mlp_ratios[3]
        
        super().__init__(name=name)
        
        self.sizes = sizes
        self.cpes = shared_cpes
        
        self.norm12 = norm_layer(epsilon=norm_layer_eps, name=f'{name}.norm12')
        self.norm13 = norm_layer(epsilon=norm_layer_eps, name=f'{name}.norm13')
        self.norm14 = norm_layer(epsilon=norm_layer_eps, name=f'{name}.norm14')
        
        self.factoratt_crpe2 = TFFactorAtt(dim=dims[1], size=sizes[1], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                           attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpes[1], name=f'{name}.factoratt_crpe2')
        
        self.factoratt_crpe3 = TFFactorAtt(dim=dims[2], size=sizes[2], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                           attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpes[2], name=f'{name}.factoratt_crpe3')
        
        self.factoratt_crpe4 = TFFactorAtt(dim=dims[3], size=sizes[3], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                           attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpes[3], name=f'{name}.factoratt_crpe4')
        
        self.upsample_3_2 = layers.UpSampling2D(size=(sizes[1][0] // sizes[2][0], sizes[1][1] // sizes[2][1]), interpolation='bilinear')
        self.upsample_4_3 = layers.UpSampling2D(size=(sizes[2][0] // sizes[3][0], sizes[2][1] // sizes[3][1]), interpolation='bilinear')
        self.upsample_4_2 = layers.UpSampling2D(size=(sizes[1][0] // sizes[3][0], sizes[1][1] // sizes[3][1]), interpolation='bilinear')
        self.downsample_2_3 = layers.AveragePooling2D(pool_size=(sizes[1][0] // sizes[2][0], sizes[1][1] // sizes[2][1]))
        self.downsample_3_4 = layers.AveragePooling2D(pool_size=(sizes[2][0] // sizes[3][0], sizes[2][1] // sizes[3][1]))
        self.downsample_2_4 = layers.AveragePooling2D(pool_size=(sizes[1][0] // sizes[3][0], sizes[1][1] // sizes[3][1]))
        
        self.reshape = []
        for size, dim in zip(sizes, dims):
            self.reshape.append((layers.Reshape((size[0], size[1], dim)),
                                 layers.Reshape((size[0] * size[1], dim))
                                ))
            
        self.norm22 = norm_layer(epsilon=norm_layer_eps, name=f'{name}.norm22')
        self.norm23 = norm_layer(epsilon=norm_layer_eps, name=f'{name}.norm23')
        self.norm24 = norm_layer(epsilon=norm_layer_eps, name=f'{name}.norm24')
        
        mlp_hidden_dim = int(dims[1] * mlp_ratios[1])
        
        self.mlp2 = TFMlp(in_features=dims[1], hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, name=f'{name}.mlp2')
        
    
    def apply(self, x, layer, input_reshape, output_reshape): # [B, N, C]
        cls_token = x[:, :1, :] # [B, 1, C]
        img_tokens = x[:, 1:, :] # [B, H*W, C]
        
        img_tokens = input_reshape[0](img_tokens) # [B, H, W, C]
        img_tokens = layer(img_tokens) # [B, A, B, C]
        img_tokens = output_reshape[1](img_tokens) # [B, A*B, C]

        return tf.concat([cls_token, img_tokens], axis=1) # [B, A*B+1, C]
        
    
    def call(self, x1, x2, x3, x4): 
        x2 = self.cpes[1](x2)
        x3 = self.cpes[2](x3)
        x4 = self.cpes[3](x4)
        
        curr2 = self.norm12(x2)
        curr3 = self.norm13(x3)
        curr4 = self.norm14(x4)
        
        curr2 = self.factoratt_crpe2(curr2)
        curr3 = self.factoratt_crpe3(curr3)
        curr4 = self.factoratt_crpe4(curr4)
        
        curr_3_2 = self.apply(curr3, self.upsample_3_2, input_reshape=self.reshape[2], output_reshape=self.reshape[1])
        curr_4_3 = self.apply(curr4, self.upsample_4_3, input_reshape=self.reshape[3], output_reshape=self.reshape[2])
        curr_4_2 = self.apply(curr4, self.upsample_4_2, input_reshape=self.reshape[3], output_reshape=self.reshape[1])
        
        curr_2_3 = self.apply(curr2, self.downsample_2_3, input_reshape=self.reshape[1], output_reshape=self.reshape[2])
        curr_3_4 = self.apply(curr3, self.downsample_3_4, input_reshape=self.reshape[2], output_reshape=self.reshape[3])
        curr_2_4 = self.apply(curr2, self.downsample_2_4, input_reshape=self.reshape[1], output_reshape=self.reshape[3])
    
        curr2 = curr2 + curr_3_2 + curr_4_2
        curr3 = curr3 + curr_4_3 + curr_2_3
        curr4 = curr4 + curr_3_4 + curr_2_4
        
        x2 = x2 + curr2
        x3 = x3 + curr3
        x4 = x4 + curr4
        
        curr2 = self.norm22(x2)
        curr3 = self.norm23(x3)
        curr4 = self.norm24(x4)
        
        curr2 = self.mlp2(curr2)
        curr3 = self.mlp2(curr3)
        curr4 = self.mlp2(curr4)
        
        x2 = x2 + curr2
        x3 = x3 + curr3
        x4 = x4 + curr4
        
        return x1, x2, x3, x4
        
class TFCoaT(tf.keras.Model):
    def __init__(self, input_size=768, patch_sizes=[4, 2, 2, 2], embed_dims=[152, 320, 320, 320], serial_depths=[2, 2, 2, 2], decoder_dim=320, parallel_depth=6,
                 num_heads=8, mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., crpe_window={3:2, 5:3, 7:3},
                 norm_layer=layers.LayerNormalization, norm_layer_eps=1e-6):
        super().__init__()
        
        # Sizes.
        size1 = (input_size // patch_sizes[0], input_size // patch_sizes[0])
        size2 = (size1[0] // patch_sizes[1], size1[1] // patch_sizes[1]) 
        size3 = (size2[0] // patch_sizes[2], size2[1] // patch_sizes[2])
        size4 = (size3[0] // patch_sizes[3], size3[1] // patch_sizes[3])
        
        # Patch embeddings.
        self.patch_embedding1 = TFPatchEmbedding(patch_size=patch_sizes[0], embedding_dim=embed_dims[0], name='patch_embed1')
        self.patch_embedding2 = TFPatchEmbedding(patch_size=patch_sizes[1], embedding_dim=embed_dims[1], name='patch_embed2')
        self.patch_embedding3 = TFPatchEmbedding(patch_size=patch_sizes[2], embedding_dim=embed_dims[2], name='patch_embed3')
        self.patch_embedding4 = TFPatchEmbedding(patch_size=patch_sizes[3], embedding_dim=embed_dims[3], name='patch_embed4')
        
        # Class tokens.
        self.cls_token1 = tf.Variable(tf.random.normal((1, 1, embed_dims[0]), stddev=0.02), name='cls_token1')
        self.cls_token2 = tf.Variable(tf.random.normal((1, 1, embed_dims[1]), stddev=0.02), name='cls_token2')
        self.cls_token3 = tf.Variable(tf.random.normal((1, 1, embed_dims[2]), stddev=0.02), name='cls_token3')
        self.cls_token4 = tf.Variable(tf.random.normal((1, 1, embed_dims[3]), stddev=0.02), name='cls_token4')
        
        # Convolutional position encodings.
        self.cpe1 = TFConvPositionEncoding(dim=embed_dims[0], k=3, size=size1, name='cpe1')
        self.cpe2 = TFConvPositionEncoding(dim=embed_dims[1], k=3, size=size2, name='cpe2')
        self.cpe3 = TFConvPositionEncoding(dim=embed_dims[2], k=3, size=size3, name='cpe3')
        self.cpe4 = TFConvPositionEncoding(dim=embed_dims[3], k=3, size=size4, name='cpe4')
        
        # Convolutional relative position encodings.
        self.crpe1 = TFConvRelativePositionEncoding(ch=embed_dims[0] // num_heads, h=num_heads, window=crpe_window, size=size1, name='crpe1')
        self.crpe2 = TFConvRelativePositionEncoding(ch=embed_dims[1] // num_heads, h=num_heads, window=crpe_window, size=size2, name='crpe2')
        self.crpe3 = TFConvRelativePositionEncoding(ch=embed_dims[2] // num_heads, h=num_heads, window=crpe_window, size=size3, name='crpe3')
        self.crpe4 = TFConvRelativePositionEncoding(ch=embed_dims[3] // num_heads, h=num_heads, window=crpe_window, size=size4, name='crpe4')
        
        # Serial blocks.
        self.serial_blocks1 = [TFSerialBlock(dim=embed_dims[0], num_heads=num_heads, mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop_rate, attn_drop=attn_drop_rate,  
                                             norm_layer=norm_layer, norm_layer_eps=norm_layer_eps, shared_cpe=self.cpe1, shared_crpe=self.crpe1,
                                             size=size1, name=f'serial_blocks1.{i}'
                                            ) 
                               for i, _ in enumerate(range(serial_depths[0]))]
         
        self.serial_blocks2 = [TFSerialBlock(dim=embed_dims[1], num_heads=num_heads, mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop_rate, attn_drop=attn_drop_rate, 
                                             norm_layer=norm_layer, norm_layer_eps=norm_layer_eps, shared_cpe=self.cpe2, shared_crpe=self.crpe2,
                                             size=size2, name=f'serial_blocks2.{i}'
                                            ) 
                               for i, _ in enumerate(range(serial_depths[1]))]
        
        self.serial_blocks3 = [TFSerialBlock(dim=embed_dims[2], num_heads=num_heads, mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop_rate, attn_drop=attn_drop_rate, 
                                             norm_layer=norm_layer, norm_layer_eps=norm_layer_eps, shared_cpe=self.cpe3, shared_crpe=self.crpe3,
                                             size=size3, name=f'serial_blocks3.{i}'
                                            ) 
                               for i, _ in enumerate(range(serial_depths[2]))]
        
        self.serial_blocks4 = [TFSerialBlock(dim=embed_dims[3], num_heads=num_heads, mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop_rate, attn_drop=attn_drop_rate,
                                             norm_layer=norm_layer, norm_layer_eps=norm_layer_eps, shared_cpe=self.cpe4, shared_crpe=self.crpe4,
                                             size=size4, name=f'serial_blocks4.{i}'
                                            ) 
                               for i, _ in enumerate(range(serial_depths[3]))]
        
        self.reshape = []
        for size, embed_dim in zip([size1, size2, size3, size4], embed_dims):
            self.reshape.append([layers.Reshape((*size, embed_dim)), 
                                 layers.Reshape((size[0] * size[1], embed_dim))])
        
        # Parallel blocks.
        self.parallel_depth = parallel_depth
        if parallel_depth >= 1:
            self.parallel_blocks = [TFParallelBlock(dims=embed_dims, sizes=[size1, size2, size3, size4], num_heads=num_heads, mlp_ratios=mlp_ratios, 
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, 
                                                    norm_layer=norm_layer, norm_layer_eps=norm_layer_eps,
                                                    shared_cpes=[self.cpe1, self.cpe2, self.cpe3, self.cpe4],
                                                    shared_crpes=[self.crpe1, self.crpe2, self.crpe3, self.crpe4],
                                                    name=f'parallel_blocks.{i}'
                                                   )
                                   for i, _ in enumerate(range(parallel_depth))]
                          
    def call(self, x):
        x1 = self.patch_embedding1(x)
        x1 = self.insert_cls(x1, self.cls_token1)
        for blk in self.serial_blocks1:
            x1 = blk(x1)
        x1_nocls = self.remove_cls(x1)
        x1_nocls = self.reshape[0][0](x1_nocls)
        
        x2 = self.patch_embedding2(x1_nocls)
        x2 = self.insert_cls(x2, self.cls_token2)
        for blk in self.serial_blocks2:
            x2 = blk(x2)
        x2_nocls = self.remove_cls(x2)
        x2_nocls = self.reshape[1][0](x2_nocls)
        
        x3 = self.patch_embedding3(x2_nocls)
        x3 = self.insert_cls(x3, self.cls_token3)
        for blk in self.serial_blocks3:
            x3 = blk(x3)
        x3_nocls = self.remove_cls(x3)
        x3_nocls = self.reshape[2][0](x3_nocls)
        
        x4 = self.patch_embedding4(x3_nocls)
        x4 = self.insert_cls(x4, self.cls_token4)
        for blk in self.serial_blocks4:
            x4 = blk(x4)
        x4_nocls = self.remove_cls(x4)
        x4_nocls = self.reshape[3][0](x4_nocls)
        
        if self.parallel_depth >= 1:
            for i, blk in enumerate(self.parallel_blocks):
                x1, x2, x3, x4 = blk(x1, x2, x3, x4)

            x1_nocls = self.remove_cls(x1)
            x1_nocls = self.reshape[0][0](x1_nocls)

            x2_nocls = self.remove_cls(x2)
            x2_nocls = self.reshape[1][0](x2_nocls)

            x3_nocls = self.remove_cls(x3)
            x3_nocls = self.reshape[2][0](x3_nocls)

            x4_nocls = self.remove_cls(x4)
            x4_nocls = self.reshape[3][0](x4_nocls)
        
        out = [x1_nocls, x2_nocls, x3_nocls, x4_nocls]
        
        return out
    
    def insert_cls(self, x, cls_token):
        B, N, C = x.shape
        cls_tokens = tf.tile(cls_token, [B, 1, 1])
        return tf.concat([cls_tokens, x], axis=1)
    
    def remove_cls(self, x):
        return x[:, 1:, :]
    
    def load_weights_from_torch(self, pytorch_weights: dict):
        keywords = {'weight': ['kernel', 'gamma'],
                    'running_mean': ['moving_mean'],
                    'running_var': ['moving_variance'],
                    'bias': ['bias', 'beta'],
                   }

        tensorflow_weights_names, pytorch_weights_names = [weights.name for weights in self.weights], pytorch_weights.keys()

        descriptions=[]
        for pytorch_name in pytorch_weights_names:
            for i, tensorflow_name in enumerate(tensorflow_weights_names):
                if pytorch_name in tensorflow_name: 
                    descriptions.append({'pname': pytorch_name, 'tname': tensorflow_name, 'tposition': i})
                    continue
                for pytorch_keyword in keywords.keys():
                    for tensorflow_keyword in keywords[pytorch_keyword]:
                        probable_name = pytorch_name.replace('.' + pytorch_keyword, '/' + tensorflow_keyword)
                        if probable_name in tensorflow_name: descriptions.append({'pname': pytorch_name, 'tname': tensorflow_name, 'tposition': i})

        tensorflow_weights = self.get_weights()
        for count, desc in enumerate(descriptions):
            p_weights, t_weights = pytorch_weights[desc['pname']], tensorflow_weights[desc['tposition']]
            if p_weights.dim() == 2: p_weights = torch.permute(p_weights, (1, 0))
            if p_weights.dim() == 4: p_weights = torch.permute(p_weights, (2, 3, 1, 0))
            tensorflow_weights[desc['tposition']] = p_weights.numpy()

        self.set_weights(tensorflow_weights)