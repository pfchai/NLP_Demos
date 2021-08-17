{
  dataset_reader: {
    type: 'weibo2018',
    tokenizer: {
      type: 'jieba',
    },
    token_indexers: {
      tokens: {
        type: 'single_id',
      },
    },
    max_sequence_length: 100,
  },

  datasets_for_vocab_creation: ['train'],
  train_data_path: './dataset/weibo2018/train.txt',
  validation_data_path: './dataset/weibo2018/test.txt',
  model: {
    type: 'basic_classifier',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          embedding_dim: 10,
          trainable: true,
        },
      },
    },
    seq2seq_encoder: {
      type: 'lstm',
      num_layers: 1,
      bidirectional: false,
      input_size: 10,
      hidden_size: 16,
    },
    seq2vec_encoder: {
      type: 'bag_of_embeddings',
      embedding_dim: 16,
      averaged: true,
    },
    feedforward: {
      input_dim: 16,
      num_layers: 1,
      hidden_dims: 20,
      activations: 'relu',
      dropout: 0.1,
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 16,
      // sorting_keys: [''],
    },
  },
  trainer: {
    optimizer: {
      type: 'adam',
      lr: 0.001,
    },
    validation_metric: '+accuracy',
    num_epochs: 10,
    grad_norm: 10.0,
    patience: 5,
    cuda_device: -1,
  },
}
