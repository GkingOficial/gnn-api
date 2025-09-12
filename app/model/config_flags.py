from types import SimpleNamespace

FLAGS = SimpleNamespace(
  task_type = 'classification',
  hidden_dim = 62,
  latent_dim = 768,
  max_atoms = 170,
  num_layers = 4,
  num_attn = 4,
  batch_size = 8,
  epoch_size = 200,
  num_train = 86,
  regularization_scale = 4e-4,
  beta1 = 0.9,
  beta2 = 0.98,
  optimizer = 'Adam',
  init_lr = 0.0003
)
