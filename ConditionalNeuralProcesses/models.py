import torch
from torch.distributions import kl_divergence

LATENT_SPACE = 100
HIDDEN_DIMS = 128


def sum_reduce(a):
    view = a.view((a.shape[0], -1))
    return view.mean(dim=0).sum()


class NPEncoder(torch.nn.Module):
    def __init__(self, input_features=2, hidden_dims=50, hidden_layers=1, latent_space=LATENT_SPACE):
        super(NPEncoder, self).__init__()

        self.model_seq = torch.nn.Sequential(*[
            torch.nn.Linear(input_features, hidden_dims),
            torch.nn.ReLU(inplace=True),

            # Different Linear Layers
            *(hidden_layers * [
                torch.nn.Linear(hidden_dims, hidden_dims),
                torch.nn.ReLU(inplace=True)]),

            torch.nn.Linear(hidden_dims, 2 * latent_space)
        ])

        self.post_layer = torch.nn.Sequential(*[
            torch.nn.Linear(2 * latent_space, 2 * latent_space),
            torch.nn.ReLU(inplace=True)
        ])

        self.mu_layer = torch.nn.Linear(latent_space, latent_space)
        self.sigma_layer = torch.nn.Linear(latent_space, latent_space)

    def forward(self, x):
        batch, points, _ = x.shape
        x = x.view((batch * points, -1))

        x = self.model_seq(x)

        x = x.view((batch, points, -1))

        # # AGGREGATOR
        z = torch.mean(x, dim=1)

        x = self.post_layer(z)

        # Get Latent distribution
        mu, log_std = torch.chunk(x, 2, 1)
        mu = self.mu_layer(mu)
        log_std = self.sigma_layer(log_std)
        dist = torch.distributions.Normal(mu, torch.exp(log_std))
        return mu, log_std, dist


class NPDecoder(torch.nn.Module):
    def __init__(self, latent_space=LATENT_SPACE, hidden_dims=50, hidden_layers=2):
        super(NPDecoder, self).__init__()

        self.decoder_layers = torch.nn.Sequential(*[
            torch.nn.Linear(latent_space + 1, hidden_dims),
            torch.nn.ReLU(inplace=True),

            *(hidden_layers * [torch.nn.Linear(hidden_dims, hidden_dims),
                               torch.nn.ReLU(inplace=True)]),
        ])

        self.mu_layer = torch.nn.Linear(hidden_dims, 1)
        self.sigma_layer = torch.nn.Linear(hidden_dims, 1)

    def forward(self, representation, target_x):
        # reshape the respresentation to fit the targets
        representation = torch.unsqueeze(representation, 1)
        representation = representation.repeat(1, target_x.shape[1], 1)

        # Copy the representation
        x = torch.cat((representation, target_x), 2)

        # Get a full vector of it
        batch, points, _ = x.shape

        x = x.view((batch * points, -1))

        x = self.decoder_layers(x)
        mu = self.mu_layer(x)
        log_sigma = self.sigma_layer(x)

        mu = mu.view(batch, points, -1)
        log_sigma = log_sigma.view(batch, points, -1)

        # [batch_size,  2*num_target_x]
        log_sigma = 0.1 + 0.9 * torch.nn.Softplus()(log_sigma)

        dist = torch.distributions.Normal(mu, log_sigma)

        return mu, log_sigma, dist


class NeuralProcess(torch.nn.Module):
    def __init__(self, hidden_dims=50, hidden_layer_encoder=1, hidden_layers_decoder=2, lr=3e-4):
        super(NeuralProcess, self).__init__()

        self.encoder = NPEncoder(hidden_dims=hidden_dims, hidden_layers=hidden_layer_encoder)
        self.decoder = NPDecoder(hidden_dims=hidden_dims, hidden_layers=hidden_layers_decoder)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, c_points, target_x, target_y=None):
        # Encode the context data

        c_mean, c_std, c_dist = self.encoder(c_points)

        if self.training:
            t_points = torch.cat([target_x, target_y], 2)
            t_mean, t_log_std, t_dist = self.encoder(t_points)

            z = c_dist.rsample()

            # Decode it using the query
            mu, log_sigma, y_dist = self.decoder(z, target_x)

            p_y_zx = y_dist.log_prob(target_y)

            y_zx_mean = p_y_zx.mean(dim=0).sum()
            kl_div = kl_divergence(t_dist, c_dist).mean(dim=0).sum()
            loss = -y_zx_mean + kl_div
            return loss, mu, log_sigma, (y_zx_mean.detach(), kl_div.detach()), z

        else:
            loss = 0
            z = c_dist.rsample()

            # Decode it using the query
            mu, log_sigma, y_dist = self.decoder(z, target_x)

            return loss, mu, log_sigma, None, z

    def train_process(self, c_points, t_points_X, t_points_y):
        self.train()
        self.optimizer.zero_grad()

        # Forward pass
        loss, mu, log_sigma, metrics, z = self(c_points,
                                            t_points_X,
                                            t_points_y)

        loss.backward()
        self.optimizer.step()

        return loss, mu, log_sigma, metrics, z
