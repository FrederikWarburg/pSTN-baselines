

def compute_loss(architecture, model, data, target, epoch, image_indices, device, batch_idx, M):
    if architecture[:5] == 'P_STN':
        S = model_specifications['train_S']
        batch_size = data.shape[0]
        transformed_images, output_samples, sigma_q, mu_q = model(data, epoch)
        batch_loss = 0
        for image in range(batch_size):
            # CLASSIFICATION LOSS
            probs = output_samples[image * S: (image + 1) * S, :]
            probs = probs.view([1, S, model.parameter_dict['nr_target_classes']])
            aggregated_probs = probs.mean(dim=1)
            data_loss = F.nll_loss(aggregated_probs, target[image].unsqueeze(0), reduction='none')
            batch_loss += data_loss

        # KL LOSS
        mu_p = torch.zeros_like(mu_q, device=model.device)
        if trafos == 'affine': # identity is parametrized differently for affine and diffeo trafos
            mu_p[:, 1] = 1

        # MANUAL KL
        #KL_constants = (-2 + 2*(sigma_p**2).log())*batch_size
        #KL_loss = ((1 / (2 * (sigma_p**2))) * (sigma_q**2).sum() +
        #            (1 / (2 * (sigma_p**2))) * (mu_p - mu_q).norm(p=2)**2 -
        #            (1 / 2) * (sigma_q**2).log().sum()) + KL_constants  # added for whole batch, will be normalized below

        # print(sigma_q, sigma_p)
        # print('MANUAL KL:', KL_loss/data.shape[0])

        # initialize distributions to compute KL
        mu_p = mu_p.flatten()
        p = MultivariateNormal(loc=mu_p, scale_tril=sigma_p*torch.eye(len(mu_p), device=device))
        mu_q = mu_q.flatten()
        q = MultivariateNormal(loc=mu_q, scale_tril=sigma_q.flatten()*torch.eye(len(mu_q), device=device))
        KL_loss = kl.kl_divergence(q, p)


        # RECONSTRUCTION LOSS
        reconstruction_loss = 0
        # for image in range(batch_size):
            # CLASSIFICATION LOSS
        #    I_obs = data[image * S: (image + 1) * S, :, :, :]
        #    I_z = transformed_images[image * S: (image + 1) * S, :, :, :]
        #    p_I_obs = MultivariateNormal(loc=I_obs, scale_tril=sigma_M*torch.eye(I_obs.shape[-1], device=device))
            # T_inverse_I_z = TODO!!!
        #    reconstruction_loss += p_I_obs.log_prob(I_obs)
        #    print(reconstruction_loss.shape)


        # normalize per batch size
        KL_loss = KL_loss / data.shape[0]
        data_loss = batch_loss / data.shape[0]
        reconstruction_loss = reconstruction_loss / data.shape[0]
        loss = data_loss + KL_loss + reconstruction_loss  # THIS IS THE ELBO NOW!
        loginfo = (data_loss, KL_loss, sigma_q)
        #print('reconstruction_loss:', reconstruction_loss)
        #exit()

    if architecture in ['STN', 'pure_CNN'] or architecture.startswith('STN'):
        output = model(data, epoch)
        batch_loss = F.nll_loss(output, target, reduction='sum')
        loss = batch_loss / data.shape[0]  # loss is just the data loss here
        loginfo = (loss, torch.zeros([1]), torch.zeros([1]))

    return loss, loginfo