import torch

from models.ofa.ofa import OFAModel, ofa_base_architecture
from fairseq.distributed import fsdp_wrap

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)



class TrajectoryOFAModel(OFAModel):
    def __init__(self,  args, task):

        # make sure all arguments are present in older models
        ofa_base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = self.build_embedding(args, src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = self.build_embedding(args, src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            decoder_embed_tokens = self.build_embedding(args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path)
        if getattr(args, "freeze_encoder_embedding", False):
            encoder_embed_tokens.weight.requires_grad = False
        if getattr(args, "freeze_decoder_embedding", False):
            decoder_embed_tokens.weight.requires_grad = False
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = self.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = self.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)

        super().__init__(args, encoder, decoder)
        return

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        src_tokens, src_lengths, prev_output_tokens = self.quantize(states, actions, rewards, returns_to_go, timesteps, attention_mask)

        x, extra = super().forward(src_tokens, src_lengths, prev_output_tokens)

        # get predictions
        return_preds = self.predict_return(x[:, 2])  # predict next return given state and action
        state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        return state_preds, action_preds, return_preds

    def quantize(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)

        return src_tokens, src_lengths, prev_output_tokens

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat([torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states], dim=1).to(dtype=torch.float32)
            actions = torch.cat([torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=actions.device), actions], dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat([torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat([torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]
